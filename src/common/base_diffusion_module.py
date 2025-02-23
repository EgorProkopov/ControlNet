import torch
import lightning.pytorch as pl
from torch.nn import MSELoss
from diffusers.optimization import get_cosine_schedule_with_warmup


class BaseDiffusionLightningModule(pl.LightningModule):
    def __init__(self, vae, unet, text_encoder, tokenizer, noise_scheduler, lr, num_training_steps):
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        self.lr = lr
        self.num_training_steps = num_training_steps
        self.criterion = MSELoss()

    def training_step(self, batch, batch_idx):
        images = batch["pixel_values"].permute(0, 3, 1, 2)
        captions = batch["caption"]

        noise_pred, noise = self(images, captions)
        loss = self.criterion(noise_pred, noise)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["pixel_values"].permute(0, 3, 1, 2)
        captions = batch["caption"]

        noise_pred, noise = self(images, captions)
        loss = self.criterion(noise_pred, noise)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def encode_text(self, captions):
        input_ids = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).input_ids.to(self.device)
        return self.text_encoder(input_ids)[0]

    def add_noise(self, latents, noise, timesteps):
        return self.noise_scheduler.add_noise(latents, noise, timesteps)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=500,
            num_training_steps=self.num_training_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class BaseDiffusionImage2ImageLightningModule(BaseDiffusionLightningModule):
    def __init__(self, vae, unet, text_encoder, tokenizer, noise_scheduler, lr, num_training_steps):
        super().__init__(vae, unet, text_encoder, tokenizer, noise_scheduler, lr, num_training_steps)

    def training_step(self, batch, batch_idx):
        images = batch["pixel_values"].permute(0, 3, 1, 2)
        conditions = batch["conditioning_pixel_values"].permute(0, 3, 1, 2)
        captions = batch["caption"]

        noise_pred, noise = self(images, captions, conditions)
        loss = self.criterion(noise_pred, noise)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["pixel_values"].permute(0, 3, 1, 2)
        conditions = batch["conditioning_pixel_values"].permute(0, 3, 1, 2)
        captions = batch["caption"]

        noise_pred, noise = self(images, captions, conditions)
        loss = self.criterion(noise_pred, noise)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

