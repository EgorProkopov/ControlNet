import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.distutils.lib2def import output_def

from torchvision.transforms import ToTensor, ToPILImage
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import lightning.pytorch as pl

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel, ControlNetModel, \
    StableDiffusionControlNetPipeline, EulerDiscreteScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid

from lightning.pytorch import Trainer
from accelerate import Accelerator

from src.dataset import MasksDataset


def tokenize_caption(caption, tokenizer):
    return tokenizer(caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids


class ControlNetLightningModule(pl.LightningModule):
    def __init__(self, vae, unet, controlnet, text_encoder, tokenizer, noise_scheduler, accelerator, lr, num_training_steps):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.controlnet = controlnet
        self.vae = vae
        self.unet = unet
        self.noise_scheduler = noise_scheduler
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.num_training_steps = num_training_steps

        self.accelerator = accelerator

    def forward(self, images, captions, conditions):
        # Encoding

        # self.vae = self.vae.to(dtype=torch.float16)
        # self.unet = self.unet.to(dtype=torch.float16)
        # self.controlnet = self.controlnet.to(dtype=torch.float16)
        # self.text_encoder = self.text_encoder.to(dtype=torch.float16)
        #
        # self.images = images.to(dtype=torch.float16)
        # self.conditions = conditions.to(dtype=torch.float16)

        latents = self.vae.encode(images).latent_dist.sample() * self.vae.config.scaling_factor
        captions = tokenize_caption(captions, self.tokenizer)
        text_embeddings = self.text_encoder(captions.to(images.device), return_dict=False)[0]

        # Add noise
        noise = torch.randn_like(latents).to(images.device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                                  device=images.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # ControlNet
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeddings,
            controlnet_cond=conditions,
            return_dict=False,
        )

        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeddings,
            down_block_additional_residuals=[res.to(dtype=torch.float16) for res in down_block_res_samples],
            mid_block_additional_residual=mid_block_res_sample.to(dtype=torch.float16),
            return_dict=True
        )['sample']

        return noise_pred, noise

    def inference(self, captions, conditions, num_inference_steps=100):
        pipeline = StableDiffusionControlNetPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=self.controlnet,
            scheduler=self.noise_scheduler,
            requires_safety_checker = False,
            safety_checker=None,
            feature_extractor=None
        )
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        pipeline.to(device)

        generated_images = pipeline(
            prompt=captions,
            image=conditions,
            num_inference_steps=num_inference_steps,
        ).images
        return generated_images

    def training_step(self, batch, batch_idx):
        images = batch["pixel_values"].to(self.device).permute(0, 3, 1, 2)
        conditions = batch["conditioning_pixel_values"].to(self.device).permute(0, 3, 1, 2)
        captions = batch["caption"]

        noise_pred, noise = self(images, captions, conditions)
        loss = self.criterion(noise_pred, noise)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["pixel_values"].to(self.device).permute(0, 3, 1, 2)
        conditions = batch["conditioning_pixel_values"].to(self.device).permute(0, 3, 1, 2)
        captions = batch["caption"]

        noise_pred, noise = self(images, captions, conditions)
        loss = self.criterion(noise_pred, noise)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=self.num_training_steps,
        )
        # optimizer = self.accelerator.prepare(optimizer)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]


class ControlNetDataModule(pl.LightningDataModule):
    def __init__(self, train_images_dir, train_masks_dir, val_images_dir, val_masks_dir, batch_size, image_size, num_workers=6):
        super().__init__()
        self.train_images_dir = train_images_dir
        self.train_masks_dir = train_masks_dir
        self.val_images_dir = val_images_dir
        self.val_masks_dir = val_masks_dir
        self.batch_size = batch_size
        self.image_size = image_size

        self.num_workers = num_workers

        self.setup()

    def setup(self, stage=None):
        self.train_dataset = MasksDataset(self.train_images_dir, self.train_masks_dir, width=self.image_size, height=self.image_size)
        self.val_dataset = MasksDataset(self.val_images_dir, self.val_masks_dir, width=self.image_size, height=self.image_size)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=self.num_workers, shuffle=False)


class GenerateImagesCallback(pl.Callback):
    def __init__(self, log_dir, log_every_n_steps=1000):
        super().__init__()
        self.log_dir = log_dir
        self.log_every_n_steps = log_every_n_steps
        self.to_tensor = ToTensor()
        self.to_pil = ToPILImage()
        os.makedirs(log_dir, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step

        if global_step % self.log_every_n_steps == 0 and global_step > 0:
            pl_module.eval()
            with torch.no_grad():
                captions = batch["caption"]
                conditions = batch["conditioning_pixel_values"].permute(0, 3, 1, 2)
                captions = captions
                conditions = conditions

                conditions = conditions.to(pl_module.device)
                generated_images = pl_module.inference(captions=captions, conditions=conditions)
                generated_images = np.array(generated_images)

                for i, img in enumerate(generated_images):
                    img_path = os.path.join(self.log_dir, f"step_{global_step}_image_{i}.png")

                    pil_conditions = self.to_pil(conditions[i])
                    pil_generated_img = self.to_pil(img)
                    grid = make_image_grid([pil_conditions, pil_generated_img], rows=1, cols=2)

                    save_image(self.to_tensor(grid).unsqueeze(0), img_path)

                # if trainer.logger:
                #     trainer.logger.experiment.add_image(
                #         f"generated_images/step_{global_step}",
                #         self.to_tensor(generated_images[0]),
                #         global_step
                #     )

                print(f"Logged generated images at step {global_step}")
            pl_module.train()


class TrainingLossCallback(pl.Callback):
    def __init__(self, log_dir, log_every_n_steps=1000):
        self.train_losses = []
        self.val_losses = []
        self.log_dir = log_dir
        self.log_every_n_steps = log_every_n_steps
        os.makedirs(log_dir, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step

        if global_step % self.log_every_n_steps == 0:
            train_loss = trainer.callback_metrics['train_loss'].item()
            self.train_losses.append(train_loss)
            val_loss = trainer.callback_metrics['val_loss'].item() if 'val_loss' in trainer.callback_metrics else None
            if val_loss:
                self.val_losses.append(val_loss)

            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Training Loss')
            if self.val_losses:
                plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label='Validation Loss')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.legend()
            plt.title(f'Training and Validation Loss at Step {global_step}')
            plt.savefig(os.path.join(self.log_dir, f'loss_plot_step_{global_step}.png'))
            plt.close()

    def on_train_end(self, trainer, pl_module):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label='Validation Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Final Training and Validation Loss')
        plt.savefig(os.path.join(self.log_dir, 'final_loss_plot.png'))
        plt.close()


def main():
    pretrained_model_name = "runwayml/stable-diffusion-v1-5"
    # output_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/ControlNet/out/controlnet"
    output_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/ControlNet/out/controlnet"
    num_epochs = 6
    learning_rate = 5e-5
    batch_size = 6
    image_size = 256
    log_step = 1000

    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name, subfolder="text_encoder")
    text_encoder.requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(pretrained_model_name, subfolder="vae")
    vae.requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name, subfolder="unet")
    unet.requires_grad_(False)
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny").train()
    # noise_scheduler = DDPMScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000)
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(pretrained_model_name, subfolder="scheduler")

    # train_images_dir = r"F:\ITMO_ML\data\bubbles\weakly_segmented\bubbles_split\train\images"
    # train_masks_dir = r"F:\ITMO_ML\data\bubbles\weakly_segmented\bubbles_split\train\masks"
    #
    # val_images_dir = r"F:\ITMO_ML\data\bubbles\weakly_segmented\bubbles_split\valid\images"
    # val_masks_dir = r"F:\ITMO_ML\data\bubbles\weakly_segmented\bubbles_split\valid\masks"

    train_images_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/data/bubbles/bubbles_split/train/images"
    train_masks_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/data/bubbles/bubbles_split/train/masks"

    val_images_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/data/bubbles/bubbles_split/valid/images"
    val_masks_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/data/bubbles/bubbles_split/valid/masks"

    data_module = ControlNetDataModule(
        train_images_dir, train_masks_dir, val_images_dir, val_masks_dir,
        batch_size=batch_size, image_size=image_size
    )

    num_training_steps = num_epochs * len(data_module.train_dataloader())
    model = ControlNetLightningModule(
        vae, unet, controlnet, text_encoder, tokenizer, noise_scheduler, accelerator,
        lr=learning_rate, num_training_steps=num_training_steps
    )

    model, data_module = accelerator.prepare([model, data_module])

    log_callback = GenerateImagesCallback(
        log_dir="/Users/egorprokopov/Documents/Work/ITMO_ML/ControlNet/out/logs/controlnet/generated_images",
        log_every_n_steps=log_step
    )
    loss_callback = TrainingLossCallback(
        log_dir="/Users/egorprokopov/Documents/Work/ITMO_ML/ControlNet/out/logs/controlnet",
        log_every_n_steps=log_step
    )

    trainer = Trainer(
        max_epochs=num_epochs,
        accelerator='mps',
        devices=accelerator.num_processes,
        precision='bf16',
        strategy="auto",
        default_root_dir=output_dir,
        log_every_n_steps=log_step,
        # accumulate_grad_batches=2,
        callbacks=[loss_callback]
    )

    trainer.fit(model, datamodule=data_module)
    print(f"Training complete! Model saved to: {output_dir}")


if __name__ == "__main__":
    main()