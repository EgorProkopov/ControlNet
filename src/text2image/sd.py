from omegaconf import OmegaConf

import torch
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers import AutoencoderKL
from diffusers import StableDiffusionPipeline
from accelerate import Accelerator
from lightning.pytorch import Trainer


from src.common.base_diffusion_module import BaseDiffusionLightningModule
from src.common.callbacks import GenerateImage2ImageCallback, TrainingLossCallback, SaveWeightsCallback, \
    GenerateText2ImageCallback
from src.data.data_modules import Text2ImageDataModule


class StableDiffusionLightningModule(BaseDiffusionLightningModule):
    def __init__(self, vae, unet, text_encoder, tokenizer, noise_scheduler, accelerator, lr, num_training_steps):
        super().__init__(vae, unet, text_encoder, tokenizer, noise_scheduler, lr, num_training_steps)

        self.accelerator = accelerator

    def forward(self, images, captions):
        latents = self.vae.encode(images).latent_dist.sample() * 0.18215
        text_embeddings = self.encode_text(captions)

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                                  device=images.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample

        return noise_pred, noise

    def inference(self, captions, num_inference_steps=50):
        pipeline = StableDiffusionPipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            unet=self.unet,
            scheduler=self.noise_scheduler,
            requires_safety_checker=False,
            safety_checker=None,
            feature_extractor=None
        )
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        pipeline = pipeline.to(device)
        # pipeline.set_progress_bar_config(disable=True)

        generated_images = pipeline(
            prompt=captions,
            num_inference_steps=num_inference_steps,
        ).images
        return generated_images


if __name__ == "__main__":
    config = OmegaConf.load("/Users/egorprokopov/Documents/Work/ITMO_ML/ControlNet/configs/sd_config.yaml")

    pretrained_model_name = config.base_model.pretrained_model_name

    output_dir = config.out_directories.output_dir
    images_logs_dir = config.out_directories.images_logs_dir
    loss_logs_dir = config.out_directories.loss_logs_dir
    weights_logs_dir = config.out_directories.weights_logs_dir

    train_images_dir = config.datasets_dirs.train_images_dir
    val_images_dir = config.datasets_dirs.val_images_dir

    num_epochs = config.train_params.num_epochs
    learning_rate = config.train_params.learning_rate
    batch_size = config.train_params.batch_size
    image_size = config.train_params.image_size
    log_images_step = config.train_params.log_images_step
    log_loss_step = config.train_params.log_loss_step
    log_weights_step = config.train_params.log_weights_step

    device = config.hardware.device
    precision = config.hardware.precision

    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name, subfolder="text_encoder")
    text_encoder.requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(pretrained_model_name, subfolder="vae")
    vae.requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name, subfolder="unet")
    unet.train()

    noise_scheduler = DDPMScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000)

    data_module = Text2ImageDataModule(
        train_images_dir, val_images_dir, caption="froth flotation bubbles",
        batch_size=batch_size, image_size=image_size
    )

    num_training_steps = num_epochs * len(data_module.train_dataloader())
    model = StableDiffusionLightningModule(
        vae, unet, text_encoder, tokenizer, noise_scheduler, accelerator,
        lr=learning_rate, num_training_steps=num_training_steps
    )

    model, data_module = accelerator.prepare([model, data_module])

    log_callback = GenerateText2ImageCallback(
        log_dir=images_logs_dir,
        log_every_n_steps=log_images_step
    )
    loss_callback = TrainingLossCallback(
        log_dir=loss_logs_dir,
        log_every_n_steps=log_loss_step
    )
    save_callback = SaveWeightsCallback(
        log_dir=weights_logs_dir,
        modules_to_save=["unet"],
        log_every_n_steps=log_weights_step
    )

    trainer = Trainer(
        max_epochs=num_epochs,
        accelerator=device,
        devices=accelerator.num_processes,
        precision=precision,
        strategy="auto",
        default_root_dir=output_dir,
        log_every_n_steps=10000,
        # accumulate_grad_batches=2,
        callbacks=[log_callback, loss_callback, save_callback]
    )

    trainer.fit(model, datamodule=data_module)
    print(f"Training complete! Model saved to: {weights_logs_dir}")
