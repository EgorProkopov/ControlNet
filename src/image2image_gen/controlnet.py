import torch
import torch.nn as nn


from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel, ControlNetModel, \
    StableDiffusionControlNetPipeline

from lightning.pytorch import Trainer
from accelerate import Accelerator

from src.data.img2img_data_module import ControlNetDataModule
from src.controlnet.controlnet_lightning import TrainingLossCallback
from src.image2image_gen.common.base_diffusion_module import BaseDiffusionLightningModule
from src.image2image_gen.common.callbacks import GenerateImagesCallback


class ControlNetLightningModule(BaseDiffusionLightningModule):
    def __init__(self, vae, unet, controlnet, text_encoder, tokenizer, noise_scheduler, accelerator, lr, num_training_steps):
        super().__init__(vae, unet, text_encoder, tokenizer, noise_scheduler, lr, num_training_steps)
        self.controlnet = controlnet
        self.vae = vae
        self.unet = unet
        self.criterion = nn.MSELoss()

        self.accelerator = accelerator

    def forward(self, images, captions, conditions):
        latents = self.vae.encode(images).latent_dist.sample() * self.vae.config.scaling_factor
        text_embeddings = self.encode_text(captions)

        noise = torch.randn_like(latents).to(images.device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                                  device=images.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

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

    @torch.no_grad()
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
        # pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        pipeline.to(device)
        pipeline.set_progress_bar_config(disable=True)


        generated_images = pipeline(
            prompt=captions,
            image=conditions,
            num_inference_steps=num_inference_steps,
        ).images
        return generated_images


if __name__ == "__main__":
    pretrained_model_name = "runwayml/stable-diffusion-v1-5"
    output_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/ControlNet/out/controlnet"
    num_epochs = 6
    learning_rate = 5e-5
    batch_size = 1
    image_size = 512
    log_step = 100

    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name, subfolder="text_encoder")
    text_encoder.requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(pretrained_model_name, subfolder="vae")
    vae.requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name, subfolder="unet")
    unet.requires_grad_(False)
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny").train()
    noise_scheduler = DDPMScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000)
    # noise_scheduler = EulerDiscreteScheduler.from_pretrained(pretrained_model_name, subfolder="scheduler")

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
        callbacks=[log_callback, loss_callback]
    )

    trainer.fit(model, datamodule=data_module)
    print(f"Training complete! Model saved to: {output_dir}")
