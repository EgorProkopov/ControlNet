import os
import numpy as np
from tqdm.auto import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms import ToTensor, ToPILImage

from transformers import AutoTokenizer, CLIPTextModel

from diffusers import DDPMScheduler, UNet2DConditionModel, ControlNetModel, StableDiffusionControlNetPipeline
from diffusers import AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid

from accelerate import Accelerator

from src.dataset import MasksDataset
from src.controlnet.controlnet_inference import load_controlnet_pipeline, controlnet_inference
from src.image_generation import generate_images


class ControlNet(nn.Module):
    def __init__(self, vae, unet, controlnet, text_encoder, tokenizer, noise_scheduler, device=None, dtype=torch.float16):
        super().__init__()

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.controlnet = controlnet
        self.vae = vae
        self.unet = unet

        self.noise_scheduler = noise_scheduler

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        self.dtype = dtype

    def inference(self, captions, conditions, num_inference_steps=100):
        pipeline = StableDiffusionControlNetPipeline(
            vae=self.vae.to(dtype=self.dtype),
            text_encoder=self.text_encoder.to(dtype=self.dtype),
            tokenizer=self.tokenizer,
            unet=self.unet.to(dtype=self.dtype),
            controlnet=self.controlnet.to(dtype=self.dtype),
            scheduler=self.noise_scheduler,
            safety_checker=None,
            feature_extractor=None
        )
        pipeline.to(self.device)

        generated_images = pipeline(
            prompt=captions,
            image=conditions,
            num_inference_steps=num_inference_steps,
        ).images
        return generated_images

    def forward(self, images, captions, conditions):
        self.text_encoder = self.text_encoder.to(dtype=self.dtype)
        self.vae = self.vae.to(dtype=self.dtype)
        self.controlnet = self.controlnet.to(dtype=self.dtype)
        self.unet = self.unet.to(dtype=self.dtype)

        images = images.to(device=self.device, dtype=self.dtype)
        conditions = conditions.to(device=self.device, dtype=self.dtype)

        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        captions = tokenize_caption(captions, self.tokenizer)
        text_embeddings = self.text_encoder(captions.to(self.device), return_dict=False)[0]

        noise = torch.randn_like(latents).to(device=self.device, dtype=self.dtype)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                                  device=self.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents.to(dtype=self.dtype), noise, timesteps)

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents.to(dtype=self.dtype),
            timesteps,
            encoder_hidden_states=text_embeddings.to(dtype=self.dtype),
            controlnet_cond=conditions.to(dtype=self.dtype),
            return_dict=False,
        )

        noise_pred = self.unet(
            noisy_latents.to(dtype=self.dtype),
            timesteps,
            encoder_hidden_states=text_embeddings.to(dtype=self.dtype),
            down_block_additional_residuals=[
                down_block_res_sample.to(dtype=self.dtype) for down_block_res_sample in down_block_res_samples
            ],
            mid_block_additional_residual=mid_block_res_sample.to(dtype=self.dtype),
            return_dict=True
        )['sample']

        return noise_pred, noise


def tokenize_caption(caption, tokenizer):
    return tokenizer(caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids


def training(
        model, train_dataloader, val_dataloader, criterion, optimizer, lr_scheduler,
        accelerator, output_dir, log_dir, num_epochs=3
):
    global_step = 0
    for epoch in range(num_epochs):
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            images = batch["pixel_values"].permute(0, 3, 1, 2)
            conditions = batch["conditioning_pixel_values"].permute(0, 3, 1, 2)
            captions = batch["caption"]

            noise_pred, noise = model(images, captions, conditions)

            loss = criterion(noise_pred, noise)
            accelerator.backward(loss)

            #loss.backward()
            model.float()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

            if step % 1000 == 0:
                for batch in val_dataloader:
                    generate_images(model, batch, global_step, log_dir)
                    break

            global_step += 1

        model.controlnet.save_pretrained(f"{output_dir}/checkpoint_step_{global_step}")

        print(f"Epoch {epoch} completed and checkpoint saved.")


def main():
    pretrained_model_name = "runwayml/stable-diffusion-v1-5"
    output_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/ControlNet/out/controlnet/experiment/checkpoints"
    log_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/ControlNet/out/controlnet/experiment/logs"
    num_epochs = 10
    learning_rate = 5e-6
    batch_size = 6
    image_size = 128

    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
        subfolder="tokenizer",
        use_fast=False,
    )

    text_encoder = CLIPTextModel.from_pretrained(
       "runwayml/stable-diffusion-v1-5", subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet"
    )
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")

    criterion = nn.MSELoss()

    noise_scheduler = DDPMScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000)

    train_images_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/CTCI/data/bubbles_split/train/images"
    train_masks_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/CTCI/data/bubbles_split/train/images_masks"

    val_images_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/CTCI/data/bubbles_split/valid/images"
    val_masks_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/CTCI/data/bubbles_split/valid/images_masks"

    train_dataset = MasksDataset(train_images_dir, train_masks_dir, width=image_size, height=image_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = MasksDataset(val_images_dir, val_masks_dir, width=image_size, height=image_size)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    device = accelerator.device

    text_encoder.eval()
    vae.eval()
    unet.eval()
    controlnet.train()

    model = ControlNet(vae, unet, controlnet, text_encoder, tokenizer, noise_scheduler)
    model.to(device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    # model, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(
    #     [model, train_dataloader, optimizer, lr_scheduler]
    # )
    model.vae.to(dtype=torch.float32)
    model.text_encoder.to(dtype=torch.float32)
    model.unet.to(dtype=torch.float16)
    model.controlnet.to(dtype=torch.float16)

    training(
        model, train_dataloader, val_dataloader, criterion, optimizer, lr_scheduler,
        accelerator, output_dir, log_dir, num_epochs=num_epochs
    )

    model.unet.save_pretrained(f"{output_dir}/final_model_controlnet")
    model.controlnet.save_pretrained(f"{output_dir}/final_model_unet")
    torch.save(model.state_dict(), f"{output_dir}/final_model.pth")

    print("Training complete! Model saved to:", output_dir)


if __name__ == "__main__":
    main()