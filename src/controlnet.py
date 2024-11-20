from tqdm.auto import tqdm

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from transformers import AutoTokenizer, CLIPTextModel

from diffusers import DDPMScheduler, UNet2DConditionModel, ControlNetModel
from diffusers import AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup

from accelerate import Accelerator


from dataset import MasksDataset


class ControlNet(nn.Module):
    def __init__(self, vae, unet, controlnet, text_encoder, tokenizer, noise_scheduler):
        super().__init__()

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.controlnet = controlnet
        self.vae = vae
        self.unet = unet

        self.noise_scheduler = noise_scheduler

    def forward(self, images, captions, conditions):
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        captions = tokenize_caption(captions, self.tokenizer)
        text_embeddings = self.text_encoder(captions.to(images.device), return_dict=False)[0]

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,),
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
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=True
        )['sample']

        return noise_pred, noise


def tokenize_caption(caption, tokenizer):
    return tokenizer(caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids


def train(model, train_dataloader, criterion, optimizer, lr_scheduler, accelerator):
    for epoch in range(num_epochs):
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # Move data to device
            images = batch["pixel_values"].to(accelerator.device).permute(0, 3, 1, 2)
            conditions = batch["conditioning_pixel_values"].to(accelerator.device).permute(0, 3, 1, 2)
            captions = batch["caption"]

            noise_pred, noise = model(images, captions, conditions)

            # Compute loss
            loss = criterion(noise_pred, noise)
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Logging
            if step % 50 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

        unet.save_pretrained(f"{output_dir}/checkpoint-{epoch}")
        print(f"Epoch {epoch} completed and checkpoint saved.")


if __name__ == "__main__":
    pretrained_model_name = "runwayml/stable-diffusion-v1-5"
    output_dir = "./stable_diffusion_finetuned"
    num_epochs = 3
    learning_rate = 5e-6
    batch_size = 6
    image_size = 256

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

    train_images_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/data/bubbles_split/test/images"
    train_masks_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/data/bubbles_split/test/masks"

    # train_images_dir = r"F:\Internship\ITMO_ML\data\weakly_segmented\bubbles_split\valid\images"
    # train_masks_dir = r"F:\Internship\ITMO_ML\data\weakly_segmented\bubbles_split\valid\masks"

    train_dataset = MasksDataset(train_images_dir, train_masks_dir, width=image_size, height=image_size)
    # Optionally, preprocess images to resize or normalize to the expected format

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = accelerator.device

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.train()
    controlnet.train()

    model = ControlNet(vae, unet, controlnet, text_encoder, tokenizer, noise_scheduler)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    model, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        [model, train_dataloader, optimizer, lr_scheduler]
    )

    train(model, train_dataloader, criterion, optimizer, lr_scheduler, accelerator)

    unet.save_pretrained(output_dir)
    print("Training complete! Model saved to:", output_dir)
