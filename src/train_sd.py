import torch
import torch.nn as nn

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel
from diffusers import DiffusionPipeline, AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tutorial_dataset import MasksDataset


class StableDiffusion(nn.Module):
    def __init__(self, vae, unet, text_encoder, tokenizer, noise_scheduler):
        super().__init__()

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.unet = unet

        self.noise_scheduler = noise_scheduler

    def forward(self, images, captions):
        # Encode image
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # Encode text
        captions = tokenize_caption(captions, tokenizer)
        text_embeddings = text_encoder(captions.to(device)).last_hidden_state

        # Sample noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,),
                                  device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict noise
        noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample

        return noise_pred, noise


def tokenize_caption(caption, tokenizer):
    return tokenizer(caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids


def train(model, train_dataloader, criterion, optimizer, lr_scheduler):
    for epoch in range(num_epochs):
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # Move data to device
            images = batch["pixel_values"].to(device).permute(0, 3, 1, 2)
            captions = batch["caption"]

            noise_pred, noise = model(images, captions)

            # Compute loss
            loss = criterion(noise_pred, noise)
            loss.backward()

            # Optimizer step
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Logging
            if step % 50 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

        # Save checkpoint
        unet.save_pretrained(f"{output_dir}/checkpoint-{epoch}")
        print(f"Epoch {epoch} completed and checkpoint saved.")


if __name__ == "__main__":
    # Configuration
    pretrained_model_name = "stabilityai/stable-diffusion-2-1-base"  # Pretrained Stable Diffusion model
    output_dir = "./stable_diffusion_finetuned"
    num_epochs = 3
    learning_rate = 5e-6
    batch_size = 2
    image_size = 512  # Adjust image size to model specs, 512x512 for SD v1.4

    # 1. Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
        subfolder="tokenizer",
        use_fast=False,
    )

    text_encoder = CLIPTextModel.from_pretrained(
       "stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", subfolder="unet"
    )

    criterion = nn.MSELoss()

    # Load DDPM scheduler for training
    noise_scheduler = DDPMScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000)

    # 2. Load the dataset
    train_images_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/data/bubbles_split/test/images"
    train_masks_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/data/bubbles_split/test/masks"

    # train_images_dir = r"F:\Internship\ITMO_ML\data\weakly_segmented\bubbles_split\valid\images"
    # train_masks_dir = r"F:\Internship\ITMO_ML\data\weakly_segmented\bubbles_split\valid\masks"

    train_dataset = MasksDataset(train_images_dir, train_masks_dir, width=image_size, height=image_size)
    # Optionally, preprocess images to resize or normalize to the expected format

    # 3. Define DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 4. Define optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    # 5. Training Loop
    device = "mps"

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.train()

    model = StableDiffusion(vae, unet, text_encoder, tokenizer, noise_scheduler)

    model.to(device)

    train(model, train_dataloader, criterion, optimizer, lr_scheduler)

    # 6. Save the final model
    unet.save_pretrained(output_dir)
    print("Training complete! Model saved to:", output_dir)