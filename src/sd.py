from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, CLIPTextModel

from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers import AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup

from accelerate import Accelerator

from peft import LoraConfig, get_peft_model

from dataset import MasksDataset


class StableDiffusion(nn.Module):
    def __init__(self, vae, unet, text_encoder, tokenizer, noise_scheduler):
        super().__init__()

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.unet = unet

        self.noise_scheduler = noise_scheduler

    def forward(self, images, captions, conditions):
        self.text_encoder = self.text_encoder.to(dtype=torch.float16)
        self.vae = self.vae.to(dtype=torch.float16)
        self.unet = self.unet.to(dtype=torch.float16)


        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        captions = tokenize_caption(captions, self.tokenizer)
        text_embeddings = self.text_encoder(captions.to(images.device), return_dict=False)[0]

        noise = torch.randn_like(latents).to(device=images.device, dtype=torch.float16)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                                  device=images.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeddings,
            return_dict=True
        )['sample']

        return noise_pred, noise


def tokenize_caption(caption, tokenizer):
    return tokenizer(caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids


def training(model, train_dataloader, criterion, optimizer, lr_scheduler, accelerator, output_dir, num_epochs=3):
    for epoch in range(num_epochs):
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            images = batch["pixel_values"].to(accelerator.device, dtype=torch.float16).permute(0, 3, 1, 2)
            conditions = batch["conditioning_pixel_values"].to(accelerator.device, dtype=torch.float16).permute(0, 3, 1, 2)
            captions = batch["caption"]

            noise_pred, noise = model(images, captions, conditions)

            loss = criterion(noise_pred, noise)
            accelerator.backward(loss)

            # loss.backward()
            model.float()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % 1 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")
                # print(noise_pred)

        model.unet.save_pretrained(f"{output_dir}/checkpoint-epoch_{epoch}/unet")
        torch.save(model.state_dict(), f"{output_dir}/checkpoint-epoch_{epoch}/model")

        print(f"Epoch {epoch} completed and checkpoint saved.")


def main():
    pretrained_model_name = "runwayml/stable-diffusion-v1-5"
    output_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/ControlNet/out/stable_diffusion"
    num_epochs = 1
    learning_rate = 5e-6
    batch_size = 6
    image_size = 64

    accelerator = Accelerator()
    lora_config = LoraConfig(
        r=4,
        lora_alpha=32,
        lora_dropout=0.1
    )

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
    unet = get_peft_model(unet, lora_config)
    criterion = nn.MSELoss()

    noise_scheduler = DDPMScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000)

    train_images_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/data/bubbles_split/test/images"
    train_masks_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/data/bubbles_split/test/masks"

    # train_images_dir = r"F:\Internship\ITMO_ML\data\weakly_segmented\bubbles_split\valid\images"
    # train_masks_dir = r"F:\Internship\ITMO_ML\data\weakly_segmented\bubbles_split\valid\masks"

    train_dataset = MasksDataset(train_images_dir, train_masks_dir, width=image_size, height=image_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    device = accelerator.device

    text_encoder.eval()
    vae.eval()
    unet.train()

    model = StableDiffusion(vae, unet, text_encoder, tokenizer, noise_scheduler)
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
    model.to(dtype=torch.float16)

    training(model, train_dataloader, criterion, optimizer, lr_scheduler, accelerator, output_dir, num_epochs=num_epochs)

    model.unet.save_pretrained(f"{output_dir}/final_model_sd")
    torch.save(model.state_dict(), f"{output_dir}/final_model.pth")

    print("Training complete! Model saved to:", output_dir)


if __name__ == "__main__":
    main()