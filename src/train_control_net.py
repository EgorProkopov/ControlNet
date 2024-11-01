import os
from PIL import Image

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms

import diffusers
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler, ControlNetModel, UNet2DConditionModel, AutoencoderKL
from diffusers import UNet2DModel
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
from dataclasses import dataclass

import accelerate

from datasets import load_dataset
from transformers import CLIPTextModel, AutoTokenizer, CLIPTokenizer

from tutorial_dataset import MasksDataset


@dataclass
class TrainingConfig:
    image_size = 128
    train_batch_size = 16
    val_batch_size = 16

    num_epochs = 50
    gradient_accumulation_steps = 2
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = 'no'

    output_dir = "checkpoints/diffusion_test"
    overwrite_output_dir = True

    seed = 239


def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.val_batch_size,
        generator=torch.Generator(device='cpu').manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


def train_loop(config, controlnet, unet, vae, tokenizer, text_encoder, train_dataloader, criterion, optimizer, noise_scheduler, lr_scheduler):
    accelerator = accelerate.Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps = config.gradient_accumulation_steps,
        log_with = "tensorboard",
        project_dir = os.path.join(config.output_dir, "logs")
    )

    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)

        accelerator.init_trackers("train_example")

    controlnet, unet, vae, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, unet, vae, text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    controlnet.to(device=accelerator.device, dtype=torch.float16)
    unet.to(device=accelerator.device, dtype=torch.float16)
    # text_encoder.to(device=accelerator.device, dtype=torch.float16)

    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(train_dataloader)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(progress_bar):
            # Also add unet
            with accelerator.accumulate(controlnet):
                clean_images = batch['jpg'].to(accelerator.device, dtype=torch.float16).permute(0, 3, 1, 2)
                mask_hint = batch['hint'].to(accelerator.device, dtype=torch.float16).permute(0, 3, 1, 2)
                text_prompt = batch['txt']

                text_tokens = tokenizer(
                    text_prompt, padding="max_length",
                    truncation=True, max_length=77, return_tensors="pt"
                ).input_ids

                # print(text_tokens)

                text_embeddings = text_encoder(text_tokens).to(device=accelerator.device, dtype=torch.float16)
                encoder_hidden_states = text_embeddings[0]

                latents = vae.encode(clean_images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,),
                    device=latents.device
                ).long()

                noisy_latents = noise_scheduler.add_noise(
                    latents.float(), noise.float(), timesteps
                ).to(device=accelerator.device, dtype=torch.float16)

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents, timesteps,
                    controlnet_cond=mask_hint,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False
                )


                loss = criterion(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

        # Sample images and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)


if __name__ == '__main__':
    device = torch.device("mps")

    config = TrainingConfig()

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
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.train()
    controlnet.train()

    train_images_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/data/bubbles_split/test/images"
    train_masks_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/data/bubbles_split/test/masks"

    train_dataset = MasksDataset(train_images_dir, train_masks_dir, width=256, height=256)
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=0,
        batch_size=2,
        shuffle=True
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    criterion = nn.MSELoss(reduction="mean").to(device)
    # TODO: add unet parameters
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=1e-4)

    num_epochs = 10

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    train_loop(config, controlnet, unet, vae, tokenizer, text_encoder, train_dataloader, criterion, optimizer, noise_scheduler, lr_scheduler)



