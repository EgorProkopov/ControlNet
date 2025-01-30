import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, AutoencoderKL, \
    UNet2DConditionModel, EulerDiscreteScheduler, DDPMScheduler
from transformers import CLIPImageProcessor, AutoTokenizer, CLIPTextModel
from accelerate import Accelerator

from src.controlnet.controlnet_lightning import ControlNetLightningModule


def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = Image.fromarray(image)
    return image


if __name__ == '__main__':
    learning_rate = 5e-5
    num_training_steps = 1000

    pretrained_model_name = "runwayml/stable-diffusion-v1-5"
    ckpt_path = r"/Users/egorprokopov/Documents/Work/ITMO_ML/ControlNet/out/controlnet/lightning_logs/version_0/checkpoints/epoch=5-step=13614.ckpt"

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
    # noise_scheduler = EulerDiscreteScheduler.from_pretrained(r"runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    model = ControlNetLightningModule(
        vae, unet, controlnet, text_encoder, tokenizer, noise_scheduler, accelerator,
        lr=learning_rate, num_training_steps=num_training_steps
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    # model = model.to("mps")

    prompt = "froth flotation bubbles"

    conditions_path = r"/Users/egorprokopov/Documents/Work/ITMO_ML/data/bubbles/bubbles_split/test/masks/F5_1_1_1.ts-frames_frame-30.png"
    conditions = preprocess_image(conditions_path)
    # conditions.show()
    generated_images = model.inference(prompt, conditions, num_inference_steps=5)
    print(np.array(generated_images[0]).max())

