import matplotlib.pyplot as plt
import cv2
import numpy as np

import torch

from diffusers.utils import load_image, make_image_grid
from diffusers import StableDiffusionControlNetPipeline, UNet2DConditionModel, ControlNetModel


def load_controlnet_pipeline(controlnet_weights):
    controlnet = ControlNetModel.from_pretrained(controlnet_weights, torch_dtype=torch.float16, use_safetensors=True)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
    )

    return pipe


def get_conrolnet_pipeline(controlnet_model):
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet_model, torch_dtype=torch.float16, use_safetensors=True
    )
    return pipe


def controlnet_inference(
        controlnet_pipe, mask,
        prompt="bubbles", negative_prompt="low quality, blurry", num_inference_steps=100
):
    output = controlnet_pipe(
        prompt,
        mask,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
    ).images[0]

    return output


def main():
    controlnet_weights = r"/Users/egorprokopov/Documents/Work/ITMO_ML/ControlNet/out/controlnet/checkpoint-epoch_8/controlnet"
    unet_weights = r"/Users/egorprokopov/Documents/Work/ITMO_ML/ControlNet/out/controlnet/checkpoint-epoch_8/unet"
    pipe = load_controlnet_pipeline(controlnet_weights).to("mps")

    prompt = "bubbles"
    # prompt = "froth flotation, bubbles, gray bubbles, many big bubbles"

    mask_path = r"/Users/egorprokopov/Documents/Work/ITMO_ML/ControlNet/data/test/masks/F5_1_1_1.ts-frames_frame-0.png"
    mask_image = load_image(mask_path)

    output = controlnet_inference(pipe, mask_image, prompt)

    cv2.imwrite(
        r"/Users/egorprokopov/Documents/Work/ITMO_ML/ControlNet/data/out/test1.png",
        np.array(make_image_grid([mask_image, output], rows=1, cols=2))
    )


if __name__ == "__main__":
    main()