import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler, EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL


def load_pipeline(checkpoint_path, base_model="runwayml/stable-diffusion-v1-5"):
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")

    unet = UNet2DConditionModel.from_pretrained(checkpoint_path)
    # unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    # noise_scheduler = DDPMScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000)
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(base_model, subfolder="scheduler")

    pipe = StableDiffusionPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False
    )
    pipe.unet.eval()
    pipe.to("mps")
    return pipe


def generate_image(pipe, prompt, num_inference_steps=50, guidance_scale=7.5):
    image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    return image


if __name__ == "__main__":
    checkpoint_path = r"/Users/egorprokopov/Documents/Work/ITMO_ML/ControlNet/out/stable_diffusion/checkpoint-epoch_3/unet"
    pipe = load_pipeline(checkpoint_path)
    #image = generate_image(pipe, "froth flotation bubbles")
    image = generate_image(pipe, "froth flotation bubbles", num_inference_steps=250)
    image.show()
