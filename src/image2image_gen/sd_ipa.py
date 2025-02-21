from typing import List

import torch
import torch.nn as nn

import lightning.pytorch as pl
from PIL import Image
from accelerate import Accelerator
from lightning.pytorch import Trainer
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, \
    DDPMScheduler, DDIMScheduler
from transformers import AutoTokenizer, CLIPTextModel, CLIPVisionModelWithProjection

from src.data.img2img_data_module import ControlNetDataModule
from src.image2image_gen.common.base_diffusion_module import BaseDiffusionLightningModule
from src.image2image_gen.common.callbacks import GenerateImagesCallback, TrainingLossCallback


class AttnProcessor(nn.Module):
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class IPAttnProcessor(nn.Module):
    r"""
    Attention processor for IP-Adapater.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = attn.head_to_batch_dim(ip_key)
        ip_value = attn.head_to_batch_dim(ip_value)

        ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
        self.attn_map = ip_attention_probs
        ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

        hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


def init_adapter_modules(unet):
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    return adapter_modules


class IPAdapter(nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds['image_embeds'].image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)

        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")


class IPAdapterLightningModule(BaseDiffusionLightningModule):
    def __init__(self, vae, unet, text_encoder, tokenizer, image_encoder, noise_scheduler, lr, num_training_steps):
        super().__init__(vae, text_encoder, tokenizer, noise_scheduler, lr, num_training_steps)

        adapter_modules = init_adapter_modules(unet)
        self.image_proj = ImageProjModel(
            cross_attention_dim=unet.config.cross_attention_dim,
            clip_embeddings_dim=image_encoder.config.projection_dim,
            clip_extra_context_tokens=4,
        )
        self.ip_adapter = IPAdapter(
            unet=unet,
            image_proj_model=self.image_proj,
            adapter_modules=adapter_modules
        )
        self.image_encoder = image_encoder
        self.image_encoder.requires_grad_(False)

        inference_noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00095,
            beta_end=0.016,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        self.pipe = StableDiffusionPipeline(
            vae=vae,
            unet=unet,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=inference_noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
        )

    def forward(self, images, captions, conditions):
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        text_embeddings = self.encode_text(captions)

        image_embeds = self.image_encoder(conditions)

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (images.shape[0],), device=self.device)
        noisy_latents = self.add_noise(latents, noise, timesteps)

        noise_pred = self.ip_adapter(
            noisy_latents,
            timesteps,
            text_embeddings,
            {"image_embeds": image_embeds}
        )

        return noise_pred, noise

    def get_generator(self, seed, device):
        if seed is not None:
            if isinstance(seed, list):
                generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
            else:
                generator = torch.Generator(device).manual_seed(seed)
        else:
            generator = None

        return generator

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    @torch.no_grad()
    def inference(
            self, pil_image, prompt, clip_image_embeds=None,
            negative_prompt="best quality regular shapes even surface round metallicslippery polished smooth",
            num_inference_steps=100, guidance_scale=7.5, scale=1.0, num_samples=4, seed=None
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)


        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = self.get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images

        return images


if __name__ == "__main__":
    pretrained_model_name = "runwayml/stable-diffusion-v1-5"
    output_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/ControlNet/out/sd_ipa"
    num_epochs = 2
    learning_rate = 5e-5
    batch_size = 6
    image_size = 224
    log_step = 5

    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name,
        subfolder="tokenizer",
        use_fast=False
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name,
        subfolder="text_encoder"
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name,
        subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name,
        subfolder="unet"
    )
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    )

    noise_scheduler = DDPMScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000)
    # noise_scheduler = EulerDiscreteScheduler.from_pretrained(pretrained_model_name, subfolder="scheduler")

    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    train_images_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/data/bubbles/bubbles_split/train/images"
    train_masks_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/data/bubbles/bubbles_split/train/masks"

    val_images_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/data/bubbles/bubbles_split/valid/images"
    val_masks_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/data/bubbles/bubbles_split/valid/masks"

    data_module = ControlNetDataModule(
        train_images_dir, train_masks_dir, val_images_dir, val_masks_dir,
        batch_size=batch_size, image_size=image_size
    )

    num_training_steps = num_epochs * len(data_module.train_dataloader())
    model = IPAdapterLightningModule(
        vae, unet, text_encoder, tokenizer, image_encoder, noise_scheduler, learning_rate, num_training_steps
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
