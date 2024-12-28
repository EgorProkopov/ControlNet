from pympler.muppy import print_summary
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, CLIPTextModel, CLIPVisionModelWithProjection
from diffusers import DDPMScheduler, UNet2DConditionModel, ControlNetModel
from diffusers import AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model

from dataset import MasksDataset


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
        ip_tokens = self.image_proj_model(image_embeds['image_embeds'])
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


class StableDiffusionIP(nn.Module):
    def __init__(self, vae, ip_adapter, image_encoder, text_encoder, tokenizer, noise_scheduler):
        super().__init__()

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.vae = vae
        self.ip_adapter = ip_adapter

        self.noise_scheduler = noise_scheduler

    def forward(self, images, captions, conditions):
        self.text_encoder = self.text_encoder.to(dtype=torch.float16)
        self.image_encoder = self.image_encoder.to(dtype=torch.float16)
        self.vae = self.vae.to(dtype=torch.float16)
        self.ip_adapter = self.ip_adapter.to(dtype=torch.float16)

        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        captions = tokenize_caption(captions, self.tokenizer)
        text_embeddings = self.text_encoder(captions.to(images.device), return_dict=False)[0]

        image_embeddings = self.image_encoder(conditions)

        noise = torch.randn_like(latents).to(device=images.device, dtype=torch.float16)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (images.shape[0],),
                                  device=images.device, dtype=torch.float16).long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        noise_pred = self.ip_adapter(noisy_latents, timesteps, text_embeddings, image_embeddings)

        return noise_pred, noise


def tokenize_caption(caption, tokenizer):
    return tokenizer(caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids


def init_adapter_modules(unet, ):
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


def train(model, train_dataloader, criterion, optimizer, lr_scheduler, accelerator, output_dir, num_epochs=3):
    for epoch in range(num_epochs):
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            images = batch["pixel_values"].to(accelerator.device, dtype=torch.float16).permute(0, 3, 1, 2)
            conditions = batch["conditioning_pixel_values"].to(accelerator.device, dtype=torch.float16).permute(0, 3, 1, 2)
            captions = batch["caption"]

            noise_pred, noise = model(images, captions, conditions)
            loss = criterion(noise_pred, noise)
            accelerator.backward(loss)

            model.float()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % 1 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")


        model.ip_adapter.save_pretrained(f"{output_dir}/checkpoint-{epoch}/ip_adapter.pth")
        torch.save(model.state_dict(), f"{output_dir}/checkpoint-{epoch}/model.pth")

        print(f"Epoch {epoch} completed and checkpoint saved.")


if __name__ == "__main__":
    pretrained_model_name = "runwayml/stable-diffusion-v1-5"
    output_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/ControlNet/out/ip_adapter"
    num_epochs = 1
    learning_rate = 5e-6
    batch_size = 2
    image_size = 224
    print_summary = False

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
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    )
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet"
    )

    unet = get_peft_model(unet, lora_config)
    image_encoder = get_peft_model(image_encoder, lora_config)

    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    )
    # adapter_modules = init_adapter_modules(unet)
    adapter_modules = None
    ip_adapter = IPAdapter(
        unet, image_proj_model, adapter_modules,
        r"/Users/egorprokopov/Documents/Work/ITMO_ML/ControlNet/models/ip-adapter_sd15.bin"
    )

    criterion = nn.MSELoss()

    noise_scheduler = DDPMScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000)

    train_images_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/data/bubbles_split/test/images"
    train_masks_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/data/bubbles_split/test/masks"

    # train_images_dir = r"F:\Internship\ITMO_ML\data\weakly_segmented\bubbles_split\valid\images"
    # train_masks_dir = r"F:\Internship\ITMO_ML\data\weakly_segmented\bubbles_split\valid\masks"

    train_dataset = MasksDataset(train_images_dir, train_masks_dir, width=image_size, height=image_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    device = accelerator.device

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    image_encoder.train()
    ip_adapter.train()

    model = StableDiffusionIP(vae, ip_adapter, image_encoder, text_encoder, tokenizer, noise_scheduler)
    model.to(device=device)

    if print_summary:
        total_params = 0
        for name, param in model.named_parameters():
            print(f"Layer: {name} | Size: {param.size()} | Params_num: {param.numel()} |Trainable: {param.requires_grad}")
            total_params += param.numel()

        print(f"Total params: {total_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    model, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        [model, train_dataloader, optimizer, lr_scheduler]
    )
    model.to(dtype=torch.float16)

    train(model, train_dataloader, criterion, optimizer, lr_scheduler, accelerator, output_dir, num_epochs=num_epochs)

    model.ip_adapter.save_pretrained(f"{output_dir}/final_model_ip_adapter")
    torch.save(model.state_dict(), f"{output_dir}/final_model.pth")

    print("Training complete! Model saved to:", output_dir)

