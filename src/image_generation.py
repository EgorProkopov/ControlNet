import os

import numpy as np
import torch
from PIL import Image
from diffusers.utils import make_image_grid
from torchvision.transforms import ToPILImage


def denormalize(tensor):
    tensor = (tensor * 0.5 + 0.5) * 255  # [-1, 1] ==> [0, 1] ==> [0, 255]
    tensor = tensor.clamp(0, 255).to(torch.uint8)
    return tensor


def generate_images(model, batch, global_step, log_dir):
    to_pil = ToPILImage()

    model.eval()
    with torch.no_grad():
        captions = batch["caption"]
        conditions = batch["conditioning_pixel_values"].permute(0, 3, 1, 2)
        real_images = batch["pixel_values"].permute(0, 3, 1, 2)

        conditions = conditions.to(model.device)
        generated_images = model.inference(captions=captions, conditions=conditions)
        generated_images = np.array(generated_images)

        for i, img in enumerate(generated_images):
            img_path = os.path.join(log_dir, f"step_{global_step}_image_{i}.png")

            pil_conditions = to_pil(conditions[i].cpu())
            pil_real_img = to_pil(denormalize(real_images[i].cpu()))

            pil_generated_img = Image.fromarray(img)
            pil_generated_img = pil_generated_img.resize(pil_conditions.size, Image.LANCZOS)
            grid = make_image_grid([pil_conditions, pil_generated_img, pil_real_img], rows=1, cols=3)

            grid.save(img_path)

            print(f"Saved grid image at {img_path}")
