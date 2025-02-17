from dataset import MasksDataset
train_images_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/data/bubbles_split/test/images"
train_masks_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/data/bubbles_split/test/masks"

dataset = MasksDataset(train_images_dir, train_masks_dir)
print(len(dataset))

item = dataset[1234]
jpg = item['pixel_values']
txt = item['caption']
hint = item['conditioning_pixel_values']
print(txt)
print(jpg.shape)
print(hint.shape)

import torch
print(torch.device("mps"))
print(torch.autocast(device_type="mps", dtype=torch.bfloat16))