from share import *
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

#torch.set_default_device("mps")

# Configs
resume_path = r"/Users/egorprokopov/Documents/Work/ITMO_ML/ControlNet/models/control_sd21_ini.ckpt"
batch_size = 1
logger_freq = 1000
learning_rate = 1e-5
sd_locked = False
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.

train_images_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/data/bubbles_split/test/images"
train_masks_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/data/bubbles_split/test/masks"

val_images_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/data/bubbles_split/valid/images"
val_masks_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/data/bubbles_split/valid/masks"

model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# torch.set_default_device(torch.device('mps:0'))

# Misc
train_dataset = MyDataset(train_images_dir, train_masks_dir, width=256, height=256)
train_dataloader = DataLoader(
    train_dataset,
    num_workers=0,
    batch_size=batch_size,
    generator=torch.Generator(device='mps:0'),
    shuffle=True
)

val_dataset = MyDataset(val_images_dir, val_masks_dir, width=256, height=256)
val_dataloader = DataLoader(
    val_dataset,
    num_workers=0,
    batch_size=1,
    generator=torch.Generator(device='mps:0'),
    shuffle=False
)

logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision='bf16', accelerator='mps', callbacks=[logger], accumulate_grad_batches=8)


# Train!
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
