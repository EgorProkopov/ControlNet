from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
resume_path = r"F:\Internship\ControlNet\lightning_logs\version_5\checkpoints\swin-13epoch-b4-512.ckpt.ckpt"
batch_size = 2
logger_freq = 6000
learning_rate = 1e-5
sd_locked = False
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.

train_images_dir = r"F:\Internship\ITMO_ML\data\weakly_segmented\bubbles_split\train\images"
train_masks_dir = r"F:\Internship\ITMO_ML\data\weakly_segmented\bubbles_split\train\masks"

val_images_dir = r"F:\Internship\ITMO_ML\data\weakly_segmented\bubbles_split\valid\images"
val_masks_dir = r"F:\Internship\ITMO_ML\data\weakly_segmented\bubbles_split\valid\masks"

model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
train_dataset = MyDataset(train_images_dir, train_masks_dir, width=224, height=224)
train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=batch_size, shuffle=True)

val_dataset = MyDataset(val_images_dir, val_masks_dir, width=224, height=224)
val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=1, shuffle=False)

logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision='f16', callbacks=[logger], accumulate_grad_batches=8)


# Train!
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
