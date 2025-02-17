import lightning.pytorch as pl
from torch.utils.data import DataLoader

from src.data.mask2image_dataset import Mask2ImageDataset


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size, image_size, num_workers=6):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False
        )


class ControlNetDataModule(BaseDataModule):
    def __init__(self, train_images_dir, train_masks_dir, val_images_dir, val_masks_dir, batch_size, image_size, num_workers=6):
        super().__init__(train_images_dir, train_masks_dir, batch_size, image_size, num_workers)
        self.train_images_dir = train_images_dir
        self.train_masks_dir = train_masks_dir
        self.val_images_dir = val_images_dir
        self.val_masks_dir = val_masks_dir
        self.batch_size = batch_size
        self.image_size = image_size

        self.num_workers = num_workers

        self.setup()

    def setup(self, stage=None):
        self.train_dataset = Mask2ImageDataset(self.train_images_dir, self.train_masks_dir, width=self.image_size, height=self.image_size)
        self.val_dataset = Mask2ImageDataset(self.val_images_dir, self.val_masks_dir, width=self.image_size, height=self.image_size)
