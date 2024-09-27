import json
import cv2
import os
import os.path as osp
import numpy as np

import albumentations as albu

from torch.utils.data import Dataset
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector


class MyDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, width=256, height=256):
        self.width = width
        self.height = height

        self.datadir = images_dir
        self.masksdir = masks_dir
        self.data = os.listdir(self.datadir)

        self.transform = albu.Compose([
            albu.HorizontalFlip(p=0.3),
            albu.VerticalFlip(p=0.3),
            albu.RandomCrop(
                np.random.randint(self.width, 800), np.random.randint(self.height, 600), p=0.8
            ),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = "froth flotation bubbles"

        image = cv2.imread(osp.join(self.datadir, item))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(osp.join(self.masksdir, item))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (self.width, self.height))
        image = resize_image(image, self.width)
        mask = cv2.resize(mask, (self.width, self.height))
        mask = resize_image(mask, self.width)

        mask = mask.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        image = (image.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=image, txt=prompt, hint=mask)


