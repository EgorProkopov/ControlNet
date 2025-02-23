import cv2
import os
import os.path as osp
import numpy as np

from torch.utils.data import Dataset


class Text2ImageDataset(Dataset):
    def __init__(self, images_dir: str, caption: str, width=256, height=256):
        self.width = width
        self.height = height

        self.datadir = images_dir
        self.images_list = os.listdir(self.datadir)

        self.caption = caption

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        prompt = self.caption

        image = cv2.imread(osp.join(self.datadir, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.width, self.height))

        image = (image.astype(np.float32) / 127.5) - 1.0

        return dict(pixel_values=image, caption=prompt)


