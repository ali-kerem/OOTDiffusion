import json
import os
import pickle
import random
import sys
import glob
from pathlib import Path
from typing import Tuple, Literal, List

PROJECT_ROOT = Path(__file__).absolute().parents[2].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw


class VDPDataset(torch.utils.data.Dataset):
    def __init__(self, path, image_processor, size: Tuple[int, int] = (512, 384)):
        super(VDPDataset, self).__init__()

        self.path = path
        self.images = glob.glob(os.path.join(self.path, '*.png'))

        self.labels = pd.read_csv(f"{path}/metadata.csv")
        
        self.image_processor = image_processor

        self.height = size[0]
        self.width = size[1]

        # TODO: Resize
        self.transform_condition = transforms.Compose([
            transforms.Resize((768, 768)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ], p=0.2),
            transforms.RandomApply([
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))
            ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.transform_main = transforms.Compose([
            transforms.Resize((768, 768)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.labels)
    
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        # img = img.resize((self.width, self.height))
        
        caption = self.labels[self.labels['file_name'] == img_path.split('/')[-1]]['text'].values[0]

        img_aug = self.transform_condition(img)
        img = self.transform_main(img)

        img = self.image_processor(img)
        img_aug = self.image_processor(img_aug)

        return {'img': img, 'img_aug': img_aug, 'caption': caption}
      
