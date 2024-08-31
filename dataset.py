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
    def __init__(self, path, image_processor=None, auto_processor=None, size: Tuple[int, int] = (512, 512), debug=False):
        super(VDPDataset, self).__init__()

        self.path = path
        self.images = glob.glob(os.path.join(self.path, '*.png'))

        self.labels = pd.read_csv(f"{path}/metadata.csv")
        
        self.image_processor = image_processor
        self.auto_processor = auto_processor

        self.height = size[0]
        self.width = size[1]

        self.debug = debug

        # TODO: Resize
        self.transform_condition = transforms.Compose([
            transforms.RandomApply([
                transforms.CenterCrop((self.height, self.width)),
            ], p=0.5),
            transforms.Resize((self.height, self.width)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.0),
            ], p=0.5),
            transforms.RandomApply([
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
            ], p=0.5),
            transforms.ToTensor(),
        ])

        self.transform_main = transforms.Compose([
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),
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

        if not self.auto_processor is None:
            prompt_image = self.auto_processor(images=img_aug, return_tensors="pt")
            

        if not self.image_processor is None:
            img = self.image_processor.preprocess(img)
            img_aug = self.image_processor.preprocess(img_aug)



        if self.debug:
            # Denormalize and convert to PIL and save
            img_debug = img * 0.5 + 0.5
            img_aug_debug = img_aug * 0.5 + 0.5
            img_debug = transforms.ToPILImage()(img_debug)
            img_aug_debug = transforms.ToPILImage()(img_aug_debug)
            img_debug.save(f"debug/{idx}_img.png")
            img_aug_debug.save(f"debug/{idx}_img_aug.png")
            return


        #return {'img': (img.squeeze() + 1) / 2, 'img_aug': (img_aug.squeeze() + 1) / 2, 'caption': caption}
        return {'img': img.squeeze(), 'img_aug': img_aug.squeeze(), 'prompt_img': prompt_image, 'caption': caption}
      

def main():
    path = '/mnt/data/vdp_diffusion/mimarlik_data_hf/test'
    dataset = VDPDataset(path=path, debug=True)
    os.makedirs(f"./debug", exist_ok=True)
    for i in range(20):
        dataset[i]

if __name__ == "__main__":
    main()