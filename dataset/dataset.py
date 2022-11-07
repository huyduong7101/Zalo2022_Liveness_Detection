import numpy as np
import cv2
import os
import torch 
from torch.utils.data import Dataset

class Liveness(Dataset):
    def __init__(self, df, root_dir, ext="jpg", transforms=None):
        self.df = df
        self.root_dir = root_dir
        self.ext = ext
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        item_name = item["fname"].split(".")[0]
        item_path = os.path.join(self.root_dir, f"{item_name}/0.{self.ext}") # just use the first frame

        img = cv2.imread(item_path)
        if self.transforms:
            img = self.transforms(image=img)["image"].float()

        label = torch.tensor(item["liveness_score"]).float()
        return img, label

