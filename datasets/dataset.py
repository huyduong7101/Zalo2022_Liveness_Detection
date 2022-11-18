import numpy as np
import cv2
import os
import torch 
from torch.utils.data import Dataset

class LivenessDataset(Dataset):
    def __init__(self, df, root_dir, ext="jpg", transforms=None, num_frames=1):
        self.df = df
        self.root_dir = root_dir
        self.ext = ext
        self.transforms = transforms
        self.num_frames = num_frames

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        item_name = item["fname"].split(".")[0]
        
        # get item from extracted frames folder
        # item_path = os.path.join(self.root_dir, f"extracted_frames/{item_name}/0.{self.ext}") # just use the first frame
        # img = cv2.imread(item_path)

        # get item directly from original video
        item_path = os.path.join(self.root_dir, f"videos/{item_name}.mp4")
        if(not os.path.exists(item_path)):
            print("Invalid path:", item_path)
        cap = cv2.VideoCapture(item_path)
        total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if self.num_frames == 1:
            cap.set(1,0)
            ret, img = cap.read()
            if self.transforms:
                imgs = self.transforms(image=img)["image"].float()
        else:
            step = np.floor(total_frame / self.num_frames)
            frames = [i*step for i in range(self.num_frames)]
            imgs = []
            for frame in frames:
                cap.set(1,frame) # 0 is the frame you want
                ret, img = cap.read()
                if ret:
                    if self.transforms:
                        img = self.transforms(image=img)["image"].float()
                    imgs.append(img)
                else:
                    print(f"The number of frames is not enough | Total frame: {total_frame} | {frames}")
            imgs = np.stack(imgs, 0)

        if "liveness_score" in self.df.columns:
            label = torch.tensor(item["liveness_score"]).float()
        else:
            label = -1
    
        return imgs, label