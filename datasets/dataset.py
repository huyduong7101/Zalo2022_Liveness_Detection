import numpy as np
import cv2
import os
import torch 
from torch.utils.data import Dataset
import albumentations as A

class LivenessDataset(Dataset):
    def __init__(self, df, root_dir, ext="jpg", transforms=None, num_frames=1):
        self.df = df
        self.root_dir = root_dir
        self.ext = ext
        self.transforms = transforms
        self.num_frames = num_frames
        print(f"Use {self.num_frames} frame | Load image from {self.ext}")
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        item_name = item["fname"].split(".")[0]


        if self.ext == "jpg" or self.ext == "png":
            item_path = os.path.join(self.root_dir, f"extracted_frames/{item_name}")
            if(not os.path.exists(item_path)):
                print("Invalid path:", item_path)
            total_frames = len(os.listdir(item_path))

            if self.num_frames == 1:
                try:
                    id_frame = np.random.randint(total_frames)
                    img = cv2.imread(f"{item_path}/{id_frame}.{self.ext}")
                    if self.transforms:
                        img = self.transforms(image=img)["image"].float()
                    imgs = img
                except:
                    print(f"Invalid path: id_frame {id_frame} | total_frames {total_frames}")
            else:
                step = min(total_frames // self.num_frames, 5)
                frames = [i*step for i in range(self.num_frames)]
                imgs = []
                for cnt, id_frame in enumerate(frames):
                    try:
                        img = cv2.imread(f"{item_path}/{id_frame}.{self.ext}")
                        if self.transforms:
                            if cnt == 0:
                                replay_aug = self.transforms(image=img)
                                img = replay_aug["image"].float()
                            else:
                                img = A.ReplayCompose.replay(replay_aug['replay'], image=img)["image"].float()
                        imgs.append(img)
                    except:
                        print(f"Invalid path: id_frame {id_frame} | total_frames {total_frames}")
                imgs = np.stack(imgs, 0)

        # get item directly from original video
        elif self.ext == "mp4":
            item_path = os.path.join(self.root_dir, f"videos/{item_name}.mp4")
            if(not os.path.exists(item_path)):
                print("Invalid path:", item_path)
            cap = cv2.VideoCapture(item_path)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            if self.num_frames == 1:
                id_frame = np.random.randint(total_frames)
                cap.set(1,id_frame)
                ret, img = cap.read()
                if self.transforms:
                    img = self.transforms(image=img)["image"].float()
                imgs = img
            else:
                step = min(total_frames // self.num_frames, 5)
                frames = [i*step for i in range(self.num_frames)]
                imgs = []

                # option 1
                # for frame in frames:
                #     cap.set(1,frame) # 0 is the frame you want
                #     ret, img = cap.read()
                #     if ret:
                #         if self.transforms:
                #             img = self.transforms(image=img)["image"].float()
                #         imgs.append(img)
                #     else:
                #         print(f"The number of frames is not enough | Total frame: {total_frames} | {frames}")

                # option 2
                count = 0
                len_imgs = 0
                while True:
                    ret, img = cap.read()
                    if ret:
                        if img is not None and count % step == 0:
                            if self.transforms:
                                if len_imgs == 0:
                                    replay_aug = self.transforms(image=img) 
                                    img = replay_aug["image"].float()
                                else:
                                    img = A.ReplayCompose.replay(replay_aug['replay'], image=img)['image'].float()
                            imgs.append(img)

                            len_imgs += 1
                            if len_imgs == self.num_frames:
                                break
                        count += 1
                    else:
                        if len_imgs < self.num_frames:
                            print(f"The number of frames is not enough | Total frame: {total_frames} | {count}")
                        break 
 
                imgs = np.stack(imgs, 0)
            cap.release()

        if "liveness_score" in self.df.columns:
            label = torch.tensor(item["liveness_score"]).float()
        else:
            label = -1
    
        return imgs, label