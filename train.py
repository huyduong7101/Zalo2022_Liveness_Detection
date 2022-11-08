import numpy as np
import pandas as pd

import os
import tqdm
import argparse
import importlib
import cv2
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from models import LivenessModel2D
from datasets import LivenessDataset


parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument('--config', type=str, default='config0', help='config file')
parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
parser.add_argument('--log_dir', type=str, default='./lightning_logs', help='data directory')
parser.add_argument('--fold', type=int, default=0, help="fold")

args = parser.parse_args()
cfg = importlib.import_module(f'configs.{args.config}').CFG

cfg.fold = args.fold
cfg.train_data_dir = os.path.join(args.data_dir, "train")
cfg.test_data_dir = os.path.join(args.data_dir, "public_test")
cfg.log_dir = os.path.join(args.log_dir, cfg.version)

data_df = pd.read_csv(os.path.join(args.data_dir, "label_with_folding.csv"))
train_df = data_df[data_df['fold']!=cfg.fold].reset_index(drop=True)
valid_df = data_df[data_df['fold']==cfg.fold].reset_index(drop=True)

train_dataset = LivenessDataset(df=train_df, root_dir=cfg.train_data_dir, ext=cfg.ext, transforms=cfg.train_transforms)
valid_dataset = LivenessDataset(df=valid_df, root_dir=cfg.train_data_dir, ext=cfg.ext, transforms=cfg.val_transforms)

cfg.len_train = len(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                            num_workers=cfg.num_workers)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False,
                                            num_workers=cfg.num_workers)

model = LivenessModel2D(cfg)

checkpoint_callback = ModelCheckpoint(
    dirpath = cfg.log_dir,
    filename = '{epoch}-{val_loss:.2f}',
    every_n_epochs= cfg.save_weight_frequency
)

trainer = pl.Trainer(max_epochs= cfg.num_epochs,
                    callbacks=[checkpoint_callback])
trainer.fit(model, train_loader, valid_loader)