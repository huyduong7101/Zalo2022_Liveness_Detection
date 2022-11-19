import numpy as np
import pandas as pd

import os
import tqdm
import argparse
import importlib
import cv2
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from models import LivenessModel2D, LivenessModel2DLSTM
from datasets import LivenessDataset

def opt():
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--config', type=str, default='config0', help='config file')
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--split_path', type=str, default='./data', help='data directory')
    parser.add_argument('--log_dir', type=str, default='./checkpoints', help='data directory')
    parser.add_argument('--accelerator', type=str, default="cuda", help="accelerator")
    parser.add_argument('--devices', type=int, default=1, help="device")
    parser.add_argument('--num_frames', type=int, default=1, help="device")


    args = parser.parse_args()
    cfg = importlib.import_module(f'configs.{args.config}').CFG
    cfg.num_frames = args.num_frames
    cfg.train_data_dir = args.data_dir
    # cfg.test_data_dir = os.path.join(args.data_dir, "public_test")
    cfg.log_dir = os.path.join(args.log_dir, cfg.version)
    cfg.split_path = args.split_path

    return cfg, args

def main(cfg, args):
    data_df = pd.read_csv(cfg.split_path)
    data_df = data_df[data_df.set == "train"]

    train_df = data_df[data_df['fold']!=float(cfg.fold)].reset_index(drop=True)
    valid_df = data_df[data_df['fold']==float(cfg.fold)].reset_index(drop=True)

    train_dataset = LivenessDataset(df=train_df, root_dir=cfg.train_data_dir, ext=cfg.ext, transforms=cfg.train_transforms, num_frames=cfg.num_frames)
    valid_dataset = LivenessDataset(df=valid_df, root_dir=cfg.train_data_dir, ext=cfg.ext, transforms=cfg.val_transforms, num_frames=cfg.num_frames)

    cfg.len_train = len(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                                num_workers=cfg.num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False,
                                                num_workers=cfg.num_workers)

    if cfg.use_lstm:
        model = LivenessModel2DLSTM(cfg)
    else:
        model = LivenessModel2D(cfg)

    checkpoint_callback = ModelCheckpoint(
        # dirpath=cfg.log_dir,
        filename='{epoch}-{val_loss:.2f}-{val_f1:.2f}-{val_eer:.2f}',
        monitor="val_eer",
        save_last=True,
        save_top_k=max(1,cfg.num_epochs//5),
        mode = "min",
        every_n_epochs=cfg.save_weight_frequency
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(default_root_dir=cfg.log_dir,
                        max_epochs= cfg.num_epochs,
                        devices=args.devices,
                        accelerator=args.accelerator,
                        callbacks=[checkpoint_callback, lr_monitor])

    print(f"Number of datapoints | train: {len(train_dataset)} | valid: {len(valid_dataset)}")
    print(f"Use {cfg.num_frames} frames")
    
    trainer.fit(model, train_loader, valid_loader)

if __name__ == "__main__":
    cfg, args = opt()
    main(cfg, args)