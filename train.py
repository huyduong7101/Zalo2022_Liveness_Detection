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
from pytorch_lightning.loggers import WandbLogger

from models import LivenessModel2D, LivenessModel2DLSTM
from datasets import LivenessDataset

def opt():
    parser = argparse.ArgumentParser(description='Arguments')

    # log path
    parser.add_argument('--config', type=str, default='config0', help='config file')
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--split_path', type=str, default='./data', help='data directory')
    parser.add_argument('--log_dir', type=str, default='./checkpoints', help='data directory')

    # training
    parser.add_argument('--accelerator', type=str, default="cuda", help="accelerator")
    parser.add_argument('--devices', type=int, default=1, help="device")
    parser.add_argument('--num_epochs', type=int, default=10, help="device")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="device")
    parser.add_argument('--learning_rate_min', type=float, default=1e-6, help="device")
    parser.add_argument('--batch_size', type=int, default=4, help="device")

    # data
    parser.add_argument('--num_frames', type=int, default=1, help="device")

    args = parser.parse_args()
    cfg = importlib.import_module(f'configs.{args.config}').CFG

    # log path
    cfg.train_data_dir = args.data_dir
    cfg.log_dir = os.path.join(args.log_dir, cfg.model_name)
    cfg.split_path = args.split_path

    # training
    cfg.num_epochs = args.num_epochs
    cfg.learning_rate = args.learning_rate
    cfg.learning_rate_min = args.learning_rate_min
    cfg.batch_size = args.batch_size

    # data
    cfg.num_frames = args.num_frames

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
                                                num_workers=cfg.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False,
                                                num_workers=cfg.num_workers, pin_memory=True)

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

    # wandb_logger = WandbLogger(project=cfg.project_name)

    trainer = pl.Trainer(default_root_dir=cfg.log_dir,
                        max_epochs= cfg.num_epochs,
                        devices=args.devices,
                        accelerator=args.accelerator,
                        callbacks=[checkpoint_callback, lr_monitor])

    print("=="*20)
    print(f"Number of datapoints | train: {len(train_dataset)} | valid: {len(valid_dataset)}")
    print(f"Use {cfg.num_frames} frames")
    print(f"Model: {cfg.model_name} | Version: {trainer.logger.version}")
    print("=="*20)
    trainer.fit(model, train_loader, valid_loader)

if __name__ == "__main__":
    cfg, args = opt()
    main(cfg, args)