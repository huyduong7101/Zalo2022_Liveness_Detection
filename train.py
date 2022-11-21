import numpy as np
import pandas as pd

import os
import tqdm
import json
import argparse
import importlib
import cv2
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, CometLogger

from models import LivenessModel2D, LivenessModel2DLSTM
from datasets import LivenessDataset

def opt():
    parser = argparse.ArgumentParser(description='Arguments')

    # log path
    parser.add_argument('--config', type=str, default='config0', help='config file')
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--split_path', type=str, default='./data', help='data directory')
    parser.add_argument('--log_dir', type=str, default='./checkpoints', help='data directory')
    parser.add_argument('--num_version', type=str, default="v0", help="accelerator")

    # training
    parser.add_argument('--accelerator', type=str, default="cuda", help="accelerator")
    parser.add_argument('--devices', type=int, default=1, help="device")
    parser.add_argument('--num_epochs', type=int, default=10, help="device")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="device")
    parser.add_argument('--learning_rate_min', type=float, default=1e-6, help="device")
    parser.add_argument('--batch_size', type=int, default=4, help="device")
    parser.add_argument('--backbone', type=str, default=None, help="accelerator")

    # data
    parser.add_argument('--num_frames', type=int, default=1, help="device")
    parser.add_argument('--ext', type=str, default="mp4", help="device")

    args = parser.parse_args()
    cfg = importlib.import_module(f'configs.{args.config}').CFG

    # log path
    cfg.train_data_dir = args.data_dir
    cfg.log_dir = os.path.join(args.log_dir, os.path.join(f"{cfg.model_name}_{cfg.backbone}", args.num_version))
    cfg.split_path = args.split_path
    cfg.num_version = args.num_version
    
    # training
    cfg.num_epochs = args.num_epochs
    cfg.learning_rate = args.learning_rate
    cfg.learning_rate_min = args.learning_rate_min
    cfg.batch_size = args.batch_size
    if args.backbone:
        cfg.backbone = args.backbone

    # data
    cfg.num_frames = args.num_frames
    cfg.ext = args.ext

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

    logger = CometLogger(api_key=cfg.comet_api_key, project_name=cfg.comet_project_name, experiment_name=f"{cfg.model_name}/{cfg.backbone}/{cfg.num_version}")
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.log_dir,
        filename='{epoch}-{val_loss:.3f}-{val_auc:.3f}-{val_eer:.3f}',
        monitor="val_auc",
        save_last=True,
        save_top_k=max(5,cfg.num_epochs//5),
        mode = "max",
        every_n_epochs=cfg.save_weight_frequency
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(default_root_dir=cfg.log_dir,
                        max_epochs= cfg.num_epochs,
                        devices=args.devices,
                        accelerator=args.accelerator,
                        logger=logger,
                        callbacks=[checkpoint_callback, lr_monitor])

    print("=="*20)
    print(f"Number of datapoints | train: {len(train_dataset)} | valid: {len(valid_dataset)}")
    print(f"Use {cfg.num_frames} frames")
    print(f"Model: {cfg.model_name} | Version: {trainer.logger.version}")
    print(f"Log path: {cfg.log_dir}")
    print(f"Config: {cfg.__dict__}")
    print("=="*20)

    # if not os.path.exists(cfg.log_dir):
    #     os.makedirs(cfg.log_dir)
    # with open(os.path.join(cfg.log_dir, 'opt.json'), 'w') as f:
    #     json.dump(cfg.__dict__.copy(), f, indent=2)

    trainer.fit(model, train_loader, valid_loader)

if __name__ == "__main__":
    cfg, args = opt()
    main(cfg, args)