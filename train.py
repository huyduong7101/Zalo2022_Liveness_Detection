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

    # path &  name
    parser.add_argument('--transform_config', type=str, default='config_v0', help='config file')
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--split_path', type=str, default='./data', help='data directory')
    parser.add_argument('--log_dir', type=str, default='./checkpoints', help='data directory')
    parser.add_argument('--version_name', type=str, default="v0", help="")
    parser.add_argument('--model_name', type=str, default="2D_baseline", help="")
    parser.add_argument('--save_weight_frequency', type=int, default=1, help="")

    # optimizer
    parser.add_argument('--accelerator', type=str, default="cuda", help="")
    parser.add_argument('--devices', type=int, default=1, help="")
    parser.add_argument('--num_workers', type=int, default=2, help="")
    parser.add_argument('--num_epochs', type=int, default=20, help="")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="")
    parser.add_argument('--learning_rate_min', type=float, default=1e-6, help="")
    parser.add_argument('--batch_size', type=int, default=8, help="")
    parser.add_argument('--liveness_threshold', type=float, default=0.5, help="")

    # model
    parser.add_argument('--backbone', type=str, default="tf_efficientnet_b2_ns", help="")
    parser.add_argument('--pretrained', type=bool , default=True, help="")
    parser.add_argument('--head_type', type=str, default="classify", help="")
    parser.add_argument('--use_lstm', type=bool , default=False, help="")

    # data
    parser.add_argument('--height', type=int, default= 224, help="")
    parser.add_argument('--width', type=int, default= 224, help="")
    parser.add_argument('--num_frames', type=int, default=1, help="")
    parser.add_argument('--ext', type=str, default="mp4", help="")
    parser.add_argument('--fold', type=int, default=0, help="")

    # comet
    parser.add_argument('--comet_api_key', type=str, default= "MqbVRKXYTLalajpK9uSDwDtOk", help="")
    parser.add_argument('--comet_project_name', type=str, default= "Zalo2022_LivenessDetection", help="")

    cfg = parser.parse_args()
    transform_cfg = importlib.import_module(f'configs.{cfg.transform_config}').CFG

    return cfg, transform_cfg

def main(cfg, transform_cfg):
    cfg.log_dir = os.path.join(cfg.log_dir, os.path.join(f"{cfg.model_name}_{cfg.backbone}", cfg.version_name))

    data_df = pd.read_csv(cfg.split_path)
    data_df = data_df[data_df.set == "train"]
    train_df = data_df[data_df['fold']!=float(cfg.fold)].reset_index(drop=True)
    valid_df = data_df[data_df['fold']==float(cfg.fold)].reset_index(drop=True)

    train_transforms = transform_cfg.create_train_transforms(cfg.width, cfg.height)
    val_transforms = transform_cfg.create_val_transforms(cfg.width, cfg.height)
    train_dataset = LivenessDataset(df=train_df, root_dir=cfg.data_dir, ext=cfg.ext, transforms=train_transforms, num_frames=cfg.num_frames)
    valid_dataset = LivenessDataset(df=valid_df, root_dir=cfg.data_dir, ext=cfg.ext, transforms=val_transforms, num_frames=cfg.num_frames)

    cfg.len_train = len(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                                num_workers=cfg.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False,
                                                num_workers=cfg.num_workers, pin_memory=True)

    if cfg.use_lstm:
        model = LivenessModel2DLSTM(cfg)
    else:
        model = LivenessModel2D(cfg)

    logger = CometLogger(api_key=cfg.comet_api_key, project_name=cfg.comet_project_name, experiment_name=f"{cfg.model_name}/{cfg.backbone}/{cfg.version_name}")
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.log_dir,
        filename='{epoch}-{val_loss:.3f}-{val_auc:.3f}-{val_eer:.3f}',
        monitor="val_loss",
        save_last=True,
        save_top_k=max(5,cfg.num_epochs//5),
        mode = "min",
        every_n_epochs=cfg.save_weight_frequency
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(default_root_dir=cfg.log_dir,
                        max_epochs= cfg.num_epochs,
                        devices=cfg.devices,
                        accelerator=cfg.accelerator,
                        logger=logger,
                        callbacks=[checkpoint_callback, lr_monitor])

    print("=="*20)
    print(f"Number of datapoints | train: {len(train_dataset)} | valid: {len(valid_dataset)}")
    print(f"Use {cfg.num_frames} frames")
    print(f"Model: {cfg.model_name} | Version: {trainer.logger.version}")
    print(f"Log path: {cfg.log_dir}")
    print(f"Config: {cfg.__dict__}")
    print("=="*20)

    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    with open(os.path.join(cfg.log_dir, 'opt.json'), 'w') as f:
        json.dump(cfg.__dict__.copy(), f, indent=2)

    trainer.fit(model, train_loader, valid_loader)

if __name__ == "__main__":
    cfg, transform_cfg = opt()
    main(cfg, transform_cfg)