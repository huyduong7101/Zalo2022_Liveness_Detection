import numpy as np
import cv2
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import timm

from utils import equal_error_rate

class LivenessModel2D(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(cfg.backbone, cfg.pretrained)

        if 'efficient' in cfg.backbone:
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(in_features, 1)
        elif 'convnext' in cfg.backbone:
            hdim = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Linear(hdim, 1)

        if cfg.head_type == "regress":
            self.head = nn.Sigmoid()
            self.criteria = nn.MSELoss()
        else:
            self.head = nn.Sigmoid()
            self.criteria = nn.BCELoss()
        self.global_avg = nn.AdaptiveAvgPool1d(1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.learning_rate)

        steps_per_epoch = int(self.cfg.len_train / self.cfg.num_epochs)
        num_train_steps = steps_per_epoch * self.cfg.num_epochs
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.cfg.learning_rate, steps_per_epoch=steps_per_epoch, epochs=self.cfg.num_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_train_steps, eta_min=self.cfg.learning_rate_min)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        batch_size = x.shape[0]
        x = x.view(batch_size*self.cfg.num_frames, 3, self.cfg.height, self.cfg.width)
        out = self.backbone(x)
        out = self.head(out)
        out = out.view(batch_size, self.cfg.num_frames).contiguous()
        out = self.global_avg(out)
        return out

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        y_pred = self(x).view(-1)

        loss = self.criteria(y_pred, y)

        self.log("train_loss", loss)
        return {'loss': loss, 'preds':y_pred, 'labels':y}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        y_pred = self(x).view(-1)
        
        loss = self.criteria(y_pred, y)

        self.log("val_loss", loss)
        return {'loss': loss, 'preds':y_pred, 'labels':y} 

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        y_pred = self(x).view(-1)

        return y_pred

    def compute_metrics(self, outputs):
        all_preds = np.concatenate([out['preds'].detach().cpu().numpy() for out in outputs])
        all_labels = np.concatenate([out['labels'].detach().cpu().numpy() for out in outputs])
        all_preds = (all_preds > self.cfg.liveness_threshold).astype(int)
        f1 = float(f1_score(y_true=all_labels, y_pred=all_preds))
        eer, _ = equal_error_rate(all_labels, all_preds)
        return {"f1": f1, "eer": eer}

    def training_epoch_end(self, training_step_outputs):
        metrics = self.compute_metrics(training_step_outputs)
        for k, v in metrics.items():
            self.log(f'train_{k}', v, prog_bar=False)
        
    def validation_epoch_end(self, validation_step_outputs):
        metrics = self.compute_metrics(validation_step_outputs)
        for k, v in metrics.items():
            self.log(f'val_{k}', v, prog_bar=True)