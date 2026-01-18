import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrajectoryTrainer:
    def __init__(self, pipeline, lr=1e-3, epochs=20, device=None):
        self.pipeline = pipeline
        self.model = pipeline.model
        self.device = device or pipeline.DEVICE
        self.epochs = epochs

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
        )

    def masked_mse_loss(self, preds, targets):
        mask = ~(targets == 0).all(dim=-1)
        mask = mask.unsqueeze(-1).expand_as(targets)

        masked_preds = preds[mask]
        masked_targets = targets[mask]

        return F.mse_loss(masked_preds, masked_targets)

    def evaluate(self, loader):
        if loader is None:
            return None

        self.model.eval()
        losses = []

        with torch.no_grad():
            for num_feats, cat_feats, targets in loader:
                num_feats = num_feats.to(self.device)
                cat_feats = cat_feats.to(self.device)
                targets = targets.to(self.device)

                preds = self.model(num_feats, cat_feats)
                loss = self.masked_mse_loss(preds, targets)
                losses.append(loss.item())

        if not losses:
            return None

        return float(np.sqrt(np.mean(losses)))

    def train(self):
        train_loader = self.pipeline.train_loader
        val_loader = self.pipeline.val_loader

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_losses = []

            for num_feats, cat_feats, targets in train_loader:
                num_feats = num_feats.to(self.device)
                cat_feats = cat_feats.to(self.device)
                targets = targets.to(self.device)

                preds = self.model(num_feats, cat_feats)

                loss = self.masked_mse_loss(preds, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())

            train_rmse = float(np.sqrt(np.mean(train_losses)))
            val_rmse = self.evaluate(val_loader)
            self.scheduler.step(val_rmse if val_rmse is not None else train_rmse)

            print(
                f"Epoch {epoch:02d} | Train RMSE: {train_rmse:.4f}"
                + (f" | Val RMSE: {val_rmse:.4f}" if val_rmse else "")
                + f" | LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )

        return self.model


class TrajectoryTransformerTrainer:
    def __init__(self, model, train_loader, val_loader=None, lr=1e-3, epochs=20, device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device or DEVICE

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=0.5,
            patience=2,
            mode="min",
            min_lr=1e-5,
        )

    def masked_mse_loss(self, preds, targets):
        mask = ~(targets == 0).all(dim=-1)
        mask = mask.unsqueeze(-1).expand_as(targets)

        masked_preds = preds[mask]
        masked_targets = targets[mask]

        return F.mse_loss(masked_preds, masked_targets)

    def evaluate(self, loader):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for src, tgt in loader:
                out = self.model(src, tgt)
