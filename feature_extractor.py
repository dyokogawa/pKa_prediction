import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, hidden_dim, feature_dim, input_dim):
        super().__init__()

        p = 0.20

        self.encoder = nn.Sequential(
                         nn.Linear(input_dim, hidden_dim), 
#                        nn.BatchNorm1d(hidden_dim),
                         nn.ReLU(), 

#                        nn.Dropout(p),

                         nn.Linear(hidden_dim, feature_dim)
        )
        self.decoder = nn.Sequential(
                         nn.Linear(feature_dim, hidden_dim), 
#                        nn.BatchNorm1d(hidden_dim),
                         nn.ReLU(), 

#                        nn.Dropout(p),

                         nn.Linear(hidden_dim, input_dim)
        )

#       print(input_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _prepare_batch(self, batch):
        x, _, _ = batch
        return x.view(x.size(0), -1)

    def _common_step(self, batch, batch_idx, stage):
        x = self._prepare_batch(batch)
        loss = F.mse_loss(x, self(x))
        self.log(f"{stage}_loss", loss, on_step=True)
        return loss

