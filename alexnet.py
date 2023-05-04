#
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class AlexNet_small(pl.LightningModule):
    def __init__(self, feature_dim, nfeatures0, nfeatures1, p, val_losses=None):
        super(AlexNet_small, self).__init__()

        self.classifier = nn.Sequential(
                            nn.Linear(feature_dim, nfeatures0),
                            nn.BatchNorm1d(nfeatures0),
                            nn.ReLU(inplace=True),

                            nn.Dropout(p),

                            nn.Linear(nfeatures0, nfeatures1),
                            nn.BatchNorm1d(nfeatures1),
                            nn.ReLU(inplace=True),
 
                            nn.Dropout(p),
              
                            nn.Linear(nfeatures1, nfeatures0),
                            nn.BatchNorm1d(nfeatures0),
                            nn.ReLU(inplace=True),
 
                            nn.Dropout(p)
                            )

        self.top_layer = nn.Linear(nfeatures0, 1)
        self.val_losses = val_losses

    def forward(self, x):
        x = self.classifier(x)
        x = self.top_layer(x)
        return x

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _prepare_batch(self, batch):
        x, y, _ = batch
        return x.view(x.size(0), -1), y.view(y.size(0),-1)

    def _common_step(self, batch, batch_idx, stage):
        x, y = self._prepare_batch(batch)
        loss = F.mse_loss(y, self(x))
        self.log(f"{stage}_loss", loss, on_step=True,on_epoch=True)
        self.val_losses[stage].append(loss.to('cpu').detach().numpy().copy())
        return loss
