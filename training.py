import os
import argparse
from pathlib import Path
from typing import Optional, Union, Tuple
from sklearn.cluster import KMeans

import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import ModelCheckpoint

from  mydatafolder import MyDatasetFolder, pKa_shift
from  alexnet import AlexNet_small

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Autoencoding')

    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of total epochs to run (default: 10)')
    parser.add_argument('--batch', default=256, type=int, help='mini-batch size (default: 256)')
    parser.add_argument('--kbond', default=10, type=int, help='number of bonds (default: 10)')
    parser.add_argument('--nprop', default=7, type=int, help='number of properties (default: 7)')
    parser.add_argument('--elist', default='', type=str, nargs='+', help='element list')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--ref_list', default='', type=str, help='reference data list')
    parser.add_argument('--nf0', default=100, type=int, help='nfeatures0 (default: 100)')
    parser.add_argument('--nf1', default=20, type=int, help='nfeatures1 (default: 20)')
    parser.add_argument('--p', default=0.2, type=float, help='dropout ratio (default: 0.2)')
    return parser.parse_args()

def main(args):
#
#   turn off the KMP WARNING
#
    os.environ['KMP_WARNINGS'] = '0'
#
#   fix random-seed
#
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
###################################
#   call back
###################################
    loss_checkpoint = ModelCheckpoint(
                      dirpath='./',
                      filename=f"best_loss",
                      monitor="val_loss",
                      save_last=True,
                      save_top_k=1,
                      save_weights_only=True,
                      mode="min",
                      )
###################################
#   training
###################################
    val_losses = {}
    val_losses['val'] = []
    val_losses['test'] = []
    val_losses['train'] = []

#
#   small size AlexNet
#
    dim = args.nprop*(args.kbond+1)

    model = AlexNet_small(dim,args.nf0,args.nf1,args.p,val_losses)

    dm = YokoDataModule(data_dir=args.data, 
                        batch_size=args.batch,
                        nprop=args.nprop,
                        kbond=args.kbond,
                        elist=args.elist,
                        ref_list=args.ref_list)
    trainer = pl.Trainer(max_epochs=args.epochs, 
                         log_every_n_steps=1, 
                         accelerator="cpu", 
                         callbacks=[loss_checkpoint],
#                        progress_bar_refresh_rate=0
    )
    trainer.fit(model, dm)
####################################
#   save checkpoint
####################################
    ckptfile = 'maxval.ckpt'
    torch.save({'maxval': dm.maxval,
                'average':dm.avg,
                }, ckptfile)
####################################
#   save loss
####################################

    save_loss(args.epochs,val_losses)

####################################
#   scatter plot
####################################
    model.eval() 

    scatter_plot('val',dm.val_dataset,dm.val_dataloader(),model)
    scatter_plot('train',dm.train_dataset,dm.train_dataloader(),model)

class YokoDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, nprop, kbond, elist, ref_list):
        super().__init__()

        dataset = MyDatasetFolder(data_dir,['.dat'],nprop,kbond,elist,ref_list)
        self.maxval = dataset.maxval
        self.avg    = dataset.avg
        print(repr(dataset))

        n_samples  = len(dataset) 
        train_size = int(len(dataset) * 0.8) 
        val_size   = n_samples - train_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        self.batch_size = batch_size
        self.data_dir = data_dir

#       print(len(self.train_dataset),len(self.val_dataset))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=1, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=1, shuffle=True
        )

class PrintArray:
    def __init__(self, lst):
        self._lst = lst
        self._format = '{:.5f}'
        
    def set_precision(self, precision):
        self._format = '{{:.{}f}}'.format(precision)
        
    def __str__(self):
        return ' '.join(
            self._format.format(e) for e in self._lst
        )

def scatter_plot(data_type,dataset,dataloader,my_model):
    N = len(dataset)

    print('# of data set(',data_type,'):',N)

    for i, (input_tensor, input_target, input_filename) in enumerate(dataloader):
        with torch.no_grad():
           aux = my_model(input_tensor).data.cpu().numpy()

        if i == 0:
            pka_cal = np.zeros((N, aux.shape[1]), dtype='float32')
            pka_exp = np.zeros((N,            1), dtype='float32')
            filename = []

        aux = aux.astype('float32')
        val = input_target.numpy().astype('float32').reshape(-1, 1)
        if i < len(dataloader) - 1:
            pka_cal[i * args.batch: (i + 1) * args.batch] = aux
            pka_exp[i * args.batch: (i + 1) * args.batch] = val
            filename.extend(list(input_filename))
        else:
            # special treatment for final batch
            pka_cal[i * args.batch:] = aux
            pka_exp[i * args.batch:] = val
            filename.extend(list(input_filename))

#   list_pka_cal = np.ravel(pka_cal).tolist()
#   list_pka_exp = np.ravel(pka_exp).tolist()

    list_pka_cal = []
    for x in np.ravel(pka_cal).tolist():
        list_pka_cal.append(pKa_shift(1,'h',x))

    list_pka_exp = []
    for x in np.ravel(pka_exp).tolist():
        list_pka_exp.append(pKa_shift(1,'h',x))

    pka_dict = {"data":filename, "pKa_cal": list_pka_cal, "pKa_exp":list_pka_exp}
    pka_df = pd.DataFrame.from_dict(pka_dict)

    pka_df.to_csv('pka_scatter_'+data_type+'.csv',sep=' ',index = False)


def save_loss(nepochs,val_losses):

    ndim_train = len(val_losses['train'])
    ndim_val = len(val_losses['val'])
    nbatch = int(ndim_train/nepochs)

    mse_train = []
    mse_val = []
    avg = 0.0
    for i, x in enumerate(val_losses['train']):
        mod = i % nbatch
        q   = i // nbatch

        if mod == nbatch-1:
            mse_train.append(avg/float(nbatch))
            avg = 0.0
        else:
            avg = avg + x

    for i, x in enumerate(val_losses['val']):
        if i > 0:
            mse_val.append(x)

    mse_dict = {"training": mse_train, "validation":mse_val}
    mse_df = pd.DataFrame.from_dict(mse_dict)

    mse_df.to_csv('mse_log.csv',sep=' ')


if __name__ == '__main__':
    args = parse_args()
    main(args)

