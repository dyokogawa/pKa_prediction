import os
import argparse

import numpy as np
import torch
from tqdm import tqdm

from  mydatafolder import mydataset_IG, pKa_shift
from  alexnet import AlexNet_small

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Integrated gradients')

    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--kbond', default=10, type=int, help='number of bonds (default: 10)')
    parser.add_argument('--nprop', default=7, type=int, help='number of properties (default: 7)')
    parser.add_argument('--ckpt2', default='', type=str, help='check point file of pKa')
    parser.add_argument('--ckpt3', default='', type=str, help='maxval')
    parser.add_argument('--fname', default='', type=str, help='file name of dat file')
    parser.add_argument('--nf0', default=100, type=int, help='nfeatures0 (default: 100)')
    parser.add_argument('--nf1', default=20, type=int, help='nfeatures1 (default: 20)')
    parser.add_argument('--p', default=0.2, type=float, help='dropout ratio (default: 0.2)')
    parser.add_argument('--ref_fname', default='', type=str, help='file name of reerence dat file')
    return parser.parse_args()

def main(args):
#
#   turn off the KMP WARNING
#
    os.environ['KMP_WARNINGS'] = '0'
###################################
#   read checkpoint
###################################
    checkpoint2 = torch.load(args.ckpt2)
    checkpoint3 = torch.load(args.ckpt3)

    data_maxval = checkpoint3['maxval']
###################################
#   set alexnet
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
    model.load_state_dict(checkpoint2['state_dict'])

    data_inp     = mydataset_IG(args.data,args.fname,data_maxval)
    data_ref_inp = mydataset_IG(args.data,args.ref_fname,data_maxval)

    model.eval()
#
#   calculate grad
#
    integrated_gradients, mean_grad, y_pred, y_avg = compute_integrated_gradient(data_inp, data_ref_inp, dim, model)

    integrated_gradients = integrated_gradients.to('cpu').detach().numpy().copy()
#
#   reshape
#
    integrated_gradients = np.reshape(integrated_gradients, (args.nprop,args.kbond+1))
#
#   save
#
#   print(y_avg,y_pred)
    print(y_pred)
    np.savetxt('IG.csv',integrated_gradients, delimiter=' ', fmt='%.5e')

def compute_integrated_gradient(data_inp, data_ref_inp, dim, model):
    mean_grad = 0
    n = 100
#
    avg = data_ref_inp
# 
#   F(z_avg)
#
    x = avg
    y_avg = model(x.view(-1,dim))
    y_avg = y_avg.to('cpu').detach().numpy().copy()[0][0]
    y_avg = pKa_shift(1, 'h', y_avg)

    for i in tqdm(range(1, n + 1)):
        x = i / n * (data_inp -avg) + avg 
#       x = i / n * data_inp

        x.requires_grad = True
        y_pred = model(x.view(-1,dim))

        y_pred[0].backward()

        grad = x.grad
       
        mean_grad += grad / n

    integrated_gradients = (data_inp - avg) * mean_grad
#
#   shift
#
    y_pred = y_pred.to('cpu').detach().numpy().copy()[0][0]
    y_pred = pKa_shift(1, 'h', y_pred)
#   print(y_pred)

    return integrated_gradients, mean_grad, y_pred, y_avg

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


if __name__ == '__main__':
    args = parse_args()
    main(args)

