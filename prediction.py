import os
import argparse

import torch

from  mydatafolder import mydataset, pKa_shift
from  alexnet import AlexNet_small
from  feature_extractor import LitAutoEncoder

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Integrated gradients')

    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--hidden_dim', type=int, default=20,
                        help='dimension of hidden layer (default: 20)')
    parser.add_argument('--kbond', default=10, type=int, help='number of bonds (default: 10)')
    parser.add_argument('--nprop', default=7, type=int, help='number of properties (default: 7)')
    parser.add_argument('--ckpt1', default='', type=str, help='check point file of element')
    parser.add_argument('--ckpt2', default='', type=str, help='check point file of pKa')
    parser.add_argument('--ckpt3', default='', type=str, help='maxval')
    parser.add_argument('--fname', default='', type=str, help='file name of dat file')
    parser.add_argument('--ref_list', default='', type=str, help='reference data list')
    parser.add_argument('--use_encoder', action='store_true', help='use encoder')
    parser.add_argument('--nf0', default=100, type=int, help='nfeatures0 (default: 100)')
    parser.add_argument('--nf1', default=20, type=int, help='nfeatures1 (default: 20)')
    parser.add_argument('--p', default=0.2, type=float, help='dropout ratio (default: 0.2)')
    return parser.parse_args()

def main(args):
#
#   turn off the KMP WARNING
#
    os.environ['KMP_WARNINGS'] = '0'
###################################
#   read checkpoint
###################################
    checkpoint1 = torch.load(args.ckpt1)
    feature_dim = checkpoint1['model_state_dict']['decoder.0.weight'].shape[1]
    descriptor_dim = checkpoint1['model_state_dict']['encoder.0.weight'].shape[1]

    checkpoint2 = torch.load(args.ckpt2)
    checkpoint3 = torch.load(args.ckpt3)

###################################
#   set encoder
###################################

    extractor = LitAutoEncoder(args.hidden_dim,feature_dim,descriptor_dim)

    data_maxval = checkpoint3['maxval']
###################################
#   set alexnet
###################################
    val_losses = {}
    val_losses['val'] = []
    val_losses['test'] = []
    val_losses['train'] = []

    if args.use_encoder :
        dim = feature_dim
    else:
        dim = descriptor_dim

    model = AlexNet_small(extractor,dim,args.use_encoder,args.nf0,args.nf1,args.p,val_losses)

    model.load_state_dict(checkpoint2['state_dict'])

    data_inp, target = mydataset(args.data,args.fname,data_maxval,args.ref_list)

    model.eval()
#
#   prediction
#
    y_pred = model(data_inp.view(-1,descriptor_dim))
    y_pred = y_pred.to('cpu').detach().numpy().copy()[0][0]
    y_pred = pKa_shift(1, 'h', y_pred)
#
#   print out
#
#   result = "predicted pKa: %.3f, reference pKa: %.3f"%(y_pred,target) 
#   print(result)

    print(y_pred,target)


if __name__ == '__main__':
    args = parse_args()
    main(args)

