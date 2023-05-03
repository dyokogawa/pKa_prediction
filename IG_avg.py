import argparse

import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Integrated gradients')

    parser.add_argument('--kbond', default=10, type=int, help='number of bonds (default: 10)')
    parser.add_argument('--nprop', default=7, type=int, help='number of properties (default: 7)')
    return parser.parse_args()

def main(args):

    IG_data = 'IG.tmp'

    IG_total = np.loadtxt(IG_data,dtype='float32')

    ndata = int(int(IG_total.shape[0])/args.nprop)

    IG_avg = np.zeros((args.nprop,args.kbond+1))

    for i in range(ndata):

       IG_avg = IG_avg + IG_total[i*args.nprop:(i+1)*args.nprop, 0:]
#
#   summarize
#
    
    df = pd.DataFrame(IG_avg/ndata, 
                  columns=["k=0","k=1","k=2","k=3","k=4","k=5","k=6","k=7","k=8","k=9","k=10"],
                  index=["Q+","Q-","F-","V","C6","sgm+","sgm-","M+","M-"] )

    print(df)

if __name__ == '__main__':
    args = parse_args()
    main(args)

