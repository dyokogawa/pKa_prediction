import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import re

import os
import os.path


def pKa_shift(itype, element, pKa):
    if element.lower() == 'h':
        if itype == 0:
            pKa = pKa + 10
        if itype == 1:
            pKa = pKa - 10

    return np.float32(pKa)


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def make_dataset(dir, extensions, elist, ref_list):
    images = []
    dir = os.path.expanduser(dir)

    df = pd.read_csv(ref_list)

    for column, row in df.iterrows():
        cid = row['CID']
        pka = row['pKa']
        group_num=re.search('(H)(.*)',row['group']).group(2)
        fname = cid+'_'+'H_'+group_num+'.dat'

        target = pKa_shift(0,'H',float(pka))
        path = os.path.join(dir, fname)
        item = (path, target)
        images.append(item)

    return images

def get_data(dir, fname, ref_list):
    images = []
    dir = os.path.expanduser(dir)

    df = pd.read_csv(ref_list)
    cid = re.search('(.*)(_)(.*)(_)',fname).group(1)
    pka = df.loc[df['CID'] == cid,'pKa']

    target = np.float32(pka)[0]

    path = os.path.join(dir, fname)
    item = (path, target)

    images.append(item)

    return images

def get_maxval(samples,nprop,kbond):

    maxval = np.zeros((nprop,1))

    for i, sample in enumerate(samples):
       path, target = sample

       data1d = np.loadtxt(path,dtype='float32')
       data2d = data1d.reshape([-1,kbond+1])

       if i == 0:
          maxval[:,0] = data2d.max(axis=1)
       else:
          maxval[:,0] = np.maximum(maxval[:,0],data2d.max(axis=1))

    maxval = maxval.astype('float32')

    for i in range(nprop):
       maxval[i,0] = max(maxval[i,0],1.0e-6)
       
    return np.ravel(np.tile(maxval,(1,kbond+1)))

def get_avg(samples,nprop,kbond,maxval):
#
#   average
# 
    avg = np.zeros((nprop*(kbond+1)))
    for i, sample in enumerate(samples):
       path, target = sample

       data1d = np.loadtxt(path,dtype='float32')
 
       avg = avg + data1d/maxval

    avg = avg/float(len(samples))

    return avg

def minmaxnormal(sample,maxval):

    sample = sample/maxval

    return sample


def default_loader(path):
    return torch.from_numpy(np.loadtxt(path,dtype='float32'))

def mydataset(root,fname,maxval,ref_list):
    samples = get_data(root, fname, ref_list)

    path, target = samples[0]

    sample = default_loader(path)

    sample = minmaxnormal(sample,maxval)

    return sample, target

class MyDatasetFolder(data.Dataset):

    def __init__(self, root, extensions, nprop, kbond, elist, ref_list, loader=default_loader):
        samples = make_dataset(root, extensions, elist, ref_list)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.samples = samples
  
        self.maxval = get_maxval(samples,nprop,kbond)
        self.avg    = get_avg(samples,nprop,kbond,self.maxval)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]
        sample = self.loader(path)

        sample = minmaxnormal(sample,self.maxval)

        basename = os.path.basename(path)

        return sample, target, basename


    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str

#dataset = DatasetFolder('../pca/element_data/',['.dat'])
#print(len(dataset))
