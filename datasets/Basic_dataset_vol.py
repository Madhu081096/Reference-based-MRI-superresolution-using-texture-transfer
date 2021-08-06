# Basic dataset
from pathlib import Path
import random
from torchvision.transforms import functional as TF
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from os import listdir
from os.path import isfile, join

class BasicDataset_vol(Dataset):
    """
    Dataset class for Ref-SR.
    """

    def __init__(self,path,percent = 1):
        super(BasicDataset_vol, self).__init__()
        self.percent = percent
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.examples = []
        files = sorted([join(path, f) for f in listdir(path) if isfile(join(path, f))])
        for fname in files:
            with h5py.File(fname,'r') as hf:
                fsvol = hf['LR']
                num_slices = fsvol.shape[0]
                self.examples += [(fname, slice) for slice in range(num_slices)]
        
            
        

    def __getitem__(self, index):
        
        fname, slice = self.examples[index] 
        with h5py.File(fname, 'r') as data:
            img_hr = np.array(data['HR'])
            img_lr = np.array(data['LR'])
            img_lr_up = np.array(data['LR_up'])
            hr=self.transforms(img_hr[slice,0])
            lr=self.transforms(img_lr[slice,0])
            lr_up=self.transforms(img_lr_up[slice,0])
        
            return {'img_hr': hr.type(torch.FloatTensor), 'img_lr': lr.type(torch.FloatTensor),
                    'img_lr_up': lr_up.type(torch.FloatTensor)}

    def __len__(self):
        return np.ceil(len(self.examples)*self.percent).astype(int)
