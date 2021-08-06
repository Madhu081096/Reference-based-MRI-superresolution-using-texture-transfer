from pathlib import Path
import random
from torchvision.transforms import functional as TF
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from os import walk
from os import listdir
from os.path import isfile, join


class ReferenceDataset_vol(Dataset):
    """
    Dataset class for Ref-SR.
    """

    def __init__(self,mypath):
 
        super(ReferenceDataset_vol, self).__init__()
        for _,directory,_ in walk(mypath):
            dirs = directory
            break
        paths =[]
        self.files=[]
        for d in sorted(dirs):
            path = join(mypath,d)
            paths.append(path)
            self.files = self.files + sorted([join(path, f) for f in listdir(path) if isfile(join(path, f))])

        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        
        fname = self.files[index]
        with h5py.File(fname, 'r') as data:
            hr = np.array(data['HR'])
            img_hr=self.transforms(hr[0])
            # print('test ',np.array(data['LR']).shape)
            img_lr=self.transforms(np.array(data['LR']))
            img_lr_up=self.transforms(np.array(data['LR_up']))
            tex=np.array(data['tex1'])
            tex=TF.to_tensor(tex).transpose(1,0)
            weights=self.transforms(np.array(data['weights']))
            return {'img_hr': img_hr.type(torch.FloatTensor), 'img_lr': img_lr.type(torch.FloatTensor),
                    'img_lr_up': img_lr_up.type(torch.FloatTensor),
                    'maps': tex.type(torch.FloatTensor),
                    'weights': weights.type(torch.FloatTensor)}

    def __len__(self):
        return len(self.files)

