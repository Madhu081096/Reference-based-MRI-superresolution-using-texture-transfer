from pathlib import Path
import numpy as np
import h5py
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
import torch


class SwappingDataset(Dataset):
    """
    Dataset class for offline feature swapping.
    """

    def __init__(self,input_path, ref_path):

        super(SwappingDataset, self).__init__()

       
        ip=h5py.File(input_path,'r')
        ref=h5py.File(ref_path,'r')
        self.LR=ip['LR']
        self.LR_up=ip['LR_up']
        self.HR=ip['HR']
        self.refLR_up=ref['LR_up']
        self.refHR=ref['HR']
        self.len=self.LR.shape[0]
        #self.ref_len=self.refLR_up.shape[0]

    def __getitem__(self, index):

        img_in = self.HR[index,0]
        img_in_lr = self.LR[index,0]
        img_in_up = self.LR_up[index,0]
        #ref_index=int(index%self.ref_len) #As we use same reference for all volumes
        img_ref = np.array(self.refHR)
        img_ref_up = np.array(self.refLR_up)

        return {'img_in': torch.Tensor(img_in_up).type(torch.FloatTensor),
                'img_ref': torch.Tensor(img_ref).type(torch.FloatTensor),
                'img_ref_blur': torch.Tensor(img_ref_up).type(torch.FloatTensor)}

    def __len__(self):
        return self.len
