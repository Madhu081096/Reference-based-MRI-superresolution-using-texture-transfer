import argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
from datasets.swapping_dataset import SwappingDataset
from models import VGG
from kymatio.torch import Scattering2D
from models.swapper import Swapper
import h5py
import os


TARGET_LAYERS = ['relu1_1']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--save_path',type=str, default='/content/gdrive/My Drive/Texture_transfer/zenodo1/train_map/2/')
    parser.add_argument('--ref_path', default='/content/gdrive/My Drive/Texture_transfer/zenodo/ref/zenodo_ref_axial.h5')
    parser.add_argument('--patch_size', default=3)
    parser.add_argument('--stride', default=1)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def main(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    dataset = SwappingDataset(
        input_path=args.input_path,ref_path=args.ref_path)
    dataloader = DataLoader(dataset)
    # model = VGG(model_type='vgg19').to(device)
    swapper = Swapper(args.patch_size, args.stride).to(device)
    
    J = 2
    P = 400 #272,400
    Q = 400 #480,400
    
    scattering2 = Scattering2D(J, shape=(P, Q))#, backend='torch_skcuda')
    if torch.cuda.is_available():
        scattering2.cuda()
    
    for i, batch in enumerate(tqdm(dataloader), 1):
        
        
        # n=int(i/1000)+1
        path=args.save_path+'slice_'+str(i)+'.h5'
        hf = h5py.File(path, 'w')
        ln= len(batch['img_ref'].shape)
        num = batch['img_ref'].shape[1]
        # print(N)
        M = batch['img_in'].shape[1]
        N = batch['img_in'].shape[2]
        scattering1 = Scattering2D(J, shape=(M, N))#, backend='torch_skcuda')
        if torch.cuda.is_available():
            scattering1.cuda()
        img_in = Variable(batch['img_in'].to(device),requires_grad=False).to(device)
        img_ref = Variable(batch['img_ref'].to(device),requires_grad=False).to(device)
        img_ref_blur = Variable(batch['img_ref_blur'].to(device),requires_grad=False).to(device)
        
        # print(img_in.cpu().shape)
        
        map_in = scattering1.forward(img_in).squeeze()
        map_ref = OrderedDict() 
        map_ref_blur = OrderedDict()
        if ln==3:
            ref = scattering2.forward(img_ref[0]).squeeze()
            ref_blur = scattering2.forward(img_ref_blur[0]).squeeze()
            map_ref[str(1)] = ref
            map_ref_blur[str(1)] = ref_blur

        else:    
            for n in range(num):
                ref = scattering2.forward(img_ref[0,n]).squeeze()
                # print(ref.shape)
                ref_blur = scattering2.forward(img_ref_blur[0,n]).squeeze()
                map_ref[str(n+1)] = ref
                map_ref_blur[str(n+1)] = ref_blur
                # if n==0:
                #     r=ref.numpy()
                #     rb=ref_blur.numpy()
                #     i =map_in.numpy()
                #     np.save('check/map_ref.npy',r[0])
                #     np.save('check/map_ref_blur.npy',rb[0])
                #     np.save('check/map_in.npy',i[0])
        # map_ref = scattering2.forward(img_ref).squeeze()
        # map_ref_blur = scattering2.forward(img_ref_blur).squeeze()
        # map_in = model(img_in, TARGET_LAYERS)
        # map_ref = model(img_ref, TARGET_LAYERS)
        # map_ref_blur = model(img_ref_blur, TARGET_LAYERS)
        # print(map_in['relu1_1'].shape)
        maps, weights, correspondences = swapper(map_in, map_ref, map_ref_blur)
        # mp=n.array(maps['tex1'])
        # np.save('check/map_swapped.npy',mp[0])
        hf.create_dataset('tex1', data=maps['tex1'])
        hf.create_dataset('weights', data=weights)
        # hf.create_dataset('correspondences', data=correspondences)
        hf.close()
        if args.debug and i == 10:
            break
        # if i == 2:
        #     break
        

if __name__ == "__main__":
    main(parse_args())
