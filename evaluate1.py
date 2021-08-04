import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import argparse
import h5py
from torch.utils.data import DataLoader
from tqdm import tqdm
from os import listdir
from os.path import isfile, join

from models.srcnn import SRCNN_basic,SRCNN_TextureTransfer
from models.RCAN import make_model
from models.RCAN_Texture_transfer import make_model_TT
from models.RFDN import make_model_RFDN, make_model_RFDN_TT
from models.SRNTT import SRNTT
from models.pix2pix import GeneratorUNet,Discriminator
from losses.metrics import PSNR, SSIM
from datasets.reference_dataset import ReferenceDataset, BasicDataset
from datasets.Basic_dataset_vol import BasicDataset_vol
from datasets.Reference_dataset_vol import ReferenceDataset_vol
from skimage.metrics import structural_similarity as SS
from skimage.measure import compare_psnr
from imresize import imresize

# %matplotlib inline
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Evaluation setup')
    parser.add_argument('--model_name',type = str , default = 'SRCNN')
    parser.add_argument('--ckpt',type = str , default = './storage/best_psnr_SRCNN_per.pt')
    parser.add_argument('--data_base',type = str, default = '/content/gdrive/My Drive/Texture_transfer/zenodo1/')
    
    ## single file test parameters
    parser.add_argument('--single', action = 'store_true')
    parser.add_argument('--file_path',default = 'None')
    parser.add_argument('--slice_no',type=int,default = 5)
    parser.add_argument('--data',default = 'sag',help = 'sag, axial')
    parser.add_argument('--ref',default = 'sag',help = 'sag, axial')
    parser.add_argument('--save',action='store_true')
    parser.add_argument('--visualize',action = 'store_true')
    parser.add_argument('--save_csv',action  = 'store_true')
    
    return parser.parse_args()
  
def load_model(args1):
    checkpoint = torch.load(args1.ckpt)
    args = checkpoint['args']
    netG = build_model(args,args1.model_name)
    netG.load_state_dict(checkpoint['netG'])
    return checkpoint, netG
    
def create_datasets(args,args1):
    
    # if args.texture:
    if args1.ref=='sag':
        if args1.data=='sag':
            eval_path = args1.data_base+'val_map/'
        else:
            eval_path = args1.data_base+'val_1_map/'
    else:
        if args1.data=='sag':
            eval_path = args1.data_base+'val_map_axial/'
        else:
            eval_path = args1.data_base+'val_1_map_axial/'
    dev_data = ReferenceDataset_vol(eval_path)
    # else:
        
    #     eval_path = args1.data_base+'val/'
    #     dev_data = BasicDataset_vol(eval_path)
    return dev_data
    
def create_data_loaders(args,args1):
    
    dev_data= create_datasets(args,args1)
    
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=1,
        #num_workers=64,
        #pin_memory=True,
    )
    return dev_loader

def build_model(args,model_name):
    if model_name == 'SRCNN':
        netG = SRCNN_basic().to(device)
    if model_name == 'SRCNN_TT':
        netG = SRCNN_TextureTransfer().to(device)
    if model_name == 'RCAN':
        netG = make_model(args).to(device)
    if model_name == 'RCAN_TT':    
        netG = make_model_TT(args).to(device)
    if model_name == 'RFDN':
        netG = make_model_RFDN(args).to(device)
    if model_name == 'RFDN_TT':    
        netG = make_model_RFDN_TT(args).to(device)
    if model_name == 'SRNTT':    
        netG = SRNTT().to(device)
    if model_name == 'pix2pix':
        netG = GeneratorUNet().to(device)

    print("Model: ", model_name)
    
    return netG

def test_single_b(args1,dev_loader):
    slno = args1.slice_no
    for i, batch in enumerate(dev_loader, 1):
        if i != slno:
            continue
        
        img_hr = batch['img_hr'].to(device).cpu().numpy()
        img_lr = batch['img_lr'].to(device).cpu().numpy()
        img_lr_up = batch['img_lr_up'].to(device).cpu().numpy()
        if args1.model_name == 'bilinear':
            img_sr = imresize(img_lr[0,0],4,method='bilinear')
        else:
            img_sr = imresize(img_lr[0,0],4,method='bicubic')
        psnr = compare_psnr(img_hr[0,0],img_sr)
        ssim = SS(img_sr,img_hr[0,0])
        break
    print('PSNR - ',psnr,' SSIM - ',ssim)
    if args1.visualize:
        # Visualize
        sr=img_sr
        hr=img_hr[0,0]
        lr=img_lr_up[0,0]
        np.save('./results/LR_up.npy',lr)
        np.save('./results/HR.npy',hr)
        np.save('./results/SR.npy',sr)

    return 0
    
def test_single(args, args1, netG,dev_loader):
    slno = args1.slice_no
    for i, batch in enumerate(dev_loader, 1):
        if i != slno:
            continue
        
        img_hr = batch['img_hr'].to(device)
        img_lr = batch['img_lr'].to(device)
        img_lr_up = batch['img_lr_up'].to(device)
        if args.texture:
            maps = batch['maps'].to(device)
            weights = batch['weights'].to(device)
            with torch.no_grad():
                img_sr = netG(img_lr.float(),img_lr_up, maps, weights.float())
                psnr = PSNR(img_hr,img_sr.clamp(0, 1))
                ssim = SSIM(img_sr.clamp(0, 1),img_hr)

        else:
            netG.eval()
            with torch.no_grad():
                if args1.model_name == 'RCAN' or args1.model_name == 'RFDN' :
                    img_sr = netG(img_lr.float()) 

                else:
                    img_sr = netG(img_lr_up.float()) 
                    # print(img_sr.cpu().shape)
                    # print(img_hr.cpu().shape)
                psnr = PSNR(img_hr,img_sr.clamp(0, 1))
                ssim = SSIM(img_sr.clamp(0, 1),img_hr)
        break

    print('PSNR - ',psnr,' SSIM - ',ssim)
    if args1.visualize:
        # Visualize
        
        sr=img_sr.cpu().numpy()
        sr=sr[0,0]
        hr=img_hr.cpu().numpy()
        hr=hr[0,0]
        lr=img_lr_up.cpu().numpy()
        lr=lr[0,0]
        np.save('./results/LR_up.npy',lr)
        np.save('./results/HR.npy',hr)
        np.save('./results/SR.npy',sr)

    return 0

def test_all(args, netG ,dev_loader): 
    val_psnr, val_ssim, val_mse = 0, 0,0
    results = np.zeros((len(dev_loader),3))
    with tqdm(total=len(dev_loader), desc='validating', unit='img') as pbar:
        for i, batch in enumerate(dev_loader, 1):
            results[i-1,0] = i-1;
            img_hr = batch['img_hr'].to(device)
            img_lr = batch['img_lr'].to(device)
            img_lr_up = batch['img_lr_up'].to(device)
            if args.texture:
                maps = batch['maps'].to(device)
                weights = batch['weights'].to(device)
                netG.eval()
                with torch.no_grad():
                    img_sr = netG(img_lr.float(),img_lr_up, maps, weights.float())
                    psnr = PSNR(img_hr,img_sr.clamp(0, 1))
                    ssim = SSIM(img_sr.clamp(0, 1),img_hr)
                    val_psnr += psnr
                    val_ssim += ssim
                    results[i-1,1] = psnr
                    results[i-1,2] = ssim
            else:
                with torch.no_grad():
                    if args1.model_name == 'RCAN'or args1.model_name == 'RFDN':
                        img_sr = netG(img_lr.float()) 
                    else:
                        img_sr = netG(img_lr_up.float()) 
                    psnr = PSNR(img_hr,img_sr.clamp(0, 1))
                    ssim = SSIM(img_sr.clamp(0, 1),img_hr)
                    val_psnr += psnr
                    val_ssim += ssim
                    results[i-1,1] = psnr
                    results[i-1,2] = ssim
                    
            
            pbar.update(1)
            
    val_psnr /= len(dev_loader)
    val_ssim /= len(dev_loader)
    
    print(f'Evaluating: PSNR:{val_psnr:.4f}, SSIM:{val_ssim:.4f}')
    return np.argmax(results[:,1]),np.argmax(results[:,2]), results

    
def main(args1):

    print('Loading from checkpoint ',args1.ckpt)
    dev_loader = create_data_loaders(None,args1)
    if args1.model_name=='bicubic' or args1.model_name=='bilinear':
        if args1.single:
            print('Testing single slice...')
            test_single_b(args1,dev_loader)
    else:
        checkpoint,netG= load_model(args1)
        args = checkpoint['args']
        
        if args1.single:
            print('Testing single slice...')
            test_single(args,args1, netG,dev_loader)
        else:
            max_psnr,max_ssim,results=test_all(args, netG, dev_loader)
            print('Slice of max_psnr ',max_psnr)
            print('Slice of max_ssim ',max_ssim)
            if args1.save_csv:
                np.savetxt("results/result_all.npy", results)
                np.savetxt("results/result_all.csv", results, delimiter=",")
    
if __name__ == "__main__":
    args1 = create_arg_parser()
    main(args1)