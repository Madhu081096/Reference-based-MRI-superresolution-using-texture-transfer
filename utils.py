# Utilities
import argparse
import torch
from torch.utils.data import DataLoader
from datasets.reference_dataset import ReferenceDataset, BasicDataset
from datasets.Basic_dataset_vol import BasicDataset_vol
from datasets.Reference_dataset_vol import ReferenceDataset_vol
import shutil


def create_datasets(args):
    if args.texture:
        train_path = args.data_base+'train_map/'
        eval_path = args.data_base+'val_map/'
        train_data = ReferenceDataset_vol(train_path)
        dev_data = ReferenceDataset_vol(eval_path)
        
        # dev_path=args.dataroot+'/'+args.data_name+'/data/val.h5'
        # map_path_dev = args.dataroot+'/'+args.data_name+'/val_map/slice_'
        # train_path=args.dataroot+'/'+args.data_name+'/data/train.h5'
        # map_path_train = args.dataroot+'/'+args.data_name+'/train_map/'
        # train_data = ReferenceDataset(train_path,map_path_train)
        # dev_data = ReferenceDataset(dev_path,map_path_dev,True)
        # train_data = ReferenceDataset(args.dataroot,args.data_name)
        # dev_data = ReferenceDataset(args.dataroot,args.data_name,True)
    else:
        # train_data = BasicDataset(args.dataroot,args.data_name,args.scale_factor,False)
        # dev_data = BasicDataset(args.dataroot,args.data_name,args.scale_factor,True)
        train_path = args.data_base+'train/'
        eval_path = args.data_base+'val/'
        train_data = BasicDataset_vol(train_path,args.data_size)
        dev_data = BasicDataset_vol(eval_path,args.data_size)
    return dev_data, train_data
    
def create_data_loaders(args):
    dev_data, train_data = create_datasets(args)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        #num_workers=64,
        #pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=1,
        #num_workers=64,
        #pin_memory=True,
    )
    return train_loader, dev_loader
	

def save_model_gan(args, path, epoch, netG,netD,optimizer_G,optimizer_D,psnr,ssim,mse,best_ssim,best_psnr,is_best_ssim=False,is_best_psnr=False):

    out = torch.save(
        {
            'epoch': epoch,
            'args': args,
            'netG': netG.state_dict(),
            'netD' : netD.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
            'PSNR': psnr,
            'SSIM':ssim,
            'MSE':mse,
            'best_PSNR':best_psnr,
            'best_SSIM':best_ssim
            #'exp_dir':exp_dir
        },
        path 
    )
    path_last=args.storage+'last_epoch_'+args.model_name+'.pt'
    shutil.copyfile(path, path_last)
    if is_best_psnr:
        path_psnr=args.storage+'best_psnr_'+args.model_name+'.pt'
        shutil.copyfile(path, path_psnr)
    if is_best_ssim:
        path_mse=args.storage+'best_ssim_'+args.model_name+'.pt'
        shutil.copyfile(path,path_mse )
		
def load_model_gan(checkpoint_file,netG,netD):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    netG.load_state_dict(checkpoint['netG'])
    netD.load_state_dict(checkpoint['netD'])
    optimizer_G= build_optim(args, netG.parameters())
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    optimizer_D= build_optim(args, netD.parameters())
    optimizer_D.load_state_dict(checkpoint['optimizer_D'])

    return checkpoint, netG,netD, optimizer_G,optimizer_D
    
def build_optim(args, params):
    if args.optimizer=='ADAM':
        optimizer_G = torch.optim.Adam(params,args.lr, betas=(args.beta1,args.beta2), eps=args.epsilon, weight_decay=args.weight_decay)
        return optimizer_G
    if args.optimizer=='SGD':
        optimizer_G=torch.optim.SGD(params,args.lr)
        
def save_model(args, path, epoch, netG,optimizer_G,psnr,ssim,mse,best_ssim,best_psnr,is_best_ssim=False,is_best_psnr=False):

    out = torch.save(
        {
            'epoch': epoch,
            'args': args,
            'netG': netG.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'PSNR': psnr,
            'SSIM':ssim,
            'MSE':mse,
            'best_PSNR':best_psnr,
            'best_SSIM':best_ssim
            #'exp_dir':exp_dir
        },
        path 
    )
    path_last=args.storage+'last_epoch_'+args.model_name+'.pt'
    shutil.copyfile(path, path_last)
    if is_best_psnr:
        path_psnr=args.storage+'best_psnr_'+args.model_name+'.pt'
        shutil.copyfile(path, path_psnr)
    if is_best_ssim:
        path_mse=args.storage+'best_ssim_'+args.model_name+'.pt'
        shutil.copyfile(path,path_mse )
		

def load_model(checkpoint_file,netG):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    netG.load_state_dict(checkpoint['netG'])
    optimizer_G= build_optim(args, netG.parameters())
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    return checkpoint, netG,optimizer_G
