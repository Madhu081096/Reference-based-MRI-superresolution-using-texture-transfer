import shutil
import numpy as np
import argparse
from tqdm import tqdm
import math

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from torch import nn
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from torch.autograd import Variable
from torch import optim

from args import create_arg_parser
from utils import save_model,load_model,create_data_loaders,build_optim
from models.srcnn import SRCNN_basic,SRCNN_TextureTransfer,SRCNN_TextureTransfer1
from models.RCAN import make_model
from models.RCAN_Texture_transfer import make_model_TT
from models.RCAN_Texture_attention import make_model_TTA
from models.SRNTT import SRNTT
from models.UDSR import UDSR
from models.RFDN import make_model_RFDN,make_model_RFDN_TT,make_model_RFDN_TTL
from losses.metrics import PSNR, SSIM
from losses.texture_loss import TextureLoss, gradient_loss
from losses.perceptual_loss import PerceptualLoss

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()

def build_model(args):
    netG = make_model_RFDN_TTL(args).to(device)
    # netG = SRNTT().to(device)
    # netG = SRCNN_TextureTransfer().to(device)
    # netG = UDSR(4).to(device)
    print("Model: RCAN_TTL ")
    return netG
    

def pre_train(args,netG,optimizer_G,criterion_rec,train_loader,dev_loader):

    print('Pre-training for ',args.n_epochs_init,' epochs')
    step = 0
    for epoch in range(1, args.n_epochs_init + 1):
        epoch_loss=0
        with tqdm(total=len(train_loader), desc='Pre-train', unit='img') as pbar:
            for i, batch in enumerate(train_loader, 1):
                img_hr = batch['img_hr'].to(device)
                img_lr_up = batch['img_lr_up'].to(device)
                img_lr = Variable(batch['img_lr'],  requires_grad=True).to(device)
                maps = batch['maps'].to(device)
                weights = batch['weights'].to(device)
    
                """ train G """
                
                img_sr = netG(img_lr.float(),img_lr_up, maps, weights.float())
                optimizer_G.zero_grad()
                g_loss = criterion_rec(img_sr, img_hr)
                g_loss.backward()
                optimizer_G.step()
                epoch_loss += g_loss.item()
                # tbar.update(1)
                """ logging """
                writer.add_scalar('pre/g_loss', g_loss.item(), step)
                step+=1
                pbar.update(1)
                
        pre_netG_path = args.exp_dir+'pre_netG_'+str(epoch)+'.pth'
        psnr,ssim,mse=evaluate(args, epoch, netG, dev_loader,False)  
        save_model(args, pre_netG_path, epoch, netG, optimizer_G,psnr,ssim,mse) 
            
        tot_loss=epoch_loss/len(train_loader)
        print(f"Pre-training: Epoch {epoch} - Loss : {tot_loss}")
        
def train(args, epoch, netG, train_loader, optimizer_G,criterion_rec):
    
    step=(epoch-1)*len(train_loader)
    epoch_loss=0
    criterion_tex = TextureLoss(args.use_weights).to(device)
    criterion_per = PerceptualLoss().to(device)
    criterion_grad = gradient_loss().to(device)
    criterion_l2 = nn.MSELoss()
    with tqdm(total=len(train_loader), desc='Training', unit='img') as pbar:
        for i, batch in enumerate(train_loader, 1):
            img_hr = batch['img_hr'].to(device)
            img_lr_up = batch['img_lr_up'].to(device)
            img_lr = Variable(batch['img_lr'], requires_grad=True).to(device)
            # img_lr_2x = batch['img_lr_up2x'].to(device)
            optimizer_G.zero_grad()
            # Train G
            
            if args.texture:
                maps = batch['maps'].to(device)
                weights = batch['weights'].to(device)
                # print(img_lr.cpu().shape)
                # print(img_lr_up.cpu().shape)
                img_sr= netG(img_lr.float(),img_lr_up, maps, weights.float())
                # print(img_sr.cpu().shape)
                # print(img_hr.cpu().shape)
                loss_tex = criterion_tex(img_sr, maps, weights,False)
                loss_rec = criterion_rec(img_sr, img_hr)
                loss_per = criterion_per(img_sr, img_hr)
                # loss_dr = criterion_rec(img_dr,img_lr)
                # loss_grad = criterion_grad(img_sr,img_hr)
                g_loss = (loss_rec * args.lambda_rec+ loss_tex * args.lambda_tex+loss_per*args.lambda_per) #+loss_dr*args.lambda_grad)
            else:
                img_sr = netG(img_lr.float())    
                loss_rec = criterion_rec(img_sr, img_hr)
                loss_per = criterion_per(img_sr, img_hr)
                # y1,y2,y3 = netG(img_lr_up.float())
                # loss_1 = criterion_l2(y1,img_lr_up)
                # loss_2 = criterion_l2(y2,img_lr_up2x)
                # loss_3 = criterion_l2(y3,img_hr)
                g_loss = (loss_rec * args.lambda_rec + loss_per * args.lambda_per)
                # g_loss = loss_1 + loss_2 + loss_3
            
            
            
            g_loss.backward()
            optimizer_G.step()
            epoch_loss+=g_loss.item()
            # tbar.update(1)
            """ logging """
            writer.add_scalar('train/g_loss', g_loss.item(), step)
            # writer.add_scalar('train/loss_rec', loss_rec.item(), step)
            # writer.add_scalar('train/loss_per', loss_per.item(), step)
            if args.texture:
                writer.add_scalar('train/loss_tex', loss_tex.item(), step)
            step+=1
            pbar.update(1)
        
    tot_loss=epoch_loss/len(train_loader)    
    print(f"Training : Epoch {epoch} - Loss : {tot_loss}")
    return netG
    
def evaluate(args, epoch, netG ,dev_loader,logging=True): 
    val_psnr, val_ssim, val_mse = 0, 0,0
    MSE=nn.MSELoss().to(device)
    # tbar = tqdm(total=len(dev_loader))
    with tqdm(total=len(dev_loader), desc='validating', unit='img') as pbar:
        for i, batch in enumerate(dev_loader, 1):
            img_hr = batch['img_hr'].to(device)
            img_lr = batch['img_lr'].to(device)
            img_lr_up = batch['img_lr_up'].to(device)
            if args.texture:
                maps = batch['maps'].to(device)
                weights = batch['weights'].to(device)
                with torch.no_grad():
                    img_sr= netG(img_lr.float(),img_lr_up, maps, weights.float())
                    val_psnr += PSNR(img_hr,img_sr.clamp(0, 1))
                    val_ssim += SSIM(img_sr.clamp(0, 1),img_hr)
                    val_mse += MSE(img_sr.clamp(0,1), img_hr)
            else:
                with torch.no_grad():
                    img_sr = netG(img_lr.float()) 
                    val_psnr += PSNR(img_hr,img_sr.clamp(0, 1))
                    val_ssim += SSIM(img_sr.clamp(0, 1),img_hr)
                    val_mse += MSE(img_sr.clamp(0,1), img_hr)
                    
            
            pbar.update(1)
            # Logging
    if logging==True:
    
        writer.add_scalar('val/psnr', val_psnr, epoch)
        writer.add_scalar('val/ssim', val_ssim, epoch)
        writer.add_scalar('val/mse',val_mse, epoch)
    
    val_psnr /= len(dev_loader)
    val_ssim /= len(dev_loader)
    val_mse /= len(dev_loader)
    # tbar.close()
    print(f'Validating: Epoch {epoch} PSNR:{val_psnr:.4f}, SSIM:{val_ssim:.4f}, MSE:{val_mse:.4f}')
    return val_psnr,val_ssim,val_mse
    
def main(args):
    
    train_loader, dev_loader = create_data_loaders(args)
    netG = build_model(args)
    
    # define criteria
    criterion_rec = nn.L1Loss().to(device)

    # Pretrain for certain epochs
    if args.netG_pre is None:
        """ pretrain """
        start_epoch=0
        best_psnr=0
        best_ssim=0
        best_mse=1e9
        optimizer_G= build_optim(args, netG.parameters())
        # pre_train(args, netG, netD, optimizer_G,optimizer_D, criterion_rec,train_loader,dev_loader)
        if args.texture:
            pass
            # pre_train(args, netG, optimizer_G,criterion_rec,train_loader,dev_loader)
        else:
            pass
    else:
        print('Loading from checkpoint ',args.netG_pre)
        checkpoint,netG,optimizer_G= load_model(args.netG_pre,netG)
        args = checkpoint['args']
        best_psnr= checkpoint['best_PSNR']
        best_ssim = checkpoint['best_SSIM']
        best_mse= checkpoint['MSE']
        start_epoch = checkpoint['epoch']
        
    
    #Scheduler
    scheduler_G = StepLR(optimizer_G, int(args.n_epochs * len(train_loader) / 2), 0.1)
    # scheduler_G = StepLR(optimizer_G,args.lr_step_size, args.lr_gamma)
    #Actual training with all losses
    
    for epoch in range(start_epoch+1, start_epoch+args.n_epochs + 1):
        is_best_psnr=False
        is_best_ssim=False
        netG=train(args, epoch, netG, train_loader, optimizer_G,criterion_rec)
        psnr,ssim,mse=evaluate(args, epoch, netG, dev_loader)
        scheduler_G.step()
        net_path = args.exp_dir+'model_'+str(epoch)+'.pt'
        if psnr>best_psnr:
            print('psnr ',psnr,'best psnr ',best_psnr)
            is_best_psnr=True
            best_psnr=psnr
        if ssim>best_ssim:
            is_best_ssim=True
            best_ssim=ssim
        save_model(args, net_path, epoch, netG, optimizer_G,psnr,ssim,mse,best_ssim,best_psnr,is_best_ssim,is_best_psnr)
        
        
if __name__ == "__main__":
    args = create_arg_parser()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
    writer.close()
        
    
 
