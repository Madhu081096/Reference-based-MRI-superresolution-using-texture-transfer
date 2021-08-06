import torch
import torch.nn as nn
import models.block as B
import torch.nn.functional as F

def make_model_RFDN(args, parent=False):
    model = RFDN()
    return model
    
def make_model_RFDN_TT(args, parent=False):
    model = RFDN_TT()
    return model
    
def make_model_RFDN_TTL(args, parent=False):
    model = RFDN_TTL()
    return model


class RFDN(nn.Module):
    def __init__(self, in_nc=1, nf=50, num_modules=4, out_nc=1, upscale=4):
        super(RFDN, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = B.RFDB(in_channels=nf)
        self.B2 = B.RFDB(in_channels=nf)
        self.B3 = B.RFDB(in_channels=nf)
        self.B4 = B.RFDB(in_channels=nf)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=4)
        self.scale_idx = 0


    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx


class RFDN_TT(nn.Module):
    def __init__(self, in_nc=1, nf=50, num_modules=4, out_nc=1, upscale=4):
        super(RFDN_TT, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)
        self.fea_conv_tt = B.conv_layer(nf+81,nf,kernel_size=3)
        self.B1 = B.RFDB(in_channels=nf)
        self.B2 = B.RFDB(in_channels=nf)
        self.B3 = B.RFDB(in_channels=nf)
        self.B4 = B.RFDB(in_channels=nf)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)
        # self.degradation_block = B.degradation_block(64)
        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=4)
        self.scale_idx = 0
        
        self.a = nn.Parameter(torch.ones(1), requires_grad=True)
        self.b = nn.Parameter(torch.ones(1), requires_grad=True)


    def forward(self, input,input_up,maps,weights):
        
        #compute weighted maps
        if hasattr(self, 'a') and weights is not None:
            weights_scaled = F.pad(weights, (1, 1, 1, 1), mode='replicate') * self.a+ self.b
            maps *= torch.sigmoid(weights_scaled)
        out_fea = self.fea_conv(input)
        res = torch.cat([out_fea, maps], 1)
        out_fea = self.fea_conv_tt(res)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)
        # dr = self.degradation_block(output)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
        
class RFDN_TTL(nn.Module):
    def __init__(self, in_nc=1, nf=50, num_modules=4, out_nc=1, upscale=4):
        super(RFDN_TTL, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)
        self.fea_conv_tt = B.conv_layer(nf+81,nf,kernel_size=3)
        self.B1 = B.RFDB(in_channels=nf)
        self.B2 = B.RFDB(in_channels=nf)
        self.B3 = B.RFDB(in_channels=nf)
        self.B4 = B.RFDB(in_channels=nf)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)
        # self.degradation_block = B.degradation_block(64)
        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=4)
        self.scale_idx = 0
        
        self.a = nn.Parameter(torch.ones(1), requires_grad=True)
        self.b = nn.Parameter(torch.ones(1), requires_grad=True)


    def forward(self, input,input_up,maps,weights):
        
        #compute weighted maps
        if hasattr(self, 'a') and weights is not None:
            weights_scaled = F.pad(weights, (1, 1, 1, 1), mode='replicate') * self.a+ self.b
            maps *= torch.sigmoid(weights_scaled)
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        res = torch.cat([out_B, maps], 1)
        out_TT = self.fea_conv_tt(res)
        out_lr = self.LR_conv(out_TT) + out_fea

        output = self.upsampler(out_lr)
        # dr = self.degradation_block(output)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx