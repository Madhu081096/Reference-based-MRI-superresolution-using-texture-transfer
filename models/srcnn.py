import torch
import torch.nn as nn
import torch.nn.functional as F

import models


# class SRCNN(nn.Module):
    
    
#     def __init__(self, ngf=64, use_weights=False):
#         super(SRCNN, self).__init__()
        
#         self.texture_transfer = SRCNN_TextureTransfer(ngf, use_weights)
#         self.basic_srcnn = SRCNN_basic(ngf)
#         # models.init_weights(self, init_type='normal', init_gain=0.02)

#     def forward(self, x, x_up, maps, weights=None):
#         """
#         Parameters
#         ---
#         x : torch.Tensor
#             the input image.
#         maps : dict of torch.Tensor
#             the swapped feature maps on relu3_1, relu2_1 and relu1_1.
#             depths of the maps are 256, 128 and 64 respectively.
#         """

#         # base = F.interpolate(x, None, 4, 'bilinear', False)
#         # print("Input upscaled with interpolation: ", base.shape)
        

#         if maps is not None:
#             if hasattr(self.texture_transfer, 'a'):  # if weight is used
#                 upscale_srntt = self.texture_transfer(x,x_up, maps, weights)
#                 # print("Texture transferred shape: ",upscale_srntt.shape)
#             else:
#                 upscale_srntt = self.texture_transfer(x,x_up, maps)
#                 # print("Texture transferred shape: ",upscale_srntt.shape)
#             return upscale_srntt 
#         else:
#             upscale_srntt = self.basic_srcnn(x_up)
#             return upscale_srntt 


class SRCNN_TextureTransfer(nn.Module):
    def __init__(self, ngf=64, use_weights=False):
        super(SRCNN_TextureTransfer, self).__init__()
        self.conv1= nn.Sequential(nn.Conv2d(1, ngf//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),            
        )
        self.conv2= nn.Sequential(nn.Conv2d((ngf//2)+81, ngf*4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),nn.PixelShuffle(2)            
        )
        
        self.conv3= nn.Sequential(nn.Conv2d(ngf, ngf*4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),nn.PixelShuffle(2)            
        )
        self.conv4= nn.Sequential(nn.Conv2d(ngf, ngf//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),            
        )
        self.conv5= nn.Sequential(nn.Conv2d(ngf//2, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),            
        )
        models.init_weights(self, init_type='normal', init_gain=0.02)
        if use_weights:
            self.a = nn.Parameter(torch.ones(1), requires_grad=True)
            self.b = nn.Parameter(torch.ones(1), requires_grad=True)
    def forward(self, x, x_up, maps, weights=None):
        # compute weighted maps
        if hasattr(self, 'a') and weights is not None:
            weights_scaled = F.pad(weights, (1, 1, 1, 1), mode='replicate') * self.a+ self.b
            maps *= torch.sigmoid(weights_scaled)

        h = self.conv1(x)
        h = torch.cat([h, maps], 1)
        h = self.conv2(h)
        h = self.conv3(h)
        # h = torch.cat([h, maps], 1)
        h = self.conv4(h)
        h = self.conv5(h)+x_up
        
        return h
        
class SRCNN_basic(nn.Module):
    def __init__(self, ngf=64):
        super(SRCNN_basic, self).__init__()
    
        self.conv1= nn.Sequential(nn.Conv2d(1, ngf//2, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.1, True))           
        self.conv2= nn.Sequential(nn.Conv2d(ngf//2, ngf, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.1, True))
        self.conv3= nn.Sequential(nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.1, True))
        self.conv4= nn.Sequential(nn.Conv2d(ngf, ngf//2, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.1, True))            
        self.conv5= nn.Sequential(nn.Conv2d(ngf//2, 1, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.1, True)) 
        models.init_weights(self, init_type='normal', init_gain=0.02)

    def forward(self, x):
        
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)+x
        return h
        

class SRCNN_TextureTransfer1(nn.Module):
    def __init__(self, ngf=64, use_weights=False):
        super(SRCNN_TextureTransfer1, self).__init__()
        self.conv1= nn.Sequential(nn.Conv2d(1, ngf//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),            
        )
        self.upsampler=nn.Sequential(nn.Conv2d(217,217*4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),nn.PixelShuffle(2) 
        )
        self.conv2= nn.Sequential(nn.Conv2d((ngf//2)+217, ngf*4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),nn.PixelShuffle(2)            
        )
        
        self.conv3= nn.Sequential(nn.Conv2d(ngf, ngf*4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),nn.PixelShuffle(2)            
        )
        self.conv4= nn.Sequential(nn.Conv2d(ngf, ngf//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),            
        )
        self.conv5= nn.Sequential(nn.Conv2d(ngf//2, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),            
        )
        models.init_weights(self, init_type='normal', init_gain=0.02)
        if use_weights:
            self.a = nn.Parameter(torch.ones(1), requires_grad=True)
            self.b = nn.Parameter(torch.ones(1), requires_grad=True)
    def forward(self, x, x_up, maps, weights=None):
        # compute weighted maps
        if hasattr(self, 'a') and weights is not None:
            weights_scaled = F.pad(weights, (1, 1, 1, 1), mode='replicate') * self.a+ self.b
            maps *= torch.sigmoid(weights_scaled)

        h = self.conv1(x)
        mp = self.upsampler(maps)
        h = torch.cat([h, mp], 1)
        h = self.conv2(h)
        h = self.conv3(h)
        # h = torch.cat([h, maps], 1)
        h = self.conv4(h)
        h = self.conv5(h)+x_up
        
        return h