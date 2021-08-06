import torch
import torch.nn as nn
import torch.nn.functional as F
from kymatio.torch import Scattering2D
import numpy as np
from models import VGG
import canny.canny as canny

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def gram_matrix(features):
    N, C, H, W = features.size()
    feat_reshaped = features.view(N, C, -1)

    # Use torch.bmm for batch multiplication of matrices
    gram = torch.bmm(feat_reshaped, feat_reshaped.transpose(1, 2))

    return gram


class TextureLoss(nn.Module):
    """
    creates a criterion to compute weighted gram loss.
    """
    def __init__(self, use_weights=False):
        super(TextureLoss, self).__init__()
        self.use_weights = use_weights

        self.register_buffer('a', torch.tensor(1, requires_grad=False))
        self.register_buffer('b', torch.tensor(0, requires_grad=False))
        self.model = VGG(model_type='vgg19').to(device)
        self.TARGET_LAYERS = ['relu1_1']

    def forward(self, x, maps, weights,vgg=False):
        
        
        _,_,M,N=x.shape
        if vgg:
            out=self.model(x, self.TARGET_LAYERS)
            x_feat=out['relu1_1']
        else:
            scattering = Scattering2D(2, shape=(M, N))
            scattering.cuda()
            x_feat=scattering.forward(x).squeeze()
            
        
        _,N,R,C=x_feat.shape
        
        if self.use_weights:
            weights_scaled = F.pad(weights, (1, 1, 1, 1), mode='replicate')

            # compute coefficients
            coeff = weights_scaled * self.a.detach() + self.b.detach()
            coeff = torch.sigmoid(coeff)

            # weighting features and swapped maps
            maps= maps * coeff
            x_feat= x_feat * coeff
            # print(x_feat.shape)

        # for small scale
        loss= torch.norm(gram_matrix(x_feat) - gram_matrix(maps)) / 4. / ((R* C * N) ** 2)

        return loss
        
# class gradient_loss(nn.Module):
#     """Gradient loss"""
#     def __init__(self):
#         super(gradient_loss, self).__init__()
#         self.eps = 1e-6
#         s=np.array([[1.0,0,-1.0],[2.0,0,-2.0],[1.0,0,-1.0]])
#         self.Gx=torch.Tensor(s).unsqueeze(0).unsqueeze(0).cuda()
#         self.Gy=torch.Tensor(s.T).unsqueeze(0).unsqueeze(0).cuda()
#         self.loss=nn.L1Loss()

#     def forward(self, X, Y):
#         ax=F.conv2d(X,self.Gx,stride=1)
#         ay=F.conv2d(X,self.Gy,stride=1)
#         bx=F.conv2d(Y,self.Gx,stride=1)
#         by=F.conv2d(Y,self.Gy,stride=1)
#         gradX=torch.sqrt(torch.add(torch.square(ax),torch.square(ay)))
#         gradY=torch.sqrt(torch.add(torch.square(bx),torch.square(by)))
#         error = self.loss(gradX,gradY)
#         return error
        
class gradient_loss(nn.Module):
    """Gradient loss"""
    def __init__(self):
        super(gradient_loss, self).__init__()
        self.loss=nn.L1Loss()

    def forward(self, X, Y):
        error1,error2 =0,0
        for i, c in enumerate(zip(X, Y)):
            ip=c[0].unsqueeze(0)
            op=c[1].unsqueeze(0)
            # print(ip.shape,op.shape)
            grad_true,orientation_true = canny.canny(ip,device)
            grad_pred,orientation_pred = canny.canny(op,device)
            error1 += self.loss(grad_true,grad_pred)
            error2 += self.loss(orientation_true,orientation_pred)
        return error1+error2