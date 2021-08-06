from math import log10
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn.functional as F
from skimage.measure import compare_psnr

def SSIM(input, target):
    s=0.0
    for i, c in enumerate(zip(input, target)):
        ip=c[0].squeeze().cpu().numpy()
        op=c[1].squeeze().cpu().numpy()
        s = s + ssim(ip,op)

    return s / (i + 1)
    
def PSNR(input, target):
    s=0.0
    for i, c in enumerate(zip(input, target)):
        ip=c[0].squeeze().cpu().numpy()
        op=c[1].squeeze().cpu().numpy()
        # mse = F.mse_loss(c[0], c[1], reduction='mean')
        # psnr = 10 * log10(1 / mse.item())
        s = s + compare_psnr(ip,op)
    return s/(i+1)