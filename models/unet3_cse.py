""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class TripleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.triple_conv(x)    


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            TripleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = TripleConv(in_channels*2, out_channels, in_channels )
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels, kernel_size=2, stride=2)
            self.conv = TripleConv(in_channels*2, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        #print(f'after up x1 shape: {x1.shape}')
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        #print(f'before concat x2 shape: {x2.shape} x1 shape: {x1.shape}')
        x = torch.cat([x2, x1], dim=1)
        #print(f'after concat x shape: {x.shape})
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

class SE_block(nn.Module):
    """squeeze and excitation block"""
    def __init__(self, num_features, reduction_factor=4):
        super(SE_block, self).__init__()
        # squeeze block
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        # excitation block
        self.excite = nn.Sequential(
            nn.Linear(num_features, num_features // reduction_factor),
            nn.ReLU(inplace=True),
            nn.Linear(num_features // reduction_factor, num_features),
            nn.Sigmoid()
        )
    def forward(self, x):
        batch, channel, _, _ = x.size()
        squeeze_res = self.squeeze(x).view(batch, channel)
        #print(f'squeeze_res: {squeeze_res.shape}')
        excite_res = self.excite(squeeze_res)
        #print(f'excite_res: {excite_res.shape}')
        f_scale = excite_res.view(batch, channel, 1, 1)
        #print(f'f_scale: {f_scale.shape}')
        return x * f_scale    


""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

#from .unet_parts import *


class UNet3SE(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        c64 = 64
        super(UNet3SE, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = TripleConv(n_channels, c64)
        self.se_inc = SE_block(c64, 2)
        self.down1 = Down(c64, c64)
        self.se_down1 = SE_block(c64, 2)
        self.down2 = Down(c64, c64)
        self.se_down2 = SE_block(c64, 2)
        self.down3 = Down(c64, c64)
        #factor = 2 if bilinear else 1
        self.se_down3 = SE_block(c64, 2)
        self.down4 = Down(c64, c64)
        self.se_down4 = SE_block(c64, 2)
        self.down5 = Down(c64, c64)
        self.se_down5 = SE_block(c64, 4)
        self.up1 = Up(c64, c64, bilinear)
        self.se_up1 = SE_block(c64, 2)
        self.up2 = Up(c64, c64, bilinear)
        self.se_up2 = SE_block(c64, 2)
        self.up3 = Up(c64, c64, bilinear)
        self.se_up3 = SE_block(c64, 2)
        self.up4 = Up(c64, c64, bilinear)
        self.se_up4 = SE_block(c64, 2)
        self.up5 = Up(c64, c64, bilinear)
        self.se_up5 = SE_block(c64, 2)
        self.outc = OutConv(c64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.se_inc(x1)
        x2 = self.down1(x1)
        x2 = self.se_down1(x2)
        x3 = self.down2(x2)
        x3 = self.se_down2(x3)
        x4 = self.down3(x3)
        x4 = self.se_down3(x4)
        x5 = self.down4(x4)
        x5 = self.se_down4(x5)
        x6 = self.down5(x5)        
        x6 = self.se_down5(x6)        
        x = self.up1(x6, x5)
        x = self.se_up1(x)
        #print(f'x after up1: {x.shape}')
        x = self.up2(x, x4)
        x = self.se_up2(x)
        x = self.up3(x, x3)
        x = self.se_up3(x)
        x = self.up4(x, x2)
        x = self.se_up4(x)
        x = self.up5(x, x1)
        x = self.se_up5(x)
        
        logits = self.outc(x)
        return logits
