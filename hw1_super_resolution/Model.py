import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual

class UpSampleBlock (nn.Module):
    def __init__ (self, in_channels, up_scale_factor):
        super(UpSampleBlock, self).__init__()

        self.conv = nn.Conv2d (in_channels=in_channels, out_channels=in_channels * up_scale_factor * 2, kernel_size=3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle (up_scale_factor)
        self.prelu = nn.PReLU()

    def forward (self, x):
        return self.prelu (self.shuffler (self.conv (x)))

class Model (nn.Module):
    def __init__ (self, nu_RB, up_scale_factor):
        super(Model, self).__init__()

        self.nu_RB = nu_RB
        self.up_scale_factor = up_scale_factor

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.PReLU()
        )

        for i in range (nu_RB):
            self.add_module ("RB{}".format (i+1), ResidualBlock ())

        self.block2 = nn.Sequential ( 
            nn.Conv2d (in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d (64)
        )

        self.nu_UB = int(math.log(up_scale_factor, 2))
        for i in range (self.nu_UB):
            self.add_module ("upsample{}".format (i+1), UpSampleBlock (64, up_scale_factor))

        self.conv = nn.Conv2d (in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3) #TODO

    def forward (self, x):
        x = self.block1 (x)
        y= x.clone ()

        for i in range (self.nu_RB):
            y = self.__getattr__ ("RB{}".format (i+1)) (y)

        x = self.block2 (y) + x

        for i in range (self.nu_UB):
            x = self.__getattr__ ("upsample{}".format (i+1)) (x)

        return self.conv (x)


