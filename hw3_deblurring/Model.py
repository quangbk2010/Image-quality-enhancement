import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import sys


class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        #self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        #self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        #residual = self.bn1(residual)
        residual = self.prelu(residual)
        #residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual

class RB9 (nn.Module):
    def __init__ (self):
        super (RB9, self).__init__()
        for i in range (9):
            self.add_module ("RB{}".format (i+1), ResidualBlock ())
    
    def forward (self, x):
        for i in range (9):
            x = self.__getattr__ ("RB{}".format (i+1)) (x)
        return x


class Layer (nn.Module):
    def __init__ (self, in_channels):
        super(Layer, self).__init__()

        self.conv1 = nn.Conv2d (in_channels=in_channels, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.prelu = nn.PReLU()
        self.rb9 = RB9 ()

        self.conv2 = nn.Conv2d (in_channels=64, out_channels=3, kernel_size=5, stride=1, padding=2)

    def forward (self, x):
        x = self.conv1 (x)
        x = self.prelu (x)
        x = self.rb9 (x)
        x = self.conv2 (x)

        return x

class UpConv (nn.Module):
    def __init__ (self, kernel_size):
        super (UpConv, self).__init__ ()
        self.upConv = nn.ConvTranspose2d (in_channels=3, out_channels=3, kernel_size=kernel_size, padding=1, stride=2, output_padding=1)

    def forward (self, x):
        #print ("\n\n---> x: {}".format (x.shape))
        x = self.upConv (x)
        #print ("\n\n---> x__: {}".format (x.shape))

        return x


class Model (nn.Module):
    def __init__ (self):
        super(Model, self).__init__()
        
        self.layer1 = Layer (6)
        self.layer2 = Layer (6)
        self.layer3 = Layer (3)
        
        self.upConv1 = UpConv (3) #65)
        self.upConv2 = UpConv (3) #129) 


    def forward (self, b1, b2, b3):

        #print ("b1: {}".format (b1.shape))
        #print ("b2: {}".format (b2.shape))
        #print ("b3: {}".format (b3.shape))
        #sys.exit (-1)
        l3 = self.layer3 (b3)
        #print ("l3: {}".format (l3.shape))
        
        l3_upconv = self.upConv1 (l3)
        #l3_upconv = UpConv ((181, 321)) (l3)
        #print ("l3_upconv: {}".format (l3_upconv.shape))

        x = torch.cat ((b2, l3_upconv), 1)
        #print ("x: {}".format (x.shape))

        l2 = self.layer2 (x)
        #print ("l2: {}".format (l2.shape))

        l2_upconv = self.upConv2 (l2)
        #l2_upconv = UpConv ((361, 641)) (l2)
        #print ("l2_upconv: {}".format (l2_upconv.shape))

        x = torch.cat ((b1, l2_upconv), 1)

        #print ("x: {}".format (x.shape))

        l1 = self.layer1 (x)
        #print ("l1: {}".format (l1.shape))

        return l1, l2, l3


