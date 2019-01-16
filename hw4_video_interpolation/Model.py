import torch
import torch.nn as nn
import sys
from Model_sep_conv import *
from libs.sepconv.SeparableConvolution import SeparableConvolution
from separable_convolution import SeparableConvolutionSlow

def conv_module (in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential (
        nn.Conv2d (in_channels, out_channels, kernel_size, stride, padding), nn.ReLU(), 
        nn.Conv2d (out_channels, out_channels, kernel_size, stride, padding), nn.ReLU(), 
        nn.Conv2d (out_channels, out_channels, kernel_size, stride, padding), nn.ReLU()
    )

def kernel_module ():
    return nn.Sequential (
        nn.Upsample (scale_factor=2, mode="bilinear", align_corners=True), 
        conv_module (in_channels=64, out_channels=51, kernel_size=3, stride=1, padding=1)
    )

class Encoder (nn.Module):
    def __init__ (self):
        super (Encoder, self).__init__()
        self.avg_pool = nn.AvgPool2d (kernel_size=2, stride=2) 

        self.conv_b1 = conv_module (in_channels=6, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv_b2 = conv_module (in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv_b3 = conv_module (in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv_b4 = conv_module (in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.conv_b5 = conv_module (in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.conv_b6 = conv_module (in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

    def forward (self, x):
        #print ("\n\n\n-------")
        #print ("x -> {}".format (x.shape))
        x_32 = self.conv_b1 (x)

        #print ("x_32 -> {}".format (x_32.shape))
        pool_64 = self.avg_pool (x_32)
        #print ("pool_64 -> {}".format (pool_64.shape))
        x_64 = self.conv_b2 (pool_64)

        #print ("x_64 -> {}".format (x_64.shape))
        pool_128 = self.avg_pool (x_64)
        x_128 = self.conv_b3 (pool_128)

        pool_256 = self.avg_pool (x_128)
        x_256 = self.conv_b4 (pool_256)

        pool_512 = self.avg_pool (x_256)
        x_512 = self.conv_b5 (pool_512)

        pool_512_2 = self.avg_pool (x_512)
        x_512_2 = self.conv_b6 (pool_512_2)

        return (x_64, x_128, x_256, x_512, x_512_2)

class Decoder (nn.Module):
    def __init__ (self):
        super (Decoder, self).__init__()
        self.up_sample = nn.Upsample (scale_factor=2, mode="bilinear", align_corners=True) 

        self.conv_b1 = conv_module (in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.conv_b2 = conv_module (in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv_b3 = conv_module (in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)

    def forward (self, x1, x2, x3, x4, x5): # --> 512*4*4

        #print ("{}, {}, {}, {}, {}".format (x1.shape, x2.shape, x3.shape, x4.shape, x5.shape))
        up_512 = self.up_sample (x5) # --> 512*8*8
        #print ("up_512: {}".format (up_512.shape))
        y = up_512 + x4

        #print ("\n\ny: {}".format (y.shape))
        x_256 = self.conv_b1 (y) # --> 256*8*8
        #print ("x_256: {}".format (x_256.shape))
        up_256 = self.up_sample (x_256) # --> 256*16*16
        #print ("up_256: {}".format (up_256.shape))
        y = up_256 + x3

        #print ("\n\ny: {}".format (y.shape))
        x_128 = self.conv_b2 (up_256) # --> 128*16*16 
        #print ("x_128: {}".format (x_128.shape))
        up_128 = self.up_sample (x_128) # --> 128*32*32
        #print ("up_128: {}".format (up_128.shape))
        y = up_128 + x2

        #print ("\n\ny: {}".format (y.shape))
        x_64 = self.conv_b3 (up_128) # --> 64*32*32
        #print ("x_64: {}".format (x_64.shape))
        up_64 = self.up_sample (x_64) # --> 64*64*64
        #print ("up_64: {}".format (up_64.shape))
        y = up_64 + x1
        #print ("y: {}".format (y.shape))

        #print ("y: {}".format (y.shape))

        return (y)

class Interpolation (nn.Module):
    def __init__ (self):
        super (Interpolation, self).__init__()
        self.block1 = kernel_module ()
        self.block2 = kernel_module ()
        self.block3 = kernel_module ()
        self.block4 = kernel_module ()
        
        #self.sep_conv_ = SepConv (51) 
        self.sep_conv_ = SeparableConvolution.apply
        #self.sep_conv_ = SeparableConvolutionSlow()

    def forward (self, x, i1, i2):
        k1v = self.block1 (x)
        k1h = self.block2 (x)
        k2v = self.block3 (x)
        k2h = self.block4 (x)

        y = self.sep_conv_ (i1, k1v, k1h) + self.sep_conv_ (i2, k2v, k2h)

        return y


class Model (nn.Module):
    def __init__ (self):
        super(Model, self).__init__()

        self.encoder = Encoder ()
        self.decoder = Decoder ()
        self.inter   = Interpolation ()

        self.pad = nn.ReplicationPad2d (51 // 2)

    

    def forward (self, x, i1, i2):
        
        x1, x2, x3, x4, x5 = self.encoder (x)
        
        out_decoder = self.decoder (x1, x2, x3, x4, x5)

        out = self.inter (out_decoder, self.pad (i1), self.pad (i2))

        return out

