import torch
import torch.nn as nn
import sys

def down_sample (in_channels, out_channels, nu_conv, kernel_size=3, stride=1, padding=1):
    if nu_conv == 2:
        block = nn.Sequential (
            nn.Conv2d (in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU (),
            nn.Conv2d (out_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU ()
        )
    elif nu_conv == 3:
        block = nn.Sequential (
            nn.Conv2d (in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU (),
            nn.Conv2d (out_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU (),
            nn.Conv2d (out_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU ()
        )
    else:
        raise ValueError ("Check the number of convolution layers! (only 2 or 3)")

    return block
    

class Encoder (nn.Module):
    def __init__ (self):
        super (Encoder, self).__init__()

        self.block1 = down_sample (3, 64, 2)
        self.pool1 = nn.MaxPool2d (kernel_size=2, stride=2, dilation=1)

        self.block2 = down_sample (64, 128, 2)
        self.pool2 = nn.MaxPool2d (kernel_size=2, stride=2, dilation=1)

        self.block3 = down_sample (128, 256, 3)
        self.pool3 = nn.MaxPool2d (kernel_size=2, stride=2, dilation=1)

        self.block4 = down_sample (256, 512, 3)
        self.pool4 = nn.MaxPool2d (kernel_size=2, stride=2, dilation=1)

        self.block5 = down_sample (512, 512, 3)
        self.pool5 = nn.MaxPool2d (kernel_size=2, stride=2, dilation=1)

    def forward (self, x):
        x_copy = x.clone()
        before_pool1 = self.block1 (x)
        x = self.pool1 (before_pool1)

        before_pool2 = self.block2 (x)
        x = self.pool2 (before_pool2)

        before_pool3 = self.block3 (x)
        x = self.pool3 (before_pool3)

        before_pool4 = self.block4 (x)
        x = self.pool4 (before_pool4)

        before_pool5 = self.block5 (x)
        y = self.pool5 (before_pool5)
        
        skip_layers = [x_copy, before_pool1, before_pool2, before_pool3, before_pool4, before_pool5]

        return (x_copy, before_pool1, before_pool2, before_pool3, before_pool4, before_pool5, y)


def up_sample (in_channels, mid_channels, out_channels, kernel_size_conv=1, kernel_size_deconv=4, stride_conv=1, stride_deconv=2, padding_conv=0, padding_deconv=1):
    block = nn.Sequential (
        nn.Conv2d           (in_channels=in_channels,   out_channels=mid_channels, kernel_size=kernel_size_conv,    stride=stride_conv,     padding=padding_conv),
        nn.BatchNorm2d      (mid_channels),
        nn.ReLU (),
        nn.ConvTranspose2d  (in_channels=mid_channels,  out_channels=out_channels, kernel_size=kernel_size_deconv,  stride=stride_deconv,   padding=padding_deconv,    bias=False)
    )
    return block

class LatentRep (nn.Module):
    def __init__ (self):
        super (LatentRep, self).__init__()

        self.block = up_sample (in_channels=512, mid_channels=512, out_channels=512, kernel_size_conv=3, padding_conv=1)

    def forward (self, x):
        return self.block (x)

class Decoder (nn.Module):
    def __init__ (self):
        super (Decoder, self).__init__()

        self.block1 = up_sample (in_channels=1024, mid_channels=512, out_channels=512)
        self.block2 = up_sample (in_channels=1024, mid_channels=512, out_channels=256)
        self.block3 = up_sample (in_channels=512,  mid_channels=256, out_channels=128)
        self.block4 = up_sample (in_channels=256,  mid_channels=128, out_channels=64)
        self.block5 = nn.Sequential (
            nn.Conv2d (in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d (in_channels=64,  out_channels=3,  kernel_size=1, stride=1, padding=0)
        ) 
        self.conv = nn.Conv2d (in_channels=6,   out_channels=3,  kernel_size=1, stride=1, padding=0)

    def forward (self, x, skip_layers):
        # skip-connection domain transformation, from LDR encoder to log HDR decoder
        for i in range (6):
            skip_layers[i] = torch.log (torch.pow (1.0/255 * skip_layers[i], 2.0) + 1.0/255) 

        x = self.block1 (torch.cat ((x, skip_layers[5]), 1))
        x = self.block2 (torch.cat ((x, skip_layers[4]), 1))
        x = self.block3 (torch.cat ((x, skip_layers[3]), 1))
        x = self.block4 (torch.cat ((x, skip_layers[2]), 1))
        x = self.block5 (torch.cat ((x, skip_layers[1]), 1))
        x = self.conv   (torch.cat ((x, skip_layers[0]), 1))
        return x


class Model (nn.Module):
    def __init__ (self, device):
        super(Model, self).__init__()

        self.encoder = Encoder ()
        self.latent_rep = LatentRep ()
        self.decoder = Decoder ()
        self.device = device
        

    def forward (self, x):
        # Output of the Unet model
        x0, x1, x2, x3, x4, x5, after_encoder = self.encoder (x)
        skip_layers = [x0, x1, x2, x3, x4, x5]
        
        after_latent_rep = self.latent_rep (after_encoder)
        
        out_unet = self.decoder (after_latent_rep, skip_layers)
        
        return out_unet

