import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, RandomCrop, ToPILImage, CenterCrop, Resize

from os import listdir
from os.path import join
from PIL import Image
import imageio
#imageio.plugins.freeimage.download()
from Util import *

import sys
import numpy as np
import random
import scipy

from skimage.transform import resize

random.seed (1)


def is_image_file (filename):
    return any (filename.endswith(extension) for extension in [".hdr", ".tif"]) #, ".jpg", ".jpeg"


def load_img (filepath, clip, ldr):
    img = imread (filepath)

    if clip:
        img = clip_img (img, ldr)
    if ldr:
        img = img.astype(np.float32)/255.0
        
    return img

def clip_img (in_img, ldr):
    sz_in = [float (x) for x in in_img.shape]
    sz_out_int = [int (np.round (sz_in[0] / 32.0) * 32), int (np.round (sz_in[1] / 32.0) * 32)]
    sz_out = [float (np.round (sz_in[0] / 32.0) * 32), float (np.round (sz_in[1] / 32.0) * 32)]

    sx, sy = (sz_out[0], sz_out[1])
    r_in = sz_in[1]/sz_in[0]
    r_out = sz_out[1]/sz_out[0]

    if r_out / r_in > 1.0:
        sx = sz_out[1]
        sy = sx/r_out
    else:
        sy = sz_out[0]
        sx = sy*r_out

    y0 = np.maximum(0.0, (sz_in[0]-sy)/2.0)
    x0 = np.maximum(0.0, (sz_in[1]-sx)/2.0)

    out_img = in_img[int(y0):int(y0+sy), int(x0):int(x0+sx),:]

    # Image resize and conversion to float
    #out_img = scipy.misc.imresize(out_img, sz_out_int)
    out_img = resize(out_img, sz_out_int, anti_aliasing=True)
    out_img = out_img.astype(np.float32)

    return out_img


class AlignRandomCrop (object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def get_params (self, ldr_img, hdr_img):
        ldr_w, ldr_h, _ = ldr_img.shape
        ldr_th, ldr_tw = self.crop_size, self.crop_size
        x1 = random.randint  (0, ldr_w - ldr_tw)
        y1 = random.randint (0, ldr_h - ldr_th)
        return x1, y1, ldr_th, ldr_tw

    @staticmethod
    def transform (self, ldr_img, hdr_img, x1, y1, h, w):
        # no self here, all required params will have to specified in params
        ldr_crop_img = ldr_img[x1:x1+h, y1:y1+w, :]
        hdr_crop_img = hdr_img[x1:x1+h, y1:y1+w, :]
        return ldr_crop_img, hdr_crop_img


    def __call__ (self, ldr_img, hdr_img):

        params = self.get_params(ldr_img, hdr_img)
        ldr_crop_img, hdr_crop_img = self.transform(self, ldr_img, hdr_img, *params)
        return ldr_crop_img, hdr_crop_img

class DatasetFromFolder (data.Dataset):
    def __init__ (self, ldr_image_dir, hdr_image_dir, patch_size, clip):
        super (DatasetFromFolder, self).__init__()
        self.ldr_image_filenames = [join(ldr_image_dir, x) for x in listdir(ldr_image_dir) if is_image_file(x)]
        #self.hdr_image_filenames = [join(hdr_image_dir, x) for x in listdir(hdr_image_dir) if is_image_file(x)]
        self.hdr_image_filenames = [x.replace ("LDR", "HDR") for x in self.ldr_image_filenames]
        self.hdr_image_filenames = [x.replace (".tif", ".hdr") for x in self.hdr_image_filenames]

        self.align_crop = AlignRandomCrop (patch_size)

        self.transform = Compose ([ToTensor ()]) # RandomCrop

        self.clip = clip


    def __getitem__ (self, index): #TODO: load LR and HR using file_name
        input = load_img (self.ldr_image_filenames[index], self.clip, True)
        target = load_img (self.hdr_image_filenames[index], self.clip, False)

        input, target = self.align_crop (input, target)

        if self.transform:
            input = self.transform (input)
            target = self.transform (target)

        return input, target

    def __len__ (self):
        return len (self.ldr_image_filenames)

        
         
