import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, RandomCrop, ToPILImage, CenterCrop, Resize

from os import listdir
from os.path import join
from PIL import Image
import sys
import random

random.seed (1)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png"]) #, ".jpg", ".jpeg"


def load_img(filepath):
    img = Image.open(filepath)#.convert('RGB')#.convert('YCbCr')
    #y, _, _ = img.split()
    #return y
    return img


class AlignRandomCrop(object):
    def __init__(self, lr_crop_size, up_scale_factor):
        self.lr_crop_size = lr_crop_size
        self.up_scale_factor = up_scale_factor
        self.hr_crop_size = lr_crop_size * up_scale_factor

    def get_params(self, lr_img, hr_img):
        lr_w, lr_h = lr_img.size
        lr_th, lr_tw = self.lr_crop_size, self.lr_crop_size
        x1 = random.randint(0, lr_w - lr_tw)
        y1 = random.randint(0, lr_h - lr_th)
        #print ("{}-{}, {}-{}".format (x1, x1+lr_th, y1, y1+lr_tw))
        return x1, y1, lr_th, lr_tw

    @staticmethod
    def transform(self, lr_img, hr_img, x1, y1, h, w):
        # no self here, all required params will have to specified in params
        lr_crop_img = lr_img.crop((x1, y1, x1+h, y1+w))
        hr_crop_img = hr_img.crop((x1*self.up_scale_factor, y1*self.up_scale_factor, (x1+h)*self.up_scale_factor, (y1+w)*self.up_scale_factor))
        return lr_crop_img, hr_crop_img


    def __call__(self, lr_img, hr_img):

        params = self.get_params(lr_img, hr_img)
        lr_crop_img, hr_crop_img = self.transform(self, lr_img, hr_img, *params)
        #print ("here: ", lr_crop_img.size, hr_crop_img.size)
        return lr_crop_img, hr_crop_img

class DatasetFromFolder(data.Dataset):
    def __init__(self, lr_image_dir, hr_image_dir, patch_size, up_sampling):
        super(DatasetFromFolder, self).__init__()
        self.lr_image_filenames = [join(lr_image_dir, x) for x in listdir(lr_image_dir) if is_image_file(x)]
        self.hr_image_filenames = [join(hr_image_dir, x) for x in listdir(hr_image_dir) if is_image_file(x)]

        self.align_crop = AlignRandomCrop (patch_size, up_sampling)

        self.input_transform = Compose ([Resize (patch_size, interpolation=Image.BICUBIC), ToTensor ()])  
        self.target_transform = Compose ([ToTensor ()]) # RandomCrop


    def __getitem__(self, index): #TODO: load LR and HR using file_name
        input = load_img(self.lr_image_filenames[index])
        target = load_img(self.hr_image_filenames[index])

        #print ("1.", input.size, target.size)
        input, target = self.align_crop (input, target)
        #print ("2.", input.size, target.size)

        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        #print (input.shape, target.shape)
        #sys.exit (-1)

        return input, target

    def __len__(self):
        return len(self.lr_image_filenames)

        
         
