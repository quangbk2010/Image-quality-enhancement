import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, RandomCrop, ToPILImage, CenterCrop, Resize

from os import listdir
from os.path import join
from PIL import Image
from skimage.transform import resize
import sys
import random

random.seed (1)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png"]) #, ".jpg", ".jpeg"


def load_img(filepath):
    img = Image.open(filepath)
    return img

class AlignRandomCrop (object):
    def __init__(self, crop_size):
        if crop_size == None:
            crop_size = 0
        self.crop_size = crop_size

    def get_params (self, blur_img, sharp_img):
        try:
            ldr_w, ldr_h, _ = blur_img.shape
        except Exception:
            ldr_w, ldr_h = blur_img.size
            
        ldr_th, ldr_tw = self.crop_size, self.crop_size
        x1 = random.randint  (0, ldr_w - ldr_tw)
        y1 = random.randint (0, ldr_h - ldr_th)
        return x1, y1, ldr_th, ldr_tw

    @staticmethod
    def transform (self, blur_img, sharp_img, x1, y1, h, w):
        # no self here, all required params will have to specified in params
        try:
            blur_crop_img = blur_img[x1:x1+h, y1:y1+w, :]
            sharp_crop_img = sharp_img[x1:x1+h, y1:y1+w, :]
        except Exception:
            blur_crop_img = blur_img.crop((x1, y1, x1+h, y1+w))
            sharp_crop_img = sharp_img.crop((x1, y1, x1+h, y1+w))

        return blur_crop_img, sharp_crop_img


    def __call__ (self, blur_img, sharp_img):

        params = self.get_params(blur_img, sharp_img)
        blur_crop_img, sharp_crop_img = self.transform(self, blur_img, sharp_img, *params)
        return blur_crop_img, sharp_crop_img


class DatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, patch_size, crop=False):
        super(DatasetFromFolder, self).__init__()

        self.crop = crop

        sub_dir_list = [join (data_dir, x) for x in listdir (data_dir)]
        
        blur_image_dir = [join (x, "blur_gamma") for x in sub_dir_list]
        sharp_image_dir =  [join (x, "sharp") for x in sub_dir_list]

        self.blur_image_filenames = [] 
        for d in blur_image_dir:
            for x in listdir (d):
                self.blur_image_filenames += [join (d, x)]

        self.sharp_image_filenames = [] 
        for d in sharp_image_dir:
            for x in listdir (d):
                self.sharp_image_filenames += [join (d, x)]

        if self.crop:
            self.align_crop = AlignRandomCrop (patch_size)

        self.transform = Compose ([ToTensor ()]) # RandomCrop

        img0 = load_img(self.blur_image_filenames[0])


        #print ("/2: {}, {}".format (int (img0.size[1]/2), int (img0.size[0]/2)))
        #print ("/4: {}, {}".format (int (img0.size[1]/4), int (img0.size[0]/4)))

        if patch_size == None:
            self.transform2 = Compose ([Resize ((int (img0.size[1]/2), int (img0.size[0]/2)), interpolation=Image.BICUBIC), 
                                        ToTensor ()])  
            self.transform3 = Compose ([Resize ((int (img0.size[1]/4), int (img0.size[0]/4)), interpolation=Image.BICUBIC), 
                                        ToTensor ()])  
        else:
            self.transform2 = Compose ([Resize (int (patch_size/2), interpolation=Image.BICUBIC), ToTensor ()])  
            self.transform3 = Compose ([Resize (int (patch_size/4), interpolation=Image.BICUBIC), ToTensor ()])  

    def __getitem__(self, index): 
        input = load_img(self.blur_image_filenames[index])
        target = load_img(self.sharp_image_filenames[index])

        #print ("\n1. {}, {}".format (input.size, target.size))
        if self.crop:
            input, target = self.align_crop (input, target)
        #print ("2. {}, {}".format (input.size, target.size))

        input1 = self.transform (input)
        target1 = self.transform (target)

        input2 = self.transform2 (input)
        target2 = self.transform2 (target)

        input3 = self.transform3 (input)
        target3 = self.transform3 (target)

        #print (input1.shape, target1.shape)
        #print (input2.shape, target2.shape)
        #print (input3.shape, target3.shape)
        #sys.exit (-1)

        return (input1, target1, input2, target2, input3, target3)

    def __len__(self):
        return len(self.blur_image_filenames)

