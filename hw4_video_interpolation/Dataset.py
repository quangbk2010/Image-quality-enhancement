import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, RandomCrop, ToPILImage, CenterCrop, Resize, RandomRotation, RandomVerticalFlip, RandomHorizontalFlip
import torch

from os import listdir
from os.path import join
from PIL import Image
import imageio
#imageio.plugins.freeimage.download()

import sys
import random


#random.seed (1)


def is_image_file (filename):
    return any (filename.endswith(extension) for extension in [".png"]) 


def load_img (filepaths):
    if len (filepaths) != 3:
        return -1
    
    i1 = Image.open(filepaths[0])
    i2 = Image.open(filepaths[1])
    i3 = Image.open(filepaths[2])
    
    return i1, i2, i3



class AlignRandomCrop (object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def get_params (self, img1, img2, img3):
        try:
            input_w, input_h, _ = img1.shape
        except Exception:
            input_w, input_h = img1.size

        input_th, input_tw = self.crop_size, self.crop_size
        x1 = random.randint  (0, input_w - input_tw)
        y1 = random.randint (0, input_h - input_th)
        return x1, y1, input_th, input_tw

    @staticmethod
    def transform (img1, img2, img3, x1, y1, h, w):
        # no self here, all required params will have to specified in params
        try:
            crop_img1 = img1[x1:x1+h, y1:y1+w, :]
            crop_img2 = img2[x1:x1+h, y1:y1+w, :]
            crop_img3 = img3[x1:x1+h, y1:y1+w, :]
        except:
            crop_img1 = img1.crop((x1, y1, x1+h, y1+w)) 
            crop_img2 = img2.crop((x1, y1, x1+h, y1+w)) 
            crop_img3 = img3.crop((x1, y1, x1+h, y1+w)) 

        return crop_img1, crop_img2, crop_img3


    def __call__ (self, img1, img2, img3):

        params = self.get_params(img1, img2, img2)
        crop_img1, crop_img2, crop_img3 = self.transform(img1, img2, img2, *params)
        return crop_img1, crop_img2, crop_img3


#---- Get list of tuples (3 frames) from a random video ----#
class TupleFromFolder ():
    def __init__ (self, data_dir):
        sub_dir_list = [join (data_dir, x) for x in listdir (data_dir)]
        
        self.image_filenames = [] 

        # Each video contains a list of frames sorted as the order in the Alphabet
        # Eg. videos[0] = {<frame_0_0>, <frame_0_1>, ...}
        self.videos = {}

        for i, d in enumerate (sub_dir_list):
            if is_image_file (join (d, listdir(d)[0])):
                frame_list = []
                for x in listdir (d):
                    frame_list.append (join (d, x))


            else:
                frame_list = []
                frame_dir = "{}/60fps".format (d)
                frame_list = [join (frame_dir, x) for x in listdir (frame_dir)]

            frame_list.sort ()
            self.videos[i] = frame_list
            print ("[*] Video {} has {} frames".format (i, len (frame_list)))

        self.no_videos = len (self.videos)
        print ("[*] There are {} videos".format (self.no_videos))

        self.tuples = []

    def get_tuples (self):
        
        for x in range (self.no_videos):
            no_frames = len (self.videos[x])
            no_tuples = no_frames // 3

            for i in range (no_tuples):
                i1, i2, i3 = self.videos[x][i * 3], self.videos[x][i * 3 + 1], self.videos[x][i * 3 + 2] 
                self.tuples.append ((i1, i2, i3))

        return self.tuples

    def __len__ (self):
        return len (self.tuples)


#---- Each tuple (3 frames) will be processed (random_crop, transform to tensor, ...) correspondingly ----#
class DatasetFromTuples (data.Dataset):
    def __init__ (self, tuples, patch_size, aug, crop):
        super (DatasetFromTuples, self).__init__()

        # Data saved in tuples
        self.tuples = tuples
        self.nu_tuples = len (tuples)
        

        # Data processing: random crop, randomly flipping horizontally and vertically, rotating by 90 degrees
        self.aug = aug
        self.crop = crop
        if crop:
            self.align_crop = AlignRandomCrop (patch_size)

        if aug:
            data_aug = [RandomRotation((90, 90)), RandomVerticalFlip(1.0), RandomHorizontalFlip(1.0)] # Choose randomly one of the augmentation techniques
            self.transform = Compose ([random.sample (data_aug, 1)[0], ToTensor ()]) 
        else:
            self.transform = Compose ([ToTensor ()]) 


    def __getitem__ (self, index): 

        #---- From a specific video, load tupple (PIL) (i1,i2,i3) ---> Tensors: input (i1,i3), target (i2) -----#
        frames = self.tuples[index]

        #print ("\n[D] Frames: {}".format (frames))

        try:
            i1, i2, i3 = load_img (frames)
        except:
            print ("\n\n[!] #images in the tuples is not 3!")
            return


        #---- Random crop -----#
        if self.crop:
            i1, i2, i3 = self.align_crop (i1, i2, i3)


        #---- Transform to tensor -----#
        if self.transform:
            i1 = self.transform (i1)
            i2 = self.transform (i2)
            i3 = self.transform (i3)

        input = torch.cat ((i1, i3), dim=0) #NOTE
        target = i2

        #print ("Test: {}, {}, {}, {}".format (input.shape, i1.shape, i3.shape, target.shape))

        return input, i1, i3, target


    def __len__ (self):
        return self.nu_tuples

        
         
