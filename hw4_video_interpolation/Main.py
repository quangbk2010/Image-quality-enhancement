from __future__ import print_function, division, unicode_literals

import argparse 
import os
import sys
from math import log10

import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from   torch.autograd import Variable
from   torch.utils.data import DataLoader

import torchvision
import torchvision.datasets as datasets
from tqdm import tqdm

from Model import Model 
from Dataset import * 

from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage



parser = argparse.ArgumentParser ()
parser.add_argument ("--mode", type=str, default="train", help="Phase: train or test")
parser.add_argument ("--train_data_dir", type=str, default="./data/train", help="The location of training dataset")
parser.add_argument ("--val_data_dir", type=str, default="./data/val", help="The location of validation dataset")

parser.add_argument ("--test_data_dir", type=str, default="./data/test", help="The location of test dataset")
parser.add_argument ("--model_dir", type=str, default="./saved_model", help="The location of saved models")
parser.add_argument ("--patch_size", type=int, default=128, help="The patch size")

parser.add_argument ("--nu_epoch", type=int, default=1, help="The number of epochs") #100
parser.add_argument ("--res_epoch", type=int, default=0, help="Restored epoch corresponds to the pre-trained model") #100
parser.add_argument ("--batch_size", type=int, default=4, help="The batch size")
parser.add_argument ("--lr", type=int, default=0.001, help="The learning rate") #0.0001

args = parser.parse_args ()

if not os.path.exists (args.model_dir):
    os.makedirs (args.model_dir)

device = torch.device ("cuda:0" if torch.cuda.is_available() else "cpu")
print ("-> Using {}".format (device))



#---- Get the tuples (3 continuous frames) ----#
print ("\n\n--- Loading training datasets...")
train_tuples_ = TupleFromFolder (args.train_data_dir)
train_tuples = train_tuples_.get_tuples ()
print ("There are {} training tuples".format (len (train_tuples)))

print ("\n\n--- Loading validation datasets...")
val_tuples_ = TupleFromFolder (args.val_data_dir)
val_tuples  = val_tuples_.get_tuples ()
print ("There are {} validation tuples".format (len (val_tuples)))


#---- Preprocess tuples to be usable in Model ----#
train_set = DatasetFromTuples (train_tuples, args.patch_size, aug=True,  crop=True)
val_set   = DatasetFromTuples (val_tuples,   args.patch_size, aug=False, crop=True)


train_data_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
val_data_loader   = DataLoader(dataset=val_set,   batch_size=1,               shuffle=False)



print ("\n\n--- Building model...")
model = Model ()
if torch.cuda.device_count() > 1:
    print ("There are %d GPUs" % (torch.cuda.device_count()))
    #model = nn.DataParallel (model)
model = model.to (device)


l1_loss = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def compute_psnr(sr_im_orig, hr_im_orig):
    # Note: function only apply for single image (not batch)
    if len(sr_im_orig.shape) == 4 and len(hr_im_orig.shape) == 4:
        sr_im = sr_im_orig.squeeze(0)
        hr_im = hr_im_orig.squeeze(0)
    else:
        print('Dimension of sr_im and hr_im different')
        return

    SCALE = 1
    _, h, w = sr_im.shape
    sr_im = sr_im[:, :h - h % SCALE, :w - w % SCALE]
    boundarypixels = SCALE
    sr_im = sr_im[:, boundarypixels:-boundarypixels, boundarypixels:-boundarypixels]

    _, h, w = hr_im.shape
    hr_im = hr_im[:, :h - h % SCALE, :w - w % SCALE]
    boundarypixels = SCALE
    hr_im = hr_im[:, boundarypixels:-boundarypixels, boundarypixels:-boundarypixels]

    MAX_PIXEL_VALUE = 1.0
    squared_error = (hr_im - sr_im)**2
    mse = torch.mean(squared_error)
    psnr = 10.0 *torch.log10(MAX_PIXEL_VALUE / mse)
    return psnr

def train (epoch):
    print ("\n -> Training, epoch: {}".format (epoch))
    epoch_loss = 0
    print (len (train_data_loader))
    #sys.exit (-1)
    for iteration, batch in tqdm(enumerate (train_data_loader, 1)):
        input, i1, i2, target = batch[0].requires_grad_().to (device), batch[1].to (device), batch[2].to (device), batch[3].to (device)
        #print ("\n\n\nTest in train(): {}, {}, {}, {}".format (input.shape, i1.shape, i2.shape, target.shape))
        
        optimizer.zero_grad ()
        try:
            out = model(input, i1, i2)
        except:
            break
        
        #print ("\n\n Out: {}, Target: {}".format (out.shape, target.shape))

        loss = l1_loss (out.to(device), target)
        epoch_loss += loss.item()

        loss.backward ()
        optimizer.step ()
        #break

        print("Iter: %d, loss: %.4f\n" % (iteration, loss))

    print("Epoch: %d, Training loss: %.4f\n" % (epoch, epoch_loss / len (train_data_loader)))

    return model

def test (model, data_loader):
    print ("Testing...")
    avg_psnr = 0
    with torch.no_grad ():

        test_loss = 0
        for batch in data_loader:
            input, i1, i2, target = batch[0].to (device), batch[1].to (device), batch[2].to (device), batch[3].to (device)

            prediction = model (input, i1, i2)

            loss = l1_loss (prediction.to(device), target) 

            test_loss += loss.item ()

        print("===> Finish testing, test loss: %.4f\n" % (test_loss / len (data_loader)))

    return test_loss

        
def save_model (epoch):
    model_path = "{}/model_epoch_{}.pth".format(args.model_dir, epoch)
    torch.save(model.state_dict(), model_path)
    print("Checkpoint saved to {}".format(model_path))

    
def get_tuples (imgs):
    
    no_frames = len (imgs)
    no_tuples = no_frames // 3
    tuples = []

    for i in range (no_tuples):
        i1, i2, i3 = imgs[i * 3], imgs[i * 3 + 1], imgs[i * 3 + 2] 
        tuples.append ((i1, i2, i3))

    return tuples               

if args.mode == "train":
    for epoch in range (args.nu_epoch):
        trained_model = train (epoch)
        if (epoch == args.res_epoch) or (epoch == 0):
            save_model (epoch)

else:
    print ("\nRestore the trained model ----...")
    res_model = Model ()
    #if torch.cuda.device_count() > 1:
    #    res_model = nn.DataParallel (res_model)
    res_model = res_model.to (device)

    model_path = "{}/model_epoch_{}.pth".format(args.model_dir, args.res_epoch)
    res_model.load_state_dict (torch.load (model_path))

    if args.mode == "val":
        print ("\nTest the model on the validation set...")
        #test_loss = test (res_model, val_data_loader)

        print ("\n\n Resolve SR image...")
        with torch.no_grad():
            sub_dir_list = [join (args.val_data_dir, x) for x in listdir (args.val_data_dir)]
            frame_list = []

            psnr = 0
            total_tuples = 0
            for d in sub_dir_list:
                frame_dir = "{}/60fps".format (d)
                frame_list = [join (frame_dir, x) for x in listdir (frame_dir)]

                frame_list.sort ()
                tuples = get_tuples (frame_list)
                no_tuples = len (tuples)

                for i in range (no_tuples):
                    transform = ToTensor () 
                    frames = tuples[i]

                    try:
                        i1, i2, i3 = load_img (frames)
                    except:
                        raise ValueError ("\n\n[!] #images in the tuples is not 3!")

                    print (i1.size)
                    print (i3.size)

                    i1 = transform (i1).unsqueeze (0).detach().to (device)
                    i2 = transform (i2).unsqueeze (0).detach().to (device)
                    i3 = transform (i3).unsqueeze (0).detach().to (device)

                    print (i1.shape)
                    print (i3.shape)

                    input = torch.cat ((i1, i3), dim=1)                    

                    print (input.shape)
                    #print (input.shape[2] % pow (2, 5))
                    #print (input.shape[3] % pow (2, 5))
                    pad = nn.ReplicationPad2d ((0,0,0,16))

                    input = pad (input)
                    i1 = pad (i1)
                    i3 = pad (i3)
                    print ("\n\n--------")
                    print (input.shape)
                    #print (input.shape[2] % pow (2, 5))
                    #print (input.shape[3] % pow (2, 5))

                    target = i2.to (device)

                    output = res_model (input, i1.to (device), i3.to (device))
                    output = output [:, :, :720, :]

                    psnr += compute_psnr (target, output)

                    output = output.clamp(0, 1)

                    print (target.shape)
                    print (output.shape)
                    interpo_image = ToPILImage () (output[0].data.cpu())
                    print ("[*] Save into: {}/out_{}.png".format (frame_dir, i))
                    interpo_image.save ("{}/out_{}.png".format (frame_dir, i))

                total_tuples += no_tuples
                #break
            
            psnr /= total_tuples
            print ("\n\n[*] Average PSNR of the images in the validation set (only 5 missing images): {}".format (psnr))



            


