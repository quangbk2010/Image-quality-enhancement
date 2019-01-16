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
parser.add_argument ("--train_hr_data_dir", type=str, default="./mscoco2017_train/HR", help="The location of training dataset")
parser.add_argument ("--train_lr_data_dir", type=str, default="./mscoco2017_train/LR", help="The location of training dataset")
parser.add_argument ("--val_hr_data_dir", type=str, default="./mscoco2017_val/HR", help="The location of validation dataset")
parser.add_argument ("--val_lr_data_dir", type=str, default="./mscoco2017_val/LR", help="The location of validation dataset")

parser.add_argument ("--test_data_dir", type=str, default="./test", help="The location of test dataset")
parser.add_argument ("--image_size", type=int, default=64, help="The size of the LR image")
parser.add_argument ("--up_scale_factor", type=int, default=2, help="The upsampling rate from LR image to HR image")

parser.add_argument ("--nu_epoch", type=int, default=100, help="The number of epochs") #100
parser.add_argument ("--res_epoch", type=int, default=99, help="Restored epoch corresponds to the pre-trained model") #100
parser.add_argument ("--batch_size", type=int, default=16, help="The batch size")
parser.add_argument ("--lr", type=int, default=0.001, help="The batch size") #0.0001

args = parser.parse_args ()



device = torch.device ("cuda:0" if torch.cuda.is_available() else "cpu")
print ("-> Using {}".format (device))




print ("\n\n--- Loading datsets...")
train_set = DatasetFromFolder (args.train_lr_data_dir, args.train_hr_data_dir, args.image_size, args.up_scale_factor)
val_set   = DatasetFromFolder (args.val_lr_data_dir, args.val_hr_data_dir, args.image_size, args.up_scale_factor)

train_data_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
val_data_loader   = DataLoader(dataset=val_set, batch_size=1, shuffle=False)


print ("\n\n--- Building model...")
model = Model (nu_RB=4, up_scale_factor=2)
if torch.cuda.device_count() > 1:
    print ("There are %d GPUs" % (torch.cuda.device_count()))
    model = nn.DataParallel (model)
model = model.to (device)



criterion = nn.L1Loss() 
mse = nn.MSELoss()
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
    for iteration, batch in tqdm(enumerate (train_data_loader, 1)):
        #input, target = Variable(batch[0]).to (device), Variable(batch[1]).to (device)
        input, target = batch[0].requires_grad_().to (device), batch[1].to (device)

        optimizer.zero_grad ()
        out = model(input)
        loss = criterion (out, target)
        loss.backward ()
        optimizer.step ()
        #print("Epoch: %d, Training loss: %.4f\n" % (epoch, loss.item()))
    return model

def test (model, data_loader):
    print ("Testing...")
    avg_psnr = 0
    with torch.no_grad ():
        for batch in data_loader:
            input, target = batch[0].to (device), batch[1].to (device)

            prediction = model (input)
            #mse = criterion(prediction, target)
            #psnr = 10 * log10(1 / mse.item())
            psnr = compute_psnr(prediction, target)
            avg_psnr += psnr

        print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(data_loader)))
                
def save_model (epoch):
    model_path = "./saved_model/model_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_path)
    print("Checkpoint saved to {}".format(model_path))

if args.mode == "train":
    for epoch in range (args.nu_epoch):
        trained_model = train (epoch)
        test (trained_model, val_data_loader)
        if (epoch % 5 == 4) or (epoch == 0):
            save_model (epoch)

else:
    print ("\nRestore the trained model...")
    res_model = Model (nu_RB=4, up_scale_factor=2)
    if torch.cuda.device_count() > 1:
        res_model = nn.DataParallel (res_model)
        res_model = res_model.to (device)

    model_path = "./saved_model/model_epoch_{}.pth".format(args.res_epoch)
    res_model.load_state_dict (torch.load (model_path))

    if args.mode == "val":
        print ("\nTest the model on the validation set...")
        test (res_model, val_data_loader)

        print ("\n\n Resolve SR image...")
        with torch.no_grad():
            for i in [15, 35, 55, 75, 95]:
                image_name = "00{}.png".format (i)
                image_path = "mscoco2017_val/LR/" 

                convert_to_tensor = ToTensor()

                image = load_img (image_path + image_name) 
                #import pdb; pdb.set_trace()
                #image = Variable (convert_to_tensor(image)).unsqueeze (0).detach()
                image = convert_to_tensor(image).requires_grad_().unsqueeze (0).detach()
                image = image.to (device)
                out = res_model (image)
                out = out.clamp(0, 1)

                hr_image = ToPILImage () (out[0].data.cpu())
                hr_image.save ("hr_result/hr_{}".format (image_name))

    else:
        #print ("\nTest the model on the test set...")
        #test_set  = DatasetFromFolder (args.test_data_dir, args.image_size, args.up_scale_factor)
        #test_data_loader  = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)

        #test (res_model, test_data_loader)
        img_dir = "./test_car"
        image_names = [join (img_dir, x) for x in listdir (img_dir)]

        print ("\n\n Resolve SR image...")
        with torch.no_grad():
            for image_name in image_names:
                convert_to_tensor = ToTensor()

                image = load_img (image_name) 
                #import pdb; pdb.set_trace()
                #image = Variable (convert_to_tensor(image)).unsqueeze (0).detach()
                image = convert_to_tensor(image).requires_grad_().unsqueeze (0).detach()
                if image.shape[1] == 1:
                    image = image.repeat (1,3,1,1)
                image = image.to (device)
                out = res_model (image)
                out = out.clamp(0, 1)

                hr_image = ToPILImage () (out[0].data.cpu())
                hr_image.save (image_name[:11] + "hr_" + image_name[11:])
