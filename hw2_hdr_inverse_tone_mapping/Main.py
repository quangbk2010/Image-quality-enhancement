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
parser.add_argument ("--train_hdr_data_dir", type=str, default="./data/train/HDR", help="The location of training dataset")
parser.add_argument ("--train_ldr_data_dir", type=str, default="./data/train/LDR", help="The location of training dataset")
parser.add_argument ("--val_hdr_data_dir", type=str, default="./data/val/HDR", help="The location of validation dataset")
parser.add_argument ("--val_ldr_data_dir", type=str, default="./data/val/LDR", help="The location of validation dataset")
parser.add_argument ("--hdr_res_dir", type=str, default="./data/val/hdr_res", help="The location of validation dataset")

parser.add_argument ("--test_data_dir", type=str, default="./data/test", help="The location of test dataset")
parser.add_argument ("--model_dir", type=str, default="./saved_model", help="The location of saved models")
parser.add_argument ("--patch_size", type=int, default=320, help="The size of the LR image")
parser.add_argument ("--clip_flag", type=bool, default=True, help="If the input image is not divided by 32 -> resize by padding and clip it again")

parser.add_argument ("--nu_epoch", type=int, default=10, help="The number of epochs") #100
parser.add_argument ("--res_epoch", type=int, default=9, help="Restored epoch corresponds to the pre-trained model") #100
parser.add_argument ("--batch_size", type=int, default=8, help="The batch size")
parser.add_argument ("--lr", type=int, default=0.00005, help="The learning rate") #0.0001

args = parser.parse_args ()

if not os.path.exists (args.model_dir):
    os.makedirs (args.model_dir)
if not os.path.exists (args.hdr_res_dir):
    os.makedirs (args.hdr_res_dir)

device = torch.device ("cuda" if torch.cuda.is_available() else "cpu")
print ("-> Using {}".format (device))




print ("\n\n--- Loading datsets...")
train_set = DatasetFromFolder (args.train_ldr_data_dir, args.train_hdr_data_dir, args.patch_size, args.clip_flag)

val_set   = DatasetFromFolder (args.val_ldr_data_dir, args.val_hdr_data_dir, args.patch_size, args.clip_flag)

train_data_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
val_data_loader   = DataLoader(dataset=val_set, batch_size=1, shuffle=False)


print ("\n\n--- Building model...")
model = Model (device)
if torch.cuda.device_count() > 1:
    print ("There are %d GPUs" % (torch.cuda.device_count()))
    model = nn.DataParallel (model)
model = model.to (device)


# Using masked loss, only use information near saturated image regions
thr = 0.05 
def compute_msk (dataset):
    msk = torch.max (dataset, 1)[0] 
    mat_0 = torch.FloatTensor ([0.0]).expand_as (msk)
    mat_1 = torch.FloatTensor ([1.0]).expand_as (msk)
    msk = torch.min (mat_1, torch.max (mat_0, msk-1.0+thr)/thr) 
    msk = msk.resize (dataset.shape[0], 1, dataset.shape[2], dataset.shape[3])
    print ("\nmsk.(min, max, mean): (%.4f, %.4f, %.4f)"%(msk.min(), msk.max(), msk.mean()))
    return msk

mse = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train (epoch):
    print ("\n -> Training, epoch: {}".format (epoch))
    epoch_loss = 0
    n = 0
    for iteration, batch in tqdm(enumerate (train_data_loader, 1)):
        input, target = batch[0].requires_grad_().to (device), batch[1].requires_grad_().to (device)
        
        msk = compute_msk (input.cpu())
        optimizer.zero_grad ()
        out = model(input*255.0)

        loss = mse (out*msk.to(device), torch.log (target + 1.0/255.0)*msk.to(device)) 
        epoch_loss += loss.item()
        n += 1
        loss.backward ()
        optimizer.step ()
        print("Iter: %d, loss: %.4f\n" % (iteration, loss))
    print("Epoch: %d, Training loss: %.4f\n" % (epoch, epoch_loss / n))

    return model

def test (model, data_loader):
    print ("Testing...")
    avg_psnr = 0
    with torch.no_grad ():
        for batch in data_loader:
            input, target = batch[0].to (device), batch[1].to (device)

            msk = compute_msk (input.cpu())
            prediction = model (input*255.0)

            loss = mse (prediction*msk.to(device), torch.log (target + 1.0/255.0)*msk.to(device)) 

        print("===> Finish testing, test loss: %.4f\n" % (loss.item()))

    return loss

        
def save_model (epoch):
    model_path = "{}/model_epoch_{}.pth".format(args.model_dir, epoch)
    torch.save(model.state_dict(), model_path)
    print("Checkpoint saved to {}".format(model_path))

def get_pred_final (x, out_unet):
    # The final prediction of the model: combine the predicted pixels with the linearized LDR input image using a blend value-alpha
    x_lin = torch.pow (x, 2.0)
    out_unet_exp = torch.exp (out_unet) - 1.0/255

    # Define alpha
    alpha = torch.max (x, 1)[0].to(device) 
    x1 = torch.FloatTensor ([1.0]).expand_as (alpha).to(device)
    x2 = torch.FloatTensor ([0.0]).expand_as (alpha).to(device)
    x3 = torch.max (x2, alpha-1.0+thr)/thr
    alpha = torch.min (x1, x3).to(device) 
    print ("\nalpha.(min, max, mean): (%.4f, %.4f, %.4f)"%(alpha.min(), alpha.max(), alpha.mean()))

    # Alpha blending
    out = (1-alpha).resize (1, alpha.shape[0], alpha.shape[1], alpha.shape[2]) * x_lin.to(device) + alpha.resize (1, alpha.shape[0], alpha.shape[1], alpha.shape[2]) * out_unet_exp.to(device)

    return out
    

if args.mode == "train":
    for epoch in range (args.nu_epoch):
        trained_model = train (epoch)
        test_loss = test (trained_model, val_data_loader)
        if (epoch == args.res_epoch) or (epoch == 0) or (test_loss > 0 and test_loss < 0.01):
            save_model (epoch)

else:
    print ("\nRestore the trained model...")
    res_model = Model (device)
    if torch.cuda.device_count() > 1:
        res_model = nn.DataParallel (res_model)
        res_model = res_model.to (device)

    model_path = "{}/model_epoch_{}.pth".format(args.model_dir, args.res_epoch)
    res_model.load_state_dict (torch.load (model_path))

    if args.mode == "val":
        print ("\nTest the model on the validation set...")
        #test_loss = test (res_model, val_data_loader)

        print ("\n\n Resolve SR image...")
        with torch.no_grad():
            for i in [44, 45, 46]:
                image_name = "C{}_LDR.tif".format (i)
                image_path = "{}/".format (args.val_ldr_data_dir)
                image = load_img (image_path + image_name, args.clip_flag, ldr=True) 

                target_name = "C{}_HDR.hdr".format (i)
                target_path = "{}/".format (args.val_hdr_data_dir)
                target = load_img (target_path + target_name, args.clip_flag, ldr=False) 

                convert_to_tensor = ToTensor()
                image = convert_to_tensor(image).unsqueeze (0).detach() 
                image = image.to (device)

                out = res_model (image*255.0)
                final_pred = get_pred_final (image, out)

                out = (out - out.min ()) / (out.max() - out.min())

                imsave ("{}/hdr_C{}_unet.hdr".format (args.hdr_res_dir, i), out[0].data.cpu().numpy())
                imsave ("{}/hdr_C{}.hdr".format (args.hdr_res_dir, i), final_pred[0].data.cpu().numpy())
                imsave ("{}/hdr_C{}_1000.hdr".format (args.hdr_res_dir, i), 1000* (final_pred[0].data.cpu().numpy()-final_pred[0].data.cpu().numpy().min()))
                #import pdb; pdb.set_trace ()
                #imageio.imwrite ("./data/val/hdr_res/target_C{}.hdr".format (i), target)

    else:
        print ("\nTest the model on the test set...")
        test_set  = DatasetFromFolder (args.test_data_dir, args.image_size, args.up_scale_factor, args.clip_flag)
        test_data_loader  = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)

        test (res_model, test_data_loader)
