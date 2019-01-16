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
parser.add_argument ("--val_data_dir", type=str, default="./data/test", help="The location of validation dataset")
parser.add_argument ("--res_dir", type=str, default="./data/sharp_res", help="The location of test result")

parser.add_argument ("--model_dir", type=str, default="./saved_model", help="The location of saved models")
parser.add_argument ("--patch_size", type=int, default=256, help="The crop size")

parser.add_argument ("--nu_epoch", type=int, default=100, help="The number of epochs") 
parser.add_argument ("--res_epoch", type=int, default=99, help="Restored epoch corresponds to the pre-trained model") #100
parser.add_argument ("--batch_size", type=int, default=8, help="The batch size")
parser.add_argument ("--lr", type=int, default=0.001, help="The learning rate") #0.0001

args = parser.parse_args ()


if not os.path.exists (args.model_dir):
    os.makedirs (args.model_dir)
if not os.path.exists (args.res_dir):
    os.makedirs (args.res_dir)


device = torch.device ("cuda:3" if torch.cuda.is_available() else "cpu")
print ("-> Using {}".format (device))




print ("\n\n--- Loading datasets...")
if args.mode == "train":
    train_set = DatasetFromFolder (args.train_data_dir, args.patch_size, crop=True)
    train_data_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)

if args.mode == "val":
    val_set   = DatasetFromFolder (args.val_data_dir, None, crop=False)
    val_data_loader   = DataLoader(dataset=val_set, batch_size=1, shuffle=False)

if args.mode == "test":
    test_set   = DatasetFromFolder (args.res_dir, None, crop=False)
    test_data_loader   = DataLoader(dataset=test_set, batch_size=1, shuffle=False)


print ("\n\n--- Building model...")
model = Model ()
"""if torch.cuda.device_count() > 1:
    print ("There are %d GPUs" % (torch.cuda.device_count()))
    model = nn.DataParallel (model)"""
model = model.to (device)


def multi_scale_loss (L, S):
    mse = nn.MSELoss()
    err1 = mse (L[0], S[0]) * 3 * 256 * 256
    err2 = mse (L[1], S[1]) * 3 * 128 * 128
    err3 = mse (L[2], S[2]) * 3 * 64  * 64
    err = (err1 + err2 + err3) / (3 * 256 * 256 + 3 * 128 * 128 + 3 * 64 * 64)
    return err

criterion = nn.L1Loss() 
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
        input, target, input2, target2, input3, target3 = batch[0].requires_grad_().to (device), batch[1].requires_grad_().to (device), batch[2].requires_grad_().to (device), batch[3].requires_grad_().to (device), batch[4].requires_grad_().to (device), batch[5].requires_grad_().to (device)

        optimizer.zero_grad ()
        out, out2, out3 = model(input, input2, input3)

        loss = multi_scale_loss ([out, out2, out3], [target, target2, target3])
        loss.backward ()
        optimizer.step ()

        epoch_loss += loss.item()

    epoch_loss /= len (train_data_loader)
    print("Epoch: %d, Training loss: %.4f\n" % (epoch, epoch_loss))
    return model

def test (model, data_loader):
    print ("Testing...")
    avg_psnr = 0
    with torch.no_grad ():
        i = 0
        for batch in data_loader:
            #input, target = batch[0].to (device), batch[1].to (device)
            input, target, input2, target2, input3, target3 = batch[0].requires_grad_().to (device), batch[1].requires_grad_().to (device), batch[2].requires_grad_().to (device), batch[3].requires_grad_().to (device), batch[4].requires_grad_().to (device), batch[5].requires_grad_().to (device)

            prediction, _, _ = model (input, input2, input3)

            if args.mode == "test":
                print ("\n\n Resolve deblurred image...")
                prediction = prediction.clamp(0, 1)
                hr_image = ToPILImage () (prediction[0].data.cpu())
                hr_image.save ("data/sharp_res/test/our_sharp/test_{}.png".format (i))

            #psnr = compute_psnr(prediction, target)
            #avg_psnr += psnr
            i += 1

        #print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(data_loader)))
                
def save_model (epoch):
    model_path = "./saved_model/model_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_path)
    print("Checkpoint saved to {}".format(model_path))

if args.mode == "train":
    for epoch in range (args.nu_epoch):
        trained_model = train (epoch)
        #test (trained_model, val_data_loader)
        if (epoch % 5 == 4) or (epoch == 0):
            save_model (epoch)

else:
    print ("\nRestore the trained model...")
    res_model = Model ()
    """if torch.cuda.device_count() > 1:
        res_model = nn.DataParallel (res_model)
        res_model = res_model.to (device)"""

    res_model = res_model.to (device)
    model_path = "./saved_model/model_epoch_{}.pth".format(args.res_epoch)

    state_dict = torch.load (model_path)

    #------ If the pytorch version is different with the version used to train the model------#
    for x in ["layer1.rb9.RB1.bn2.num_batches_tracked", "layer1.rb9.RB2.bn2.num_batches_tracked", "layer1.rb9.RB3.bn2.num_batches_tracked", "layer1.rb9.RB4.bn2.num_batches_tracked", "layer1.rb9.RB5.bn2.num_batches_tracked", "layer1.rb9.RB6.bn2.num_batches_tracked", "layer1.rb9.RB7.bn2.num_batches_tracked", "layer1.rb9.RB8.bn2.num_batches_tracked", "layer1.rb9.RB9.bn2.num_batches_tracked", "layer2.rb9.RB1.bn2.num_batches_tracked", "layer2.rb9.RB2.bn2.num_batches_tracked", "layer2.rb9.RB3.bn2.num_batches_tracked", "layer2.rb9.RB4.bn2.num_batches_tracked", "layer2.rb9.RB5.bn2.num_batches_tracked", "layer2.rb9.RB6.bn2.num_batches_tracked", "layer2.rb9.RB7.bn2.num_batches_tracked", "layer2.rb9.RB8.bn2.num_batches_tracked", "layer2.rb9.RB9.bn2.num_batches_tracked", "layer3.rb9.RB1.bn2.num_batches_tracked", "layer3.rb9.RB2.bn2.num_batches_tracked", "layer3.rb9.RB3.bn2.num_batches_tracked", "layer3.rb9.RB4.bn2.num_batches_tracked", "layer3.rb9.RB5.bn2.num_batches_tracked", "layer3.rb9.RB6.bn2.num_batches_tracked", "layer3.rb9.RB7.bn2.num_batches_tracked", "layer3.rb9.RB8.bn2.num_batches_tracked", "layer3.rb9.RB9.bn2.num_batches_tracked"]:
        del state_dict[x]

    res_model.load_state_dict (state_dict)

    if args.mode == "val":
        print ("\nTest the model on the validation set...")
        len_test = len (val_data_loader)
        if len_test > 0:
            test (res_model, val_data_loader)
        else:
            raise ValueError ("Please check the validation set")

    else:
        print ("\nTest the model on the validation set...")
        len_test = len (test_data_loader)
        if len_test > 0:
            test (res_model, test_data_loader)
        else:
            raise ValueError ("Please check the test set")
        

