import torch 
import torch.nn as nn

def local_sep_conv (img, hori, vert, kernel_size, output=None):
    channels = img.size(0)
    size_w = img.size (1)
    size_h = img.size (2)
    size_w_out = size_w - kernel_size + 1
    size_h_out = size_h - kernel_size + 1


    if output is None:
        output = torch.zeros ((channels, size_w_out, size_w_out))
    
    for i in range (size_w_out):
        for j in range (size_h_out):
            patch = img [:, i:i+kernel_size, j:j+kernel_size]
            local_hori = hori [:, i, j]
            local_vert = vert [:, i, j].view (-1, 1)

            #print ("\n\n Here {}-{}: {}, {}, {}, {}".format (i, j, patch.shape, local_hori.shape, local_vert.shape, output.shape))

            output[:, i, j] = (patch * local_hori * local_vert).sum (dim=1).sum (dim=1) #NOTE
    return output

def sep_conv (img, hori, vert, kernel_size, output):
    batch = img.size (0)
    #print ("batch: {}".format (batch))
    
    for i in range (0, batch):
        #print ("sep_conv: {}".format (i))
        if i == 0:
            output[i,:,:,:] = local_sep_conv (img[i], hori[i], vert[i], kernel_size, output[i]) 
        else:
            output[i,:,:,:] = local_sep_conv (img[i], hori[i], vert[i], kernel_size) 


    return output

class SepConv (nn.Module):
    def __init__ (self, kernel_size):
        super (SepConv, self).__init__()
        self.kernel_size = kernel_size
    
    def forward (self, img, hori, vert):
        batch = img.size (0)
        channels = img.size (1)
        size_w = img.size (2)
        size_h = img.size (3)
        size_w_out = size_w - self.kernel_size + 1
        size_h_out = size_h - self.kernel_size + 1

        #assert img.size(2)  == img.size(3)
        #assert vert.size(0) == hori.size(0) == batch
        #assert vert.size(1) == hori.size(1) == self.kernel_size
        #assert vert.size(2) == hori.size(2) == vert.size(3) == vert.size(3) == size_out

        output = img.new().resize_(batch, channels, size_w_out, size_h_out).zero_()
        output = sep_conv (img, hori, vert, self.kernel_size, output)
        
        #print ("Here: {}".format (output.shape))

        return output


