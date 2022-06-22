import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet
USE_GAP = True

class up_conv(nn.Module):
    "Reduce the number of features by 2 using Conv with kernel size 1x1x1 and double the spatial dimension using 3D trilinear upsampling"
    def __init__(self, ch_in, ch_out, k_size=1, scale=2, align_corners=False):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=k_size),
            nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=align_corners),
        )
    def forward(self, x):
	    return self.up(x)

class MNIST_LeNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32
        self.pool = nn.MaxPool2d(2, 2)


        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2) #small square 28×28 pixel 
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 16, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))# 16 x 1 x 1
        self.fc1 = nn.Linear(16, self.rep_dim, bias=False) #16 to 8


    def forward(self, x):

        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))  #-> [200, 16, 7, 7]
        x = self.avgpool(x) #->  [200, 16, 1, 1]
        x = torch.flatten(x, 1) #->  [200, 16]
        x = self.fc1(x) #->  [200, 32]


        return x


class MNIST_LeNet_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 16
        self.pool = nn.MaxPool3d((2, 2, 2))

        self.conv1 = nn.Conv3d(1, 8, kernel_size=(3, 3, 3), bias=False, padding=0) 
        self.bn1 = nn.BatchNorm3d(8)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), bias=False, padding=0)
        self.bn2 = nn.BatchNorm3d(16, eps=1e-04, affine=False)
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc1 = nn.Linear(16, self.rep_dim, bias=False) 

        self.deconv1 = up_conv(ch_in=2, ch_out=4, k_size=3, scale=2)
        self.bn3 = nn.BatchNorm3d(4, eps=1e-04, affine=False)
        self.deconv2 = up_conv(ch_in=4, ch_out=8, k_size=3, scale=2)
        self.bn4 = nn.BatchNorm3d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.Conv3d(8, 1, kernel_size=(3, 3, 3), bias=False, padding=1) 
            

    def forward(self, x):
        x = self.conv1(x)                                               #->torch.Size([8, 8, 46, 46, 46])
        x = self.pool(F.leaky_relu(self.bn1(x)))                         #->torch.Size([8, 8, 23, 23, 23])
        x = self.conv2(x)                                                #->torch.Size([8, 16, 21, 21, 21])
        x = self.pool(F.leaky_relu(self.bn2(x)))                         #-> torch.Size([8, 16, 10, 10, 10])
        x = self.avgpool(x)                                              #->  torch.Size([8, 16, 1, 1, 1])
        x = torch.flatten(x, 1)                                         #->  [8, 16]
        x = self.fc1(x)                                                  #->  [8, 32]
        ####Dec
        x = x.view(x.size(0), 2, 2, 2, 2)                                #-> torch.Size([8, 2, 2, 2, 2])
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)               #torch.Size([8, 2, 4, 4, 4])
        x = self.deconv1(x)                                              #-> torch.Size([8, 4, 4, 4, 4])
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)     #-> ([8, 4, 8, 8, 8])
        x = self.deconv2(x)                                              #-> #-> ([8, 8, 12, 12, 12])
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)     #-> ([8, 8, 24, 24, 24])
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)     #->  ([8, 8, 48, 48, 48])
        x = self.deconv3(x)                                              #->  torch.Size([8, 1, 48, 48, 48])
        x = torch.sigmoid(x)

        return x
