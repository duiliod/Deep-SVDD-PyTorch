import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet
USE_GAP = True

class MNIST_LeNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32
        self.pool = nn.MaxPool2d(2, 2)

        if USE_GAP == True:
            self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2) #small square 28×28 pixel 
            self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
            self.conv2 = nn.Conv2d(8, 16, 5, bias=False, padding=2)
            self.bn2 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))# 16 x 1 x 1
            self.fc1 = nn.Linear(16, self.rep_dim, bias=False) #16 to 8
        else:
            self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2) #small square 28×28 pixel 
            self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
            self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
            self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
            self.fc1 = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False) #196 to 32

    def forward(self, x):
        if USE_GAP == True:
            x = self.conv1(x)
            x = self.pool(F.leaky_relu(self.bn1(x)))
            x = self.conv2(x)
            x = self.pool(F.leaky_relu(self.bn2(x)))  #-> [200, 16, 7, 7]
            x = self.avgpool(x) #->  [200, 16, 1, 1]
            x = torch.flatten(x, 1) #->  [200, 16]
            x = self.fc1(x) #->  [200, 32]
        else:
            x = self.conv1(x)
            x = self.pool(F.leaky_relu(self.bn1(x)))
            x = self.conv2(x)
            x = self.pool(F.leaky_relu(self.bn2(x)))
            x = x.view(x.size(0), -1)
            x = self.fc1(x) #->[200, 32]

        return x


class MNIST_LeNet_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32
        self.pool = nn.MaxPool2d(2, 2)

        if USE_GAP == True:
            self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2) #small square 28×28 pixel 
            self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
            self.conv2 = nn.Conv2d(8, 16, 5, bias=False, padding=2)
            self.bn2 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))# 16 x 1 x 1
            self.fc1 = nn.Linear(16, self.rep_dim, bias=False) #16 to 8

            self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
            self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
            self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
            self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
            self.deconv3 = nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)
            
        else:
            # Encoder (must match the Deep SVDD network above)
            self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
            self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
            self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
            self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False) #torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
            self.fc1 = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False) #torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)

            # Decoder
            self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
            self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
            self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
            self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
            self.deconv3 = nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)

    def forward(self, x):
        if USE_GAP == True:
            x = self.conv1(x)
            x = self.pool(F.leaky_relu(self.bn1(x)))
            x = self.conv2(x)
            x = self.pool(F.leaky_relu(self.bn2(x)))  #-> [200, 16, 7, 7]
            x = self.avgpool(x) #->  [200, 16, 1, 1]
            x = torch.flatten(x, 1) #->  [200, 16]
            x = self.fc1(x) #->  [200, 32]
            ####Dec
            x = x.view(x.size(0), 2, 4, 4)  #-> [200, 2, 2, 2]
            x = F.interpolate(F.leaky_relu(x), scale_factor=2)  #2*4*4
            x = self.deconv1(x)                                 #4*4*4
            x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
            x = self.deconv2(x)
            x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
            x = self.deconv3(x)
            x = torch.sigmoid(x)

        else:
            x = self.conv1(x)
            x = self.pool(F.leaky_relu(self.bn1(x)))
            x = self.conv2(x) #-> [200, 4, 14, 14]
            x = self.pool(F.leaky_relu(self.bn2(x))) #-> [200, 4, 7, 7])
            x = x.view(x.size(0), -1) #-> [200, 196]
            x = self.fc1(x)
            x = x.view(x.size(0), int(self.rep_dim / 16), 4, 4)
            x = F.interpolate(F.leaky_relu(x), scale_factor=2)
            x = self.deconv1(x)
            x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
            x = self.deconv2(x)
            x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
            x = self.deconv3(x)
            x = torch.sigmoid(x) #->[200, 1, 28, 28]

        return x
