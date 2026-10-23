import torch.nn as nn
import torch
from .blocks import ConvolutionalBlock, DeConvolutionalBlock, BottleConvolutionalBlock
from .base_net import BaseNet
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, num_input_channels, rep_dim, filters, use_batchnorm, activation_unit, voxel, hipo_diff, fc_out, VolumeDifference, VolumePassthrough=0.0):
        super().__init__()

        # VolumePassthrough = alpha > 0: the output of the network is [f(x), alpha * v(volumes)] where
        # v = (relu(-z_total_volume), relu(z_asymmetry)) is computed from the volumes WITHOUT parameters.
        # Deep SVDD minimises ||out - c||^2, so a trainable path from the volumes to the output would
        # just be driven to a constant; a fixed path cannot collapse, it only adds
        # alpha^2 * ||v - c_v||^2 to the anomaly score (c_v = mean of v on the train set).
        self.volume_passthrough = float(VolumePassthrough or 0.0)
        if self.volume_passthrough > 0:
            # train-set statistics of (total volume, |L-R|/(L+R)), set by ModelTrainerSVDD.init_volume_stats
            self.register_buffer('vfeat_mean', torch.zeros(2))
            self.register_buffer('vfeat_std', torch.ones(2))


        # Downsampling branch
        n_hippo_Volume = 0
        down_channels = [num_input_channels] + filters[:-1]
        self.down_blocks = [
            ConvolutionalBlock(in_channel,out_channel,use_batchnorm, activation_unit)
            for in_channel, out_channel in zip(down_channels, down_channels[1:])
        ]
        self.maxpool = nn.MaxPool3d(kernel_size=2)
        self.hipo_diff = hipo_diff
        self.fc_out = fc_out
        self.VolumeDifference = VolumeDifference

        n_hipo_diff = 1
        if  self.hipo_diff == 'cat':
            n_hipo_diff = 2

        if self.VolumeDifference:
            n_hippo_Volume = 1
            # Volume normalization (voxels -> z-score). The defaults are the old fixed constants
            # (v-395)/4650, which squeezed the volumes of every structure into a band ~20x narrower
            # than the image features, so the volume could not influence the output. The real
            # statistics of the train set are set by ModelTrainerSVDD.init_center_c_full() before
            # c is computed, and travel with the checkpoint (they are buffers).
            self.register_buffer('volume_mean', torch.tensor(395.0))
            self.register_buffer('volume_std', torch.tensor(4650.0))

        if self.fc_out:
            self.fcSVDD1 = nn.Linear(n_hippo_Volume + filters[-1]*voxel*voxel*voxel, self.fc_out, bias=False)
            self.fcSVDD2 = nn.Linear(n_hipo_diff*self.fc_out,rep_dim, bias=False)
        else:
            self.fcSVDD = nn.Linear(n_hippo_Volume + n_hipo_diff*filters[-1]*voxel*voxel*voxel, rep_dim, bias=False)

        #self.fcSVDD = nn.Linear(filters[-1]*voxel*voxel*voxel, rep_dim, bias=False)
        #self.fcSVDD1 = nn.Linear(filters[-1]*voxel*voxel*voxel, 256, bias=False)
        #self.fcSVDD2 = nn.Linear(256,32, bias=False)
        self.bottleneck = BottleConvolutionalBlock(filters[-2], filters[-1], use_batchnorm, activation_unit)
        #self.bn1d = nn.BatchNorm1d(rep_dim, eps=1e-04, affine=False)

        for i, block in enumerate(self.down_blocks):
            self.add_module(f"down_block_{i}", block)

    def volume_features(self, left_volume, right_volume):
        '''Fixed (parameter-free) one-sided anomaly features of the volumes: low total volume, high asymmetry.'''
        left_volume, right_volume = left_volume.float(), right_volume.float()
        total = left_volume + right_volume
        f = torch.stack([total, torch.abs(left_volume - right_volume) / total], 1)
        z = (f - self.vfeat_mean) / self.vfeat_std
        return torch.stack([torch.relu(-z[:, 0]), torch.relu(z[:, 1])], 1)

    def forward(self, inputs):
        inputs, left_volume, right_volume = inputs[0], inputs[1], inputs[2] #$$$#
        raw_left_volume, raw_right_volume = left_volume, right_volume   # (left_volume is normalized below)
        # Downsampling branch for one hippocampus
        inputs1 = inputs[:,0,:,:,:].to(torch.float)
        out = torch.unsqueeze(inputs1, 1) #torch.Size([12, 1, 64, 64, 64])
        for block in self.down_blocks:
            out = block(out)
            out = self.maxpool(out)
        # Bottleneck layer
        out = self.bottleneck(out)  #torch.Size([12, 64, 16, 16, 16])
        out = self.maxpool(out)     #torch.Size([12, 64, 8, 8, 8])
        out1 = out.view(out.size(0), -1)     #torch.Size([12, 32768])

        if self.VolumeDifference:
            left_volume = ((left_volume.float() - self.volume_mean)/self.volume_std)
            out1 = torch.cat((out1, torch.unsqueeze(left_volume, 1)), 1) #adding the right volume  #torch.Size([12, 32769])

        if self.fc_out:
            out1 = self.fcSVDD1(out1)
            out1 = F.relu(out1)
        #out1 = self.fcSVDD1(out1)
        #out1 = F.relu(out1)

        # Downsampling branch for the other hippocampus
        inputs2 = inputs[:,1,:,:,:].to(torch.float)
        out = torch.unsqueeze(inputs2, 1) #torch.Size([12, 1, 64, 64, 64])
        for block in self.down_blocks:
            out = block(out)
            out = self.maxpool(out)
        # Bottleneck layer
        out = self.bottleneck(out)  #torch.Size([12, 64, 16, 16, 16])
        out = self.maxpool(out)     #torch.Size([12, 64, 8, 8, 8])
        out2 = out.view(out.size(0), -1)     #torch.Size([12, 32768])

        if self.VolumeDifference:
            right_volume = ((right_volume.float() - self.volume_mean)/self.volume_std)
            out2 = torch.cat((out2, torch.unsqueeze(right_volume, 1)), 1) #adding the right volume  #torch.Size([12, 32769])

        if self.fc_out:
            out2 = self.fcSVDD1(out2)
            out2 = F.relu(out2)
        #out2 = self.fcSVDD1(out2)
        #out2 = F.relu(out2)

        # concatenation or substraction of both hippocampi
        if self.hipo_diff == 'sub':
            out = torch.sub(out1, out2)
        elif self.hipo_diff == 'cat':
            out = torch.cat((out1, out2), 1)

        #out = torch.sub(out1, out2)
        #out = torch.cat((out1, out2), 1)

        #1a/b experiment
        if self.fc_out:
            ##out = self.bn1d(self.fcSVDD2(out))  #torch.Size([12, 128])
            out = self.fcSVDD2(out)
        else:
            #2a/b experiment
            ##out = self.bn1d(self.fc1(out))  #torch.Size([12, 128])  
            out = self.fcSVDD(out)

        #out = self.fcSVDD2(out)  #torch.Size([12, 128])
        #out = self.fcSVDD(out)
        if self.volume_passthrough > 0:
            out = torch.cat((out, self.volume_passthrough * self.volume_features(raw_left_volume, raw_right_volume)), 1)
        return out



class LeNet_ELU_SVDD(BaseNet):

    def __init__(self, filters=None, encoder_class : object = Encoder, **config):
        ''' 
        Constructor.
        -------
        Inputs:
            config: a ConfigParser object with the model configuration
        '''
        super().__init__(**config)
        
        # this model is a patch-based unet
        self.name = self.name + '-unet'
        # number of channels of each conv layer
        self.filters = filters
        self.voxel = int(64/(2**len(filters)))
        self.rep_dim = self.rep_dim
        self.fc_out = self.fc_out
        self.hipo_diff = self.hipo_diff
        self.VolumeDifference = self.VolumeDifference
        self.VolumePassthrough = self.VolumePassthrough
        self.encoder = encoder_class(self.num_input_channels, self.rep_dim,  self.filters, self.use_batchnorm, self.activation_unit, self.voxel, self.hipo_diff, self.fc_out, self.VolumeDifference, self.VolumePassthrough)
        

    def forward(self, inputs):
        # print('forward_LeNet_ELU_SVDD')

        out = self.encoder(inputs)

        return out

    def set_volume_stats(self, mean, std):
        '''Set the volume normalization (only when VolumeDifference is on).'''
        self.encoder.volume_mean.fill_(float(mean))
        self.encoder.volume_std.fill_(float(std))

    def set_volume_feature_stats(self, mean, std):
        '''Train-set mean/std of (total volume, asymmetry) used by VolumePassthrough.'''
        self.encoder.vfeat_mean.copy_(torch.as_tensor(mean, dtype=torch.float))
        self.encoder.vfeat_std.copy_(torch.as_tensor(std, dtype=torch.float))

    def predict(self, inputs):
        print('predict_LeNet_ELU_SVDD')

        out = self.encoder(inputs)
        return out

        # self.c = torch.zeros(32).to('cuda')

        # x1 = inputs[:,0,:,:,:].to(torch.float)
        # x2 = inputs[:,1,:,:,:].to(torch.float)    
        # x1 = self.encoder(x1) 
        # x2 = self.encoder(x2) 
        # y_hat = torch.sub(x1, x2)
        # # print ('cantaaa', y_hat.shape, self.c.shape)
        # return y_hat

        # scores = torch.sum((y_hat - self.c) ** 2, dim=1)
        # return scores
