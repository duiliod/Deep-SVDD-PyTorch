#https://towardsdatascience.com/pytorch-step-by-step-implementation-3d-convolution-neural-network-8bf38c70e8b3

# importing the libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split
# for evaluating the model
from sklearn.metrics import accuracy_score


# PyTorch libraries and modules
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
import h5py
# from plot3D import *

import torch
import torch.utils.data

# To begin with, since the dataset is a bit specific, we use the following to helper functions to process them before giving them to the network.
def array_to_color(array, cmap="Oranges"):
    s_m = plt.cm.ScalarMappable(cmap=cmap)
    return s_m.to_rgba(array)[:,:-1]


def rgb_data_transform(data):
    data_t = []
    for i in range(data.shape[0]):
        data_t.append(array_to_color(data[i]).reshape(16, 16, 16, 3))
    return np.asarray(data_t, dtype=np.float32)


# Plus, the dataset is stored as h5 file, so to extract the actual data points, 
# we are required to read from h5 file, and use the to_categorical function to transform it into vectors. 
# In this step, we also prepare for cross-validation.
with h5py.File("../data3d/full_dataset_vectors.h5", "r") as hf:    

    # Split the data into training/test features/targets
    X_train = hf["X_train"][:]
    targets_train = hf["y_train"][:]
    X_test = hf["X_test"][:] 
    targets_test = hf["y_test"][:]

    # Determine sample shape
    sample_shape = (16, 16, 16, 3)

    # Reshape data into 3D format
    X_train = rgb_data_transform(X_train)
    X_test = rgb_data_transform(X_test)

    # Convert target vectors to categorical targets
    #targets_train = to_categorical(targets_train).astype(np.integer)
    #targets_test = to_categorical(targets_test).astype(np.integer)
print('training dataset shape: {} '.format(X_train.shape))
print('testing dataset shape: {}'.format(X_test.shape))

# Supposedly, the variables X_train/X_test should have respectively shape (10000, 16, 16, 16, 3) and (2000, 16, 16, 16, 3) 
# and targets_train/targets_test respectively (10000,) (2000,). But again we now convert all of that to PyTorch tensor format. 
# Which we do the following way.
train_x = torch.from_numpy(X_train).float()
train_y = torch.from_numpy(targets_train).long()
test_x = torch.from_numpy(X_test).float()
test_y = torch.from_numpy(targets_test).long()

batch_size = 100 #We pick beforehand a batch_size that we will use for the training

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(train_x,train_y)
test = torch.utils.data.TensorDataset(test_x,test_y)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

# For the model here is the architecture that we will be using:
# 2 sets of ConvMake:
# a 3d Convolution Layer with filter size (3x3x3) and stride (1x1x1) for both sets
# a Leaky Relu Activation function
# a 3d MaxPool Layer with filters size (2x2x2) and stride (2x2x2)
# 2 FC Layers with respectively 512 and 128 nodes.
# 1 Dropout Layer after first FC layer.
# The model is then translated into the code the following way:

num_classes = 10

# Create CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        self.conv_layer1 = self._conv_layer_set(3, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.fc1 = nn.Linear(2**3*64, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.LeakyReLU()
        self.batch=nn.BatchNorm1d(128)
        self.drop=nn.Dropout(p=0.15)        
        
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
    

    def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)
        
        return out

# Create AAE Model
class AAEModel(nn.Module):
    def __init__(self):
        super(AAEModel, self).__init__()
        
        self.conv_layer1 = self._conv_layer_set(3, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.fc1 = nn.Linear(2**3*64, 128)
        # self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.LeakyReLU()
        # self.batch=nn.BatchNorm1d(128)
        # self.drop=nn.Dropout(p=0.15)        
        self.up_conv_layer1 = self._up_conv_layer()
        self.up_conv_layer2 = self._up_conv_layer()

    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.BatchNorm3d(ch_out),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def _up_conv_layer_set(self, in_c, out_c):
        up_conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.BatchNorm3d(ch_out),
        nn.LeakyReLU(),
        # nn.MaxPool3d((2, 2, 2)),
        nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=align_corners),
        )
        return up_conv_layer

    def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        # out = self.relu(out)
        # out = self.batch(out)
        # out = self.drop(out)
        # out = self.fc2(out)
        out = out.view #TO DO
        out = self.up_conv_layer1(out)
        out = self.up_conv_layer2(out)
        out = torch.sigmoid(out)

        return out


#Definition of hyperparameters
n_iters = 4500
num_epochs = n_iters / (len(train_x) / batch_size)
num_epochs = int(num_epochs)

# Create CNN
model = CNNModel()
#model.cuda()
print(model)

# Cross Entropy Loss 
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# In terms of parameters pay attention to the number of input nodes on your 
# first Fully Convolutional Layer. Our data set being of shape (16,16,16,3), 
# that is how we are getting filtered outputs of size (2x2x2).

# CNN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        
        train = Variable(images.view(100,3,16,16,16))
        labels = Variable(labels)
        # Clear gradients
        optimizer.zero_grad()
        # Forward propagation
        outputs = model(train)
        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)
        # Calculating gradients
        loss.backward()
        # Update parameters
        optimizer.step()
        
        count += 1
        if count % 50 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                
                test = Variable(images.view(100,3,16,16,16))
                # Forward propagation
                outputs = model(test)

                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                
                # Total number of labels
                total += len(labels)
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        if count % 500 == 0:
            # Print Loss
            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))