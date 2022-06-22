from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from sklearn.metrics import roc_auc_score

import logging
import time
import torch
import torch.optim as optim
import numpy as np


import nibabel as ni
import os, glob
import csv
from tqdm import tqdm
import shutil
import random
def load_mri_images(path, batch_size):
    filenames = [i for i in os.listdir(path) if i.endswith(".nii")] #and i.startswith("norm_023_S_0030")
    random.shuffle(filenames, random.random)
    n = 0
    while n < len(filenames):
        batch_image = []
        for i in range(n, n + batch_size):
            if i >= len(filenames):
                ##n = i
                break
            #print(filenames[i])
            image = ni.load(os.path.join(path, filenames[i]))
            image = np.array(image.dataobj)
            # image = np.pad(image, ((1,0), (1,0), (1, 0)), "constant", constant_values=0)
            image = torch.Tensor(image)
            image = torch.reshape(image, (1,1, 48, 48, 48))
            #image = (image - image.min()) / (image.max() - image.min())
            image = image / 255.
            batch_image.append(image)
        n += batch_size
        batch_image = torch.cat(batch_image)
        yield batch_image


class AETrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

    def train(self, dataset: BaseADDataset, ae_net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        logger.info('Starting pretraining...')
        start_time = time.time()
        ae_net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()

            #For hippocampal MRI
            for batch_images in tqdm(load_mri_images("/home/duilio/Downloads/seg/cropped_data", 8)):
                # print((batch_images.shape))  #torch.Size([8, 1, 48, 48, 48])
                inputs = batch_images #all labels are 0 cuz are the normal class
                # ae training data shape torch.Size([200, 1, 28, 28]) batch size 200
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        pretrain_time = time.time() - start_time
        logger.info('Pretraining time: %.3f' % pretrain_time)
        logger.info('Finished pretraining.')

        return ae_net

    def test(self, dataset: BaseADDataset, ae_net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Testing autoencoder...')
        # loss_epoch = 0.0
        # n_batches = 0
        # start_time = time.time()
        # idx_label_score = []
        # ae_net.eval()
        # with torch.no_grad():
        #     for data in test_loader:
        #         inputs, labels, idx = data #labels contain normal and anomalies
        #         # shape torch.Size([200, 1, 28, 28]) torch.Size([200]) torch.Size([200]
        #         inputs = inputs.to(self.device)
        #         outputs = ae_net(inputs)
        #         scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
        #         loss = torch.mean(scores)

        #         # Save triple of (idx, label, score) in a list
        #         idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
        #                                     labels.cpu().data.numpy().tolist(),
        #                                     scores.cpu().data.numpy().tolist()))

        #         loss_epoch += loss.item()
        #         n_batches += 1

        # logger.info('Test set Loss: {:.8f}'.format(loss_epoch / n_batches))

        # _, labels, scores = zip(*idx_label_score)
        # labels = np.array(labels)
        # scores = np.array(scores)

        # auc = roc_auc_score(labels, scores)
        # logger.info('Test set AUC: {:.2f}%'.format(100. * auc))

        # test_time = time.time() - start_time
        # logger.info('Autoencoder testing time: %.3f' % test_time)
        logger.info('Finished testing autoencoder.')
