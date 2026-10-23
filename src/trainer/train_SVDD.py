#!/usr/bin/env python3

# conda activate deepsvd
# python train_SVDD.py --configAE '../../configs/AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filters.yml' --configSVDD '../../configs/AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filtersSVDD.yml' --structure HIP
# python train_SVDD.py --configAE '../../configs/AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filters.yml' --configSVDD '../../configs/AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filtersSVDD.yml' --structure THA

import os
import re
from argparse import ArgumentParser
from utils import yaml
import torch.nn as nn
from pytorch_lightning import seed_everything
import Models
from lightning import getModules
from lightning import getModulesSVDD


def train(configAE, configSVDD, fast_dev_run=False, accelerator="gpu", checkpoint=None, pretrained=True, train_ae=True):
    seed_everything(42, workers=True)

    if pretrained and train_ae:
    	trainer, modelTrainer, dataModule = getModules(configAE, fast_dev_run, accelerator)
    	print('deep_SVDD.pretrain')
    	trainer.fit(modelTrainer, dataModule, ckpt_path=checkpoint) # train the AEmodel
    elif pretrained:
    	print('deep_SVDD.pretrain: skipped, reusing the latest autoencoder checkpoint of ' + configAE['name'])
    # self.init_network_weights_from_pretraining()
    trainerSVDD, modelTrainerSVDD, dataModuleSVDD = getModulesSVDD(configSVDD, fast_dev_run, accelerator, pretrained, ae_name=configAE['name'])
    print('deep_SVDD.train')
    trainerSVDD.fit(modelTrainerSVDD, dataModuleSVDD, ckpt_path=checkpoint) # train the SVDDmodel  


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--configAE',type=str, help='Config file full path')
    parser.add_argument('--configSVDD',type=str, help='Config file full path')
    parser.add_argument('--structure', type=str, required=True, choices=['THA', 'PAL', 'PUT', 'HIP', 'AMY'],
                         help='Brain structure to train on (required). Resolves any {structure} placeholder '
                              'found in the config/split files, and keeps each structure\'s results/checkpoints '
                              'in separate folders so that training different structures with the same shared '
                              'config files never overwrites each other.')
    parser.add_argument('--fast-dev-run', default=False, action='store_true', help='if set, only run a single batch')
    parser.add_argument('--accelerator', type=str, default="gpu", help='accelerator to use')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    parser.set_defaults(pretrained=True)
    parser.add_argument('--skip-ae-training', dest='train_ae', action='store_false',
                        help='do not train the autoencoder: initialize the SVDD from the latest autoencoder '
                             'checkpoint already in ../results (e.g. to compare SVDD variants on the same autoencoder)')
    parser.set_defaults(train_ae=True)

    params = parser.parse_args()

    # prefix the experiment name with the structure (if given) so that the
    # results/checkpoints/logger folders of each structure never collide,
    # even though all 5 structures share the exact same config files.
    name_prefix = (params.structure + '_') if params.structure else ''

    config_name = os.path.basename(params.configAE)
    config_name = re.sub(r'.ya?ml$', '', config_name)
    configAE = {'name': name_prefix + config_name, **yaml.load(params.configAE, structure=params.structure)}

    config_name = os.path.basename(params.configSVDD)
    config_name = re.sub(r'.ya?ml$', '', config_name)
    configSVDD = {'name': name_prefix + config_name, **yaml.load(params.configSVDD, structure=params.structure)}

    train(configAE, configSVDD, params.fast_dev_run, params.accelerator, params.checkpoint, params.pretrained, params.train_ae)
