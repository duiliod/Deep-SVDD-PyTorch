#!/usr/bin/env python3
#sbatch -p GPU hipp_AEs.sh

import os
import re
from argparse import ArgumentParser

import torch.nn as nn
from pytorch_lightning import seed_everything

import Models
from lightning import getModules
from utils import yaml


def train(config, fast_dev_run=False, accelerator="gpu", checkpoint=None):
    seed_everything(42, workers=True)
    trainer, modelTrainer, dataModule = getModules(config, fast_dev_run, accelerator)    
    trainer.fit(modelTrainer, dataModule, ckpt_path=checkpoint) # train the model


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--config',type=str, help='Config file full path')
    parser.add_argument('--structure', type=str, default=None, choices=['THA', 'PAL', 'PUT', 'HIP', 'AMY'],
                         help='Brain structure to train on. Resolves any {structure} placeholder found in '
                              'the config/split files, and keeps each structure\'s results/checkpoints in '
                              'separate folders so that training different structures with the same shared '
                              'config files never overwrites each other.')
    parser.add_argument('--fast-dev-run', default=False, action='store_true', help='if set, only run a single batch')
    parser.add_argument('--accelerator', type=str, default="gpu", help='accelerator to use')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    params = parser.parse_args()

    name_prefix = (params.structure + '_') if params.structure else ''
    config_name = os.path.basename(params.config)
    config_name = re.sub(r'.ya?ml$', '', config_name)
    config = {'name': name_prefix + config_name, **yaml.load(params.config, structure=params.structure)}

    train(config, params.fast_dev_run, params.accelerator, params.checkpoint)
