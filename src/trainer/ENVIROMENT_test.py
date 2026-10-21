import numpy as np
import numpy as np
import pandas as pd 
#import pydicom as dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel as nb
from skimage import measure
import re
from argparse import ArgumentParser

import torch
import torch.nn as nn
from pytorch_lightning import seed_everything
import pickle
from lightning import getModules
from logging import logMultiprocessing
from pytorch_lightning import Trainer
import DataModule
import Models
from Training import TrainFactorySVDD
from lightning import getModulesSVDD

print('chupa')
