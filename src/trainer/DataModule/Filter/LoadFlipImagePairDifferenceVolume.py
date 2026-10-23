import os

import numpy as np
from PIL import Image
import torch

from .Filter import Filter

import nibabel as nb

class LoadFlipImagePairDifferenceVolume(Filter):
    '''
    Dataset object for image datasets
    '''
    
    def __init__(self, input_path, subfolder='', canonical_order=True, **args):
        '''
        Constructor
        '''
        super().__init__(**args)
        self._input_path = input_path
        self._subfolder = subfolder
        # With 'sub' the network output is antisymmetric: f(a, b) = -f(b, a). A 'left' entry
        # used to give the channels (flipped left, right) and a 'right' entry (right, flipped
        # left), so training mixed both signs (mean of the outputs ~ 0, unstable center c)
        # while testing only uses 'right' entries. canonical_order=True gives every entry
        # the order (right, flipped left), the one used at test time.
        self._canonical_order = canonical_order

    def apply_one(self,  filename, i, prepared):
        '''
        Read an image
        '''
        # load the image using nibabel
        # print(self._input_path)
        # print('filename',filename)
        image = nb.load(os.path.join(self._input_path, self._subfolder, filename + '.nii'))
        image = np.array(image.get_fdata())

        if 'left' in filename:
            image = np.flip(image,axis=0).copy()

            new_filename = filename.replace("left", "right" )
            imagepair = nb.load(os.path.join(self._input_path, self._subfolder, new_filename + '.nii'))
            imagepair = np.array(imagepair.get_fdata())

            left_volume, right_volume = np.count_nonzero(image), np.count_nonzero(imagepair)
            if self._canonical_order:
                image, imagepair = imagepair, image   # (right, flipped left), like a 'right' entry


        else:
            new_filename = filename.replace("right","left" )
            imagepair = nb.load(os.path.join(self._input_path, self._subfolder, new_filename + '.nii'))
            imagepair = np.array(imagepair.get_fdata())
            imagepair = np.flip(imagepair,axis=0).copy()

            left_volume, right_volume = np.count_nonzero(imagepair), np.count_nonzero(image)
        # print(filename,new_filename)

        
        # difference_volume = np.absolute(np.count_nonzero(image) - np.count_nonzero(imagepair))

        # print('loadflipimagepair')
        image = torch.tensor(image)
        imagepair = torch.tensor(imagepair)

        image = torch.cat((torch.unsqueeze(image, 0),torch.unsqueeze(imagepair, 0)), 0)

        return [image,left_volume, right_volume]

