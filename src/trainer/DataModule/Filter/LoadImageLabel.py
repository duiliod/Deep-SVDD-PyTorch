import os

import numpy as np
from PIL import Image

from .Filter import Filter

import pickle

class LoadImageLabel(Filter):
    '''
    Dataset object for image datasets

    Loads a label (0=control, 1=anomaly) for a given filename from a pickle
    file. The pickle path is resolved as follows, in order of priority:
      1. The explicit `labels_path` config parameter, if provided.
      2. A path derived from `input_path` (the dataset's data directory,
         coming from the split's `path:` field): assumes the layout
         `.../data/{structure}/hippocampi_images` -> `.../data/labels/<labels_file>`,
         i.e. two levels up from input_path, then into `labels/`.

    This makes the filter generic across brain structures: no hardcoded
    machine- or structure-specific path is baked into the source code.
    '''

    def __init__(self, input_path, subfolder='', labels_path=None, labels_file='all_labels_anomaly_detection.pkl', **args):
        '''
        Constructor
        '''
        super().__init__(**args)
        self._input_path = input_path
        self._subfolder = subfolder

        if labels_path is not None:
            self._labels_path = labels_path
        else:
            # input_path is typically .../data/{structure}/hippocampi_images
            # go two levels up to .../data, then into labels/<labels_file>
            data_root = os.path.dirname(os.path.dirname(input_path))
            self._labels_path = os.path.join(data_root, 'labels', labels_file)

        # lazily loaded once per filter instance (previously this pickle was
        # re-read from disk on every single apply_one() call, which was both
        # slow and unnecessary since labels do not change during training)
        self._labels = None

    def _get_labels(self):
        if self._labels is None:
            with open(self._labels_path, 'rb') as fp:
                self._labels = pickle.load(fp)
        return self._labels

    def apply_one(self,  filename, i, prepared):
        '''
        Read a label
        '''
        label = self._get_labels().get(filename)
        return label

