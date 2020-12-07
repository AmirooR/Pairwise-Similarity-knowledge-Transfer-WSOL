import cPickle as pickle
import gzip
import numpy as np

import functools
import logging
import os, sys

MNIST_DATA_FOLDER = 'data/mnist'
#'/mnt/scratch/amir/detection/CoLoc/rcnn_attention/data_old/detection-data/coloc/mnist'

logger = logging.getLogger(__name__)

def _load_mnist_data(mnist_data_folder, split, use_features=False):
  if not use_features:
    pkl_file = os.path.join(mnist_data_folder, 'mnist.pkl.gz')
    assert os.path.exists(pkl_file), 'pkl file {} does not exist'.format(pkl_file)
    logger.info('loading MNIST data')
    with gzip.open(pkl_file, 'rb') as f:
      train, val, test = pickle.load(f)
    x, y = None, None
    if split == 'train':
      x = train[0] #between 0 and 1, float (N, 728) array 
      y = train[1] #int64 between 0 and 9
    elif split == 'val':
      x = val[0]
      y = val[1]
    elif split == 'test':
      x = test[0]
      y = test[1]
    else:
      raise ValueError('split: {} is not defined for mnist'.format(split))
    logger.info('loaded {} images.'.format(x.shape[0]))
    x = x.reshape((-1,28,28,1))
    return x, y
  else:
    pkl_file = os.path.join(mnist_data_folder, 'mnist_'+split+'_fea.pkl')
    assert os.path.exists(pkl_file), 'pkl_file {} does not exist'.format(pkl_file)
    logger.info('loading MNIST features data')
    with open(pkl_file, 'rb') as f:
      data = pickle.load(f)
    x = data['X']
    y = data['Y']
    return x, y

load_mnist_data = functools.partial(
    _load_mnist_data,
    mnist_data_folder = MNIST_DATA_FOLDER,
)


