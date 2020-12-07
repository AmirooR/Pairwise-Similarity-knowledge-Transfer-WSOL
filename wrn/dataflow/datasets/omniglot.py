import functools
import logging
import os
import tarfile

import numpy as np
from six.moves import urllib

from collections import defaultdict
import os
import sys
import cPickle as pickle
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

IMAGE_SIZE = 28
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 1)
IMAGE_SHAPE_CHANNELS_FIRST = (1, IMAGE_SIZE, IMAGE_SIZE)

#OMNIGLOT_DATA_PATH = '/home/amir/data/omniglot_resized'
SAVED_PICKLE = 'data/omniglot/mine'
#'/home/amir/data/omniglot/mine'

def create_info_pkl(data_path, save_path, seed=1357):
  rotations=[0]
  alphabets = os.listdir(data_path)
  labels_and_images = {}
  for alphabet in alphabets:
    base_folder = os.path.join(data_path, alphabet)
    label_names = os.listdir(base_folder)
    labels_and_images.update({alphabet+os.path.sep+ln: os.listdir(os.path.join(base_folder,ln))
                              for ln in label_names})
  _rand = np.random.RandomState(seed)
  all_clss = labels_and_images.keys()
  _rand.shuffle(all_clss)
  n_splits = (0, 1100, 1200, len(all_clss))
  splits = ('train','val','test')

  for start, end, split in zip(n_splits, n_splits[1:], splits):
    print('saving {} data, len is {}'.format(split,end-start))
    classes = {k: labels_and_images[k] for k in all_clss[start:end]}
    flat_data = []
    flat_targets = []
    name2label = {}
    cls2files = defaultdict(lambda: [])
    def add_label(name):
      if not name2label.has_key(name):
        name2label[name] = len(name2label)
      return name2label[name]
    for c in classes:
      all_images = list(classes[c])
      for img_name in all_images:
        for rot in rotations:
          flat_data.append(os.path.join(c,img_name))
          flat_targets.append(add_label(c+os.path.sep+'rot_'+str(rot)))
          cls2files[flat_targets[-1]].append(flat_data[-1])
    data = dict(files=flat_data, labels=flat_targets, cls2files=dict(cls2files), data_root=data_path)
    with open(os.path.join(save_path, split+'.pkl'), 'wb') as f:
      pickle.dump(data, f)

def save_omniglot(data_path, seed=1357, rotations=[0,90,180,270], prefix=''):
  alphabets = os.listdir(data_path)
  labels_and_images = {}
  for alphabet in alphabets:
    base_folder = os.path.join(data_path, alphabet)
    label_names = os.listdir(base_folder)
    labels_and_images.update({alphabet+os.path.sep+ln: os.listdir(os.path.join(base_folder,ln))
                              for ln in label_names})
  _rand = np.random.RandomState(seed)
  all_clss = labels_and_images.keys()
  _rand.shuffle(all_clss)
  n_splits = (0, 1100, 1300, len(all_clss))
  splits = ('train','val','test')
  loaded_images = defaultdict(lambda: [])
  from scipy.ndimage import imread
  from scipy.ndimage.interpolation import rotate

  for start, end, split in zip(n_splits, n_splits[1:], splits):
    print('saving {} data, len is {}'.format(split,end-start))
    classes = {k: labels_and_images[k] for k in all_clss[start:end]}
    flat_data = []
    flat_targets = []
    data = {}
    name2label = {}
    def add_label(name):
      if not name2label.has_key(name):
        name2label[name] = len(name2label)
      return name2label[name]
    for c in classes:
      all_images = list(classes[c])
      for img_name in all_images:
        img = imread(os.path.join(data_path, os.path.join(c, img_name)))
        img = 1. - np.reshape(img, (28,28,1)) / 255.
        for rot in rotations:
          img_rot = rotate(img, rot, reshape=False)
          flat_data.append(img_rot)
          flat_targets.append(add_label(c+os.path.sep+'rot_'+str(rot)))
    indices = np.arange(len(flat_data))
    _rand.shuffle(indices)
    data['X'] = np.stack(flat_data)[indices]
    data['Y'] = np.stack(flat_targets)[indices]
    with open(SAVED_PICKLE+prefix+'_'+split+'.pkl', 'wb') as f:
      pickle.dump(data, f)

def _load_omniglot_data(data_path,split,prefix='', use_features=False):
    pkl_file = data_path+prefix+'_'+split+'.pkl'
    if use_features:
      pkl_file = data_path+prefix+'_'+split+'_fea.pkl'
    assert os.path.exists(pkl_file), "pkl_file {} does not exist".format(pkl_file)

    logger.info("loading OMNIGLOT data")
    with open(pkl_file, 'rb') as f:
      data = pickle.load(f)

    x = data['X']
    y = data['Y']
    return x, y

# -----------------------------------------------------------------------------
load_omniglot_data = functools.partial(
    _load_omniglot_data,
    data_path=SAVED_PICKLE,
    use_features=False
)
# -----------------------------------------------------------------------------


if __name__ == '__main__':
  #save_omniglot(data_path=OMNIGLOT_DATA_PATH,rotations=[0], prefix='_no_rot')
  create_info_pkl(data_path=OMNIGLOT_DATA_PATH, seed=1357, save_path=SAVED_PICKLE)
