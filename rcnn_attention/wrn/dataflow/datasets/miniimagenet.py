import functools
import logging
import os
import tarfile

import numpy as np
from six.moves import urllib

from dl_papers.common.data import BatchIterator as _BatchIterator
from dl_papers.env import DATA_DIR

from collections import defaultdict
import os
import sys
import cPickle as pickle

# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

IMAGE_SIZE = 84
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
IMAGE_SHAPE_CHANNELS_FIRST = (3, IMAGE_SIZE, IMAGE_SIZE)

MINIIMAGENET_DATA_PATH = '/home/amir/Work/codes/co-detection/CoLoc/rcnn_attention/wrn/data'
LABELS_AND_IMAGES_PKL = 'labels_and_images.pkl'
TRANSLATE_PIXELS = 4

def create_info_pkl(data_path, split, use_features, save_path):
  data_dir = os.path.join(data_path, split)
  assert os.path.exists(data_dir), "data_dir {} does not exist".format(data_dir)
  logger.info("loading MiniImageNet data")
  label_names = os.listdir(data_dir)
  labels_and_images = {ln: os.listdir(os.path.join(data_dir, ln)) for ln in label_names}
  loaded_files  = defaultdict(lambda: [])
  def load_all_images():
    def _load_class(c):
      all_images = list(labels_and_images[c])
      for img_name in all_images:
        if use_features:
          np_file  = os.path.join('features', split, c, img_name[:-3]+'npy')
          #fea = np.load(np_file)
          #loaded_images[c].append(fea)
          loaded_files[c].append(np_file)
        else:
          img_file = os.path.join(split, c,img_name)
          #img = imread(img_file, mode='RGB')
          #img = imresize(img, size=IMAGE_SHAPE)
          #loaded_images[c].append(img)
          loaded_files[c].append(img_file)
    for cls in labels_and_images:
      _load_class(cls)

  def check_loaded_images(n_min):
    return loaded_files  and all([len(v) >= n_min for v in loaded_files.values()])

  load_all_images()
  check = check_loaded_images(600)
  assert check == True, 'loaded images are less than 600!'

  classes = loaded_files.keys()
  all_files = []
  all_labels = []
  cls2files = {}
  for i, cls in enumerate(classes):
    all_files.extend([file_name for file_name in loaded_files[cls]])
    all_labels.extend([i for _ in loaded_files[cls]])
    cls2files[i] = loaded_files[cls]
  save_dict = dict(files=all_files, labels=all_labels, cls2files=cls2files, data_root=data_path)
  with open(save_path, 'wb') as f:
    pickle.dump(save_dict, f)


def _load_data(data_path, split, train_num=500, use_features=False):
    data_dir = os.path.join(data_path, split)
    #if use_features:
    #  data_dir = os.path.join(data_path, 'features', split)
    assert os.path.exists(data_dir), "data_dir {} does not exist".format(data_dir)

    logger.info("loading MiniImageNet data")

    #label_names = os.listdir(data_dir)
    #NOTE, TODO: it should change to sorted later
    #labels_and_images = {ln: os.listdir(os.path.join(data_dir, ln)) for ln in label_names}
    l2i_pkl = os.path.join(data_path, LABELS_AND_IMAGES_PKL)
    with open(l2i_pkl, 'rb') as f:
      data = pickle.load(f)
    labels_and_images = data[split]
    loaded_images = defaultdict(lambda: [])
    loaded_files  = defaultdict(lambda: [])

    def load_all_images():
      from scipy.ndimage import imread
      from scipy.misc import imresize
      def _load_class(c):
        all_images = list(labels_and_images[c])
        for img_name in all_images:
          if use_features:
            np_file  = os.path.join(data_path,'features', split, c, img_name[:-3]+'npy')
            fea = np.load(np_file)
            loaded_images[c].append(fea)
            loaded_files[c].append(np_file)
          else:
            img_file = os.path.join(data_path, split, c,img_name)
            img = imread(img_file, mode='RGB')
            img = imresize(img, size=IMAGE_SHAPE)
            loaded_images[c].append(img)
            loaded_files[c].append(img_file)
      for cls in labels_and_images:
        _load_class(cls)

    def check_loaded_images(n_min):
      return loaded_images  and all([len(v) >= n_min for v in loaded_images.values()])

    load_all_images()
    check = check_loaded_images(600)
    assert check == True, 'loaded images are less than 600!'
    test_num = 600 - train_num
    x_train,x_test,y_train,y_test = None,None,None,None
    files_train = []
    files_test  = []
    classes = loaded_images.keys()
    if train_num > 0:
      x_train = np.vstack([np.stack(loaded_images[k][:train_num]) for k in classes])
      y_train = np.hstack([[i for _ in range(train_num)] for i in range(len(classes))])
      for k in classes:
        files_train.extend(loaded_files[k][:train_num])
    if test_num > 0:
      y_test  = np.hstack([[i for _ in range(600-train_num)] for i in range(len(classes))])
      x_test  = np.vstack([np.stack(loaded_images[k][train_num:]) for k in classes])
      for k in classes:
        files_test.extend(loaded_files[k][train_num:])
    x_train_mean, x_train_std = None, None
    if not use_features:
      # This is the standard ResNet mean/std image normalization.
      #x_train_mean = np.mean(
      #    x_train, axis=(0, 1, 2), keepdims=True, dtype=np.float64,
      #).astype(np.float32)
      #x_train_std = np.std(
      #    x_train, axis=(0, 1, 2), keepdims=True, dtype=np.float64,
      #).astype(np.float32)
      x_train_mean = np.array([[[[123.05668 , 117.817535, 105.03199 ]]]], dtype=np.float32) #mean of first 500 images of all classes (uncomment below if want the mean,std for 600 images)
      x_train_std  = np.array([[[[68.69664 , 66.528915, 70.08921 ]]]], dtype=np.float32) #std of first 500 images of all classes
      #x_train_mean = np.array([[[[123.1 , 117.82, 105.09 ]]]], dtype=np.float32) #mean of all images
      #x_train_std  = np.array([[[[68.73 , 66.54, 70.083 ]]]], dtype=np.float32) #std of all images

      if train_num > 0:
        x_train = (x_train - x_train_mean) / x_train_std
      if test_num > 0:
        x_test = (x_test - x_train_mean) / x_train_std

    logger.info("loaded Mini-ImageNet data")
    if train_num > 0:
      logger.info("training set size: {}".format(len(x_train)))
    if test_num > 0:
      logger.info("test set size: {}".format(len(x_test)))

    return x_train, x_test, y_train, y_test, x_train_mean, x_train_std, files_train, files_test

# -----------------------------------------------------------------------------
load_miniimagenet_data = functools.partial(
    _load_data,
    data_path=MINIIMAGENET_DATA_PATH,
)
# -----------------------------------------------------------------------------

class BatchIterator(_BatchIterator):
    def __init__(self, batch_size, data_format='channels_last', **kwargs):
        super(BatchIterator, self).__init__(batch_size, **kwargs)

        self.data_format = data_format

    def transform(self, x_batch, y_batch, training=False):
        if training:
            x_batch = self.augment_x_batch(x_batch)
        if self.data_format == 'channels_last':
            x_batch = np.moveaxis(x_batch, 1, 3)

        return x_batch, y_batch

    def augment_x_batch(self, x_batch):
        # Zero-padding here follows the DenseNets paper rather than the Wide
        # ResNets paper.
        x_batch_padded = np.pad(
            x_batch,
            (
                (0, 0),
                (0, 0),
                (TRANSLATE_PIXELS, TRANSLATE_PIXELS),
                (TRANSLATE_PIXELS, TRANSLATE_PIXELS),
            ),
            'constant',
        )

        batch_size = len(x_batch)
        x_batch_t = np.empty_like(x_batch)

        translations = self.random.randint(
            2 * TRANSLATE_PIXELS + 1, size=(batch_size, 2),
        )
        flips = self.random.rand(batch_size) > 0.5

        for i in range(batch_size):
            translation_y, translation_x = translations[i]
            x_batch_t[i] = x_batch_padded[
                i,
                :,
                translation_y:translation_y + IMAGE_SIZE,
                translation_x:translation_x + IMAGE_SIZE,
            ]

            if flips[i]:
                x_batch_t[i] = x_batch_t[i, :, :, ::-1]

        return x_batch_t

if __name__ == '__main__':
  create_info_pkl(data_path=MINIIMAGENET_DATA_PATH, split='train', use_features=False,save_path='/home/amir/data/mini-imagenet/mine/train.pkl')
  create_info_pkl(data_path=MINIIMAGENET_DATA_PATH, split='val', use_features=False,save_path='/home/amir/data/mini-imagenet/mine/val.pkl')
  create_info_pkl(data_path=MINIIMAGENET_DATA_PATH, split='test', use_features=False,save_path='/home/amir/data/mini-imagenet/mine/test.pkl')
  create_info_pkl(data_path=MINIIMAGENET_DATA_PATH, split='val', use_features=True,save_path='/home/amir/data/mini-imagenet/mine/val_fea.pkl')
  create_info_pkl(data_path=MINIIMAGENET_DATA_PATH, split='test', use_features=True,save_path='/home/amir/data/mini-imagenet/mine/test_fea.pkl')
