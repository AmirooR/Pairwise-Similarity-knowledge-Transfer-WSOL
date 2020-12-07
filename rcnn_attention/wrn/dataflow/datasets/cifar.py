import functools
import logging
import os
import tarfile

import numpy as np
from six.moves import urllib

#from dl_papers.common.data import BatchIterator as _BatchIterator
#from dl_papers.env import DATA_DIR

# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

IMAGE_SIZE = 32
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
IMAGE_SHAPE_CHANNELS_FIRST = (3, IMAGE_SIZE, IMAGE_SIZE)

CIFAR10_DATA_PATH = 'cifar10/cifar-10-batches-py'
CIFAR10_NUM_DATA_FILES = 5

TRANSLATE_PIXELS = 4

# -----------------------------------------------------------------------------


def _get_x(data_batch):
    return np.reshape(data_batch['data'], (-1,) + IMAGE_SHAPE_CHANNELS_FIRST)


def _get_y(data_batch):
    return np.asarray(data_batch['labels'], dtype=np.int32)


# -----------------------------------------------------------------------------


def _download_data(data_url, data_dir_base):
    logger.info("downloading CIFAR data")

    data_bundle_filename = os.path.join(
        data_dir_base, os.path.basename(data_url),
    )

    os.makedirs(data_dir_base)
    urllib.request.urlretrieve(data_url, data_bundle_filename)

    with tarfile.open(data_bundle_filename, 'r:gz') as tar:
        tar.extractall(data_dir_base)

    os.unlink(data_bundle_filename)


def _load_data(data_path, num_data_files, data_url=None):
    data_dir = os.path.join(DATA_DIR, data_path)
    if not os.path.exists(data_dir):
        assert data_url, "must provide data_url to download data"
        _download_data(data_url, os.path.dirname(data_dir))

    logger.info("loading CIFAR data")

    data_train = tuple(
        np.load(os.path.join(
            data_dir, 'data_batch_{}'.format(data_file_index),
        ))
        for data_file_index in range(1, num_data_files + 1)
    )

    # It's not generally correct to use the actual test set here, but our goal
    # is to compare results to existing papers, not to generate novel results.
    data_test = np.load(os.path.join(data_dir, 'test_batch'))

    x_train = np.vstack(_get_x(data_batch) for data_batch in data_train)
    x_test = _get_x(data_test)

    y_train = np.hstack(
        _get_y(data_batch) for data_batch in data_train,
    )
    y_test = _get_y(data_test)

    # This is the standard ResNet mean/std image normalization.
    x_train_mean = np.mean(
        x_train, axis=(0, 2, 3), keepdims=True, dtype=np.float32,
    )
    x_train_std = np.std(
        x_train, axis=(0, 2, 3), keepdims=True, dtype=np.float32,
    )

    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    logger.info("loaded CIFAR data")
    logger.info("training set size: {}".format(len(x_train)))
    logger.info("test set size: {}".format(len(x_test)))

    return x_train, x_test, y_train, y_test


# -----------------------------------------------------------------------------

load_cifar10_data = functools.partial(
    _load_data,
    data_path=CIFAR10_DATA_PATH,
    num_data_files=CIFAR10_NUM_DATA_FILES,
    data_url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
)

# -----------------------------------------------------------------------------


#class BatchIterator(_BatchIterator):
    #def __init__(self, batch_size, data_format='channels_last', **kwargs):
        #super(BatchIterator, self).__init__(batch_size, **kwargs)
#
        #self.data_format = data_format
#
    #def transform(self, x_batch, y_batch, training=False):
        #if training:
            #x_batch = self.augment_x_batch(x_batch)
        #if self.data_format == 'channels_last':
            #x_batch = np.moveaxis(x_batch, 1, 3)
#
        #return x_batch, y_batch
#
    #def augment_x_batch(self, x_batch):
        ## Zero-padding here follows the DenseNets paper rather than the Wide
        ## ResNets paper.
        #x_batch_padded = np.pad(
            #x_batch,
            #(
                #(0, 0),
                #(0, 0),
                #(TRANSLATE_PIXELS, TRANSLATE_PIXELS),
                #(TRANSLATE_PIXELS, TRANSLATE_PIXELS),
            #),
            #'constant',
        #)
#
        #batch_size = len(x_batch)
        #x_batch_t = np.empty_like(x_batch)
#
        #translations = self.random.randint(
            #2 * TRANSLATE_PIXELS + 1, size=(batch_size, 2),
        #)
        #flips = self.random.rand(batch_size) > 0.5
#
        #for i in range(batch_size):
            #translation_y, translation_x = translations[i]
            #x_batch_t[i] = x_batch_padded[
                #i,
                #:,
                #translation_y:translation_y + IMAGE_SIZE,
                #translation_x:translation_x + IMAGE_SIZE,
            #]
#
            #if flips[i]:
                #x_batch_t[i] = x_batch_t[i, :, :, ::-1]
#
        #return x_batch_t
