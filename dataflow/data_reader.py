import cPickle as pickle
import os.path as osp
from easydict import EasyDict
import numpy as np

class RandomIterator:
  """
  iterates over [0..N) and returns a number of random indices
  Args:
    N: number of indices to iterate on
  """
  def __init__(self, N, rng_fn):
    assert N>=0, 'N should be greater than zero'
    self.indices = range(N)
    self.current_idx = 0
    self.N = N
    self._rng_fn = rng_fn
    self.epoch = -1
    self._shuffle()

  def _shuffle(self):
    """
    Does the shuffling operation. might be changed!
    """
    rng = self._rng_fn()
    rng.shuffle(self.indices)
    self.epoch += 1

  def getNextIndices(self, num=1):
    """
    returns `num` random indices
    Note: assumes num <= N
    """
    assert num > 0, 'requested number (%d) should be greater than zero' % num
    assert num <= self.N, 'too many items requested N is %d, num requested is %d' % (self.N, num)
    ret_indices = []
    if self.current_idx + num < self.N:
      ret_indices += self.indices[self.current_idx:self.current_idx+num]
      self.current_idx += num
    else:
      ret_indices += self.indices[self.current_idx:]
      self._shuffle()
      num_remained = self.current_idx + num - self.N
      ret_indices += self.indices[:num_remained]
      self.current_idx = num_remained
    return ret_indices

class DataReader:
  """
  reads a database
  """
  def __init__(self, root_path, rng_fn, split='train', ext='.pkl',
               roi_filter_fn=None, folds=None, feas_key=None):
    """
    Args:
      root_path: address of the ImageSet folder where the pkl files are stored
    """
    dataset_path = osp.join( root_path,'ImageSet', split + ext)
    with open( dataset_path , 'rb') as f:
      self.data = EasyDict(pickle.load(f))

    ## We merge the fea_fns and image_fns into a tuple
    if feas_key is not None:
      feas = [self.data.images]
      for fkey in feas_key:
        if fkey in self.data:
          feas.append(self.data[fkey])
        self.data.pop(fkey)
      self.data.images = list(zip(*feas))

    if 'feas' in self.data:
      self.data.pop('feas')
    if 'det_feas' in self.data:
      self.data.pop('det_feas')

    self.filter_by_folds(folds)
    self._check_data(roi_filter_fn)
    self.nr_images = len(self.data.images)

    # number of classes including background class 0
    class_ids = [synset[0] for synset in self.data.synsets]
    self.nr_classes = max(len(self.data.synsets), max(class_ids)+1)
    # class indices stores current index to read for each class
    self.class_indices = [ [] for _ in range(self.nr_classes) ]
    # fill them
    self._fill_class_indices()

    # random iterators for each class/roi
    self.class_iterators = [ RandomIterator(len(x), rng_fn) for x in self.class_indices ]
    # random iterator for each image
    self.image_iterator = RandomIterator(self.nr_images, rng_fn)

  def filter_by_folds(self, folds, keys=['images','rois', 'folds']):
    """filters by folds,
    Note: it should be called after zipping feas with images or
          'feas' should also be passed as keys
    """
    if folds is None:
      return
    if len(folds) == 0:
      return
    print('###### Filtering by folds')
    assert('folds' in self.data), 'folds field is not available'
    assert(all([key in self.data.keys() for key in keys])), 'some keys are not valid'
    assert(np.equal.reduce([len(self.data[key]) for key in keys])), 'keys do not have same len'
    filtered_data = {key:[] if key in keys else self.data[key] for key in self.data.keys()}
    for i in range(len(self.data.folds)):
      if self.data.folds[i] in folds:
        for key in keys:
          filtered_data[key].append(self.data[key][i])

    self.data = EasyDict(filtered_data)
    print('###### Number of filtered images: {}'.format(len(self.data[keys[0]])))

  def get_class_len(self, class_idx):
    return len(self.class_indices[class_idx])

  def _check_data(self, roi_filter_fn=None):
    """
    Checks for the validity of data.

    Note: Somee tests (e.g ensuring the values of the keys are of list type )
    are not yet implemented.
    """
    assert 'synsets' in self.data.keys(), 'dataset does not have "synsets" key'
    assert 'images' in self.data.keys(), 'dataset does not have "images" key'
    assert 'rois' in self.data.keys(), 'dataset does not have "rois" key'
    assert len(self.data.images) == len(self.data.rois), 'rois and images length mismatch'
    if roi_filter_fn is not None:
      filtered_data = EasyDict(synsets=self.data.synsets,
                               images=[],
                               rois=[])
      for image, roi in zip(self.data.images, self.data.rois):
        if roi_filter_fn(roi):
          filtered_data.rois.append(roi)
          filtered_data.images.append(image)
      self.data = filtered_data
      print('###### Number of filtered images:{}'.format(len(self.data.images)))

  def _fill_class_indices(self):
    """
    for each class fills rois ( or images) indices of that class in `self.class_indices`
    Note: there are 3 cases
    """
    for i, roi in enumerate(self.data.rois):
      roi.classes = np.int32(roi.classes)
      # 1) Image with no boxes but image_class
      if len(roi.classes) == 0 and 'image_class' in roi.keys():
        class_idx = roi.image_class
        self.class_indices[ class_idx ].append( i )
      # 2) Image with boxes
      else:
        for class_idx in set(roi.classes):
          self.class_indices[ class_idx ].append( i )

  def get_class_indices(self):
    return self.class_indices

  def next_images_data(self, num):
    """
    returns next `num` random images and rois
    """
    indices = self.image_iterator.getNextIndices(num = num)
    images = [self.data.images[i] for i in indices]
    rois   = [self.data.rois[i] for i in indices]
    return images, rois

  def next_class_data(self, class_idx, num ):
    """
    Selects `num` items from class `class_idx`
    Note: here we assume the num is too much less than the
    items in class_idx so if we reach the last index in the
    array we need to start from the beginning only once.
    """

    assert class_idx < self.nr_classes, "class_idx (%d) should be less than the number of classes (%d)! " % (class_idx, self.nr_classes)
    assert class_idx >= 0, "class_idx (%d) should be greater than or equal to zero!" % class_idx
    assert num > 0, "number of selected data (%d) should be greater than zero!" % num
    assert len(self.class_indices[class_idx]) >= num, """this class (%d) with (%d) items does not
                              have enought items (%d)!""" % (class_idx, len(self.class_indices[class_idx]), num)

    current_class_index = self.class_indices[class_idx]

    indices = self.class_iterators[class_idx].getNextIndices(num = num)
    current_indices = [current_class_index[i] for i in indices]

    images =  [self.data.images[i] for i in current_indices]
    rois   =  [self.data.rois[i] for i in current_indices]
    return images, rois
