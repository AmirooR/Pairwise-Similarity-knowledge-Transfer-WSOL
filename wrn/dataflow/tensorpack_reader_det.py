from rcnn_attention.wrn.dataflow.data_sampler_det import DataSamplerDet
import os.path as osp
import numpy as np
import cv2
from tensorpack.dataflow import RNGDataFlow

class TensorpackReaderDet(RNGDataFlow):
  def __init__(self,
               root,
               split,
               k_shot=1,
               shuffle=False,
               noise_rate=0.0,
               class_agnostic=False,
               num_negative_bags=0,
               min_nclasses_in_positive_images=0,
               smart_neg_bag_sampler=False,
               pos_threshold=0.5,
               output_im_shape=None,
               objectness_key='objectness',
               mode=None,
               folds=None,
               feature_normalization=None,
               bag_size=None,
               feas_key=None):

    rng = np.random.RandomState(1357)
    def _rng_fn():
      return rng
    self.root = root
    self.k_shot = k_shot
    self._shuffle = shuffle
    self._is_shuffled = False
    self.split = split
    self.class_agnostic = class_agnostic
    self.noise_rate = noise_rate
    self.num_negative_bags = num_negative_bags
    self.min_nclasses_in_positive_images = min_nclasses_in_positive_images
    self.pos_threshold = pos_threshold
    self.output_im_shape = tuple(output_im_shape)
    self.objectness_key = objectness_key
    self.mode = mode
    self.folds = folds
    self.smart_neg_bag_sampler = smart_neg_bag_sampler
    assert(feature_normalization in [None, '', 'l2'])
    self.feature_normalization = feature_normalization
    self.bag_size = bag_size
    self.feas_key = feas_key

    self.init_data_sampler(_rng_fn)

  def init_data_sampler(self, rng_fn):
     self.sampler = DataSamplerDet(self.root, self.split,
                                  rng_fn, self.k_shot,
                                  self.num_negative_bags,
                                  self.noise_rate,
                                  self.class_agnostic,
                                  min_nclasses_in_positive_images=self.min_nclasses_in_positive_images,
                                  smart_neg_bag_sampler=self.smart_neg_bag_sampler,
                                  pos_threshold=self.pos_threshold,
                                  objectness_key=self.objectness_key,
                                  mode=self.mode,
                                  folds=self.folds,
                                  feas_key=self.feas_key)

  def size(self):
    return self.sampler.cls_reader.nr_images

  def _load_feas(self, fns):
    feas = []
    for fn in fns:
      fea_path = osp.join(self.root, 'Feas', fn)
      fea = np.load(fea_path)
      if isinstance(fea, np.lib.npyio.NpzFile):
        assert 'fea' in fea, 'fea is not in npz file'
        fea = fea['fea']
      if self.bag_size is not None:
        fea = fea[:self.bag_size]
      feas.append(fea)
    return np.concatenate(feas, axis=-1)

  def _load_image_and_feas(self, data_fns):
    imgs, feas, img_fns = [], [], []
    for fn in data_fns:
      img_path = osp.join(self.root, 'Images', fn[0])
      img_fns.append(fn[0])
      img = cv2.imread(img_path, cv2.IMREAD_COLOR)
      if self.output_im_shape is not None and img.shape[:2] != self.output_im_shape:
        img = cv2.resize(img, self.output_im_shape[::-1])
      imgs.append(img)
      fea = self._load_feas(fn[1:])
      #print(fea.shape)
      feas.append(fea)
    return imgs, feas, img_fns

  def _rel_to_abs(self, b, image_shape):
      assert(isinstance(b, np.ndarray))
      assert(np.all(np.logical_and(0 <= b, b <= 1)))
      b = np.array(b)
      b[...,0::2] *= image_shape[0]
      b[...,1::2] *= image_shape[1]
      return b

  def format_boxes(self, boxes, image_shape):
    xy_format, coordinates_format = self.sampler.meta.split(',')
    assert(xy_format == 'y0x0y1x1')
    assert(coordinates_format  == 'relative'), 'Only supports relative bb format.'

    #if coordinates_format == 'relative':
    #  if isinstance(boxes, list):
    #    boxes = [self._rel_to_abs(b, image_shape) for b in boxes]
    #  else:
    #    boxes = self._rel_to_abs(boxes, image_shape)
    return boxes

  def get_data(self):
    if not self._is_shuffled and self._shuffle:
      def _rng_fn():
        return self.rng
      self.init_data_sampler(_rng_fn)

    for _ in list(xrange(self.size())):
      (data_fns, boxes, classes, fea_boxes,
          fea_classes, objectness, target_class) = self.sampler.next()
      imgs, feas, img_fns = self._load_image_and_feas(data_fns)
      boxes = self.format_boxes(boxes, imgs[0].shape[:2])
      fea_boxes = self.format_boxes(np.stack(fea_boxes), imgs[0].shape[:2])
      fea_target_classes = []
      for fea_class in fea_classes:
        fea_target_class = np.zeros_like(fea_class)
        fea_target_class[fea_class == target_class] = 1
        fea_target_classes.append(fea_target_class)

      feas = np.stack(feas)
      if self.feature_normalization == 'l2':
        feas = feas / (1e-12 + np.linalg.norm(feas, axis=-1)[..., np.newaxis])

      ret = [feas, fea_boxes, np.stack(fea_target_classes),
             np.stack(fea_classes), np.stack(imgs), np.stack(objectness), target_class,
             img_fns]+boxes+classes
      if self.bag_size is not None:
        def get_first_bag_size_elements(arr_list, indices):
          for ind in indices:
            arr_list[ind] = arr_list[ind][:,:self.bag_size]
          return arr_list
        ret = get_first_bag_size_elements(ret, [0,1,2,3,5])
      yield ret
