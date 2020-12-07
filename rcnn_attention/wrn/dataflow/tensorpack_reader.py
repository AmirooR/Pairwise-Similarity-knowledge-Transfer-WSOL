from data_sampler import DataSamplerWRN
import os.path as osp
import numpy as np
import cv2
from tensorpack.dataflow import RNGDataFlow

#imgs: k_shot*bag_size,32,32,3 if channels_last
#      k_shot*bag_size,3,32,32 if channels_first
#labels: (k_shot*bag_size,)

class TensorpackReaderWRN(RNGDataFlow):
  def __init__(self,
               is_training,
               k_shot=1,
               bag_size=10,
               data_format='channels_last',
               has_bag_image_iterator=True,
               shuffle=False,
               dataset_name='miniimagenet',
               split='train',
               train_num=500,
               use_features=False,
               has_single_target=False,
               num_negative_bags=0,
               num_sample_classes=0,
               omniglot_prefix='',
               num_sample_classes_min=5,
               one_example_per_class=False,
               add_images=True):
    rng = np.random.RandomState(1357)
    def _rng_fn():
      return rng
    self.is_training = is_training
    self.k_shot = k_shot
    self.bag_size = bag_size
    self.data_format = data_format
    self.has_bag_image_iterator = has_bag_image_iterator
    self._shuffle = shuffle
    self._is_shuffled = False
    self.dataset_name = dataset_name
    self.split = split
    self.train_num = train_num
    self.use_features = use_features
    self.has_single_target = has_single_target
    self.num_negative_bags = num_negative_bags
    self.num_sample_classes = num_sample_classes
    self.omniglot_prefix = omniglot_prefix
    self.num_sample_classes_min = num_sample_classes_min
    self.one_example_per_class = one_example_per_class
    self.add_images = add_images
    self.total_bags = k_shot + num_negative_bags
    self.sampler = DataSamplerWRN(_rng_fn,
                                  is_training=is_training,
                                  k_shot=k_shot,
                                  bag_size=bag_size,
                                  data_format=data_format,
                                  has_bag_image_iterator=has_bag_image_iterator,
                                  dataset_name=dataset_name,
                                  split=split,
                                  train_num=train_num,
                                  use_features=use_features,
                                  has_single_target=has_single_target,
                                  num_negative_bags=num_negative_bags,
                                  num_sample_classes=num_sample_classes,
                                  omniglot_prefix=omniglot_prefix,
                                  num_sample_classes_min=num_sample_classes_min,
                                  one_example_per_class=one_example_per_class,
                                  add_images=add_images)

  def size(self):
    return self.sampler.nr_images

  def get_data(self):
    if not self._is_shuffled and self._shuffle:
      self._is_shuffled = True
      def _rng_fn():
        return self.rng
      self.sampler = DataSamplerWRN(_rng_fn,
                                        is_training=self.is_training,
                                        k_shot=self.k_shot,
                                        bag_size=self.bag_size,
                                        data_format=self.data_format,
                                        has_bag_image_iterator=self.has_bag_image_iterator,
                                        dataset_name=self.dataset_name,
                                        split=self.split,
                                        train_num=self.train_num,
                                        use_features=self.use_features,
                                        has_single_target=self.has_single_target,
                                        num_negative_bags=self.num_negative_bags,
                                        num_sample_classes=self.num_sample_classes,
                                        omniglot_prefix=self.omniglot_prefix,
                                        num_sample_classes_min=self.num_sample_classes_min,
                                        one_example_per_class=self.one_example_per_class,
                                        add_images=self.add_images)
    def _r(arr):
      arr = np.stack(arr).astype(np.float32)
      return arr.reshape((self.total_bags, self.bag_size) + arr.shape[1:])

    for _ in list(xrange(self.size())):
      fea_or_imgs, labels, original_labels, original_imgs, class_idx = self.sampler.next()
      if len(original_imgs) == 0:
        original_imgs = [np.zeros((84,84,3), dtype=np.uint8) for _ in labels]
      feas = _r(fea_or_imgs) #k,b,...
      num_bags = feas.shape[0]
      ids = np.arange(self.bag_size)
      sz = original_imgs[0].shape[0]
      y1 = ids*sz+1
      y2 = y1 + sz - 3
      x1 = np.ones_like(ids)
      x2 = np.ones_like(ids)*(sz-3)
      img_box = np.vstack([y1,x1,y2,x2]).T.astype(np.float32) #b,4
      boxes = [img_box for _ in range(num_bags)]
      fea_boxes = np.tile(img_box[None,:], [num_bags, 1, 1]) #k,b,4
      fea_target_classes = _r(labels) #k,b
      fea_classes = _r(original_labels) + 1 #0 is bg. k,b
      imgs = _r(original_imgs) #k,b,h,w,3
      #stack images vertically
      imgs = imgs.reshape((imgs.shape[0],-1,)+imgs.shape[3:]) #k, b*h, w, 3 
      target_class = class_idx + 1 #0 is bg
      classes = [x for x in fea_classes]
      yield [feas, fea_boxes, fea_target_classes,
             fea_classes, imgs, target_class]+boxes+classes



