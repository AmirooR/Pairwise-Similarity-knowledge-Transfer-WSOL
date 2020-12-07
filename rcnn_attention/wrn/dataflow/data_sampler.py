import numpy as np
from rcnn_attention.wrn.dataflow.datasets.cifar import load_cifar10_data
from rcnn_attention.wrn.dataflow.datasets.miniimagenet_v2 import load_miniimagenet_data
from rcnn_attention.dataflow.data_reader import RandomIterator
from rcnn_attention.wrn.dataflow.datasets.omniglot import load_omniglot_data
from rcnn_attention.wrn.dataflow.datasets.mnist import load_mnist_data


class DataSamplerWRN:
  def __init__(self,
               rng_fn,
               is_training=True,
               k_shot=4,
               bag_size=10,
               data_format='channels_last',
               has_bag_image_iterator=True,
               dataset_name='cifar10',
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
    if num_sample_classes_min is None and num_sample_classes > 0:
      num_sample_classes_min = bag_size
    self.one_example_per_class = one_example_per_class
    self.num_sample_classes_min = num_sample_classes_min
    self.is_training = is_training
    self.k_shot = k_shot
    self.bag_size = bag_size
    self.data_format = data_format
    self.has_bag_image_iterator = has_bag_image_iterator
    self.dataset_name = dataset_name
    self.has_single_target = has_single_target
    self.num_negative_bags = num_negative_bags
    self.num_sample_classes = num_sample_classes
    self.dataset_name = dataset_name
    self.original_imgs = None
    self.total_bags = k_shot + num_negative_bags

    if dataset_name == 'cifar10':
      data = load_cifar10_data()
    elif dataset_name == 'miniimagenet':
      data = load_miniimagenet_data(split=split,train_num=train_num, use_features=use_features, add_images=add_images)
    elif dataset_name == 'omniglot':
      data_ = load_omniglot_data(split=split, prefix=omniglot_prefix, use_features=use_features)
      if add_images:
        raise ValueError('Adding images is not implemented for omniglot')
      data = [data_[0], data_[0], data_[1], data_[1]] #just copy, there is no train/test split 
    elif dataset_name == 'mnist':
      data_ = load_mnist_data(split=split, use_features=use_features)
      if add_images:
        raise ValueError('Adding images is not implemented for mnist')
      data = [data_[0], data_[0], data_[1], data_[1]] #same as omniglot, not trian/test for train classes
    else:
      raise ValueError('dataset_name {} is not implemented'.format(dataset_name))
    if is_training:
      self.x = data[0]
      self.y = data[2]
      if add_images:
        self.original_imgs = data[-2] #original_img_train
    else:
      self.x = data[1]
      self.y = data[3]
      if add_images:
        self.original_imgs = data[-1] #original_img_test
    if data_format == 'channels_last' and dataset_name == 'cifar10':
      self.x = np.moveaxis(self.x, 1, 3)
    self.nr_images = self.x.shape[0]
    if dataset_name == 'cifar10':
      self.nr_classes = 10
    elif dataset_name == 'miniimagenet':
      if split == 'train':
        self.nr_classes = 64
      elif split == 'test':
        self.nr_classes = 20
      elif split == 'val':
        self.nr_classes = 16
      else:
        raise ValueError('split {} is not recognized for miniimagenet'.format(split))
    elif dataset_name == 'omniglot':
      self.nr_classes = self.y.max()+1
    elif dataset_name == 'mnist':
      self.nr_classes = 10
    else:
      raise ValueError('dataset_name {} nr_classes is not defined.'.format(dataset_name))
    self.class_indices = [[] for _ in range(self.nr_classes)]
    self._fill_class_indices()
    self.class_iterators = [RandomIterator(len(x), rng_fn) for x in self.class_indices]
    self.image_iterator = RandomIterator(self.nr_images, rng_fn)
    self.bag_image_iterator = RandomIterator(self.nr_images, rng_fn) #It might be repeated images
    if self.bag_size > 1:
      self.bag_iterator = RandomIterator(self.bag_size, rng_fn)

    if self.num_sample_classes > 1:
      assert self.num_sample_classes >= self.num_sample_classes_min, 'num_sample_classes is not enough'
      self.sample_class_iterator = RandomIterator(self.nr_classes, rng_fn)
      self.sample_classes_range = range(self.num_sample_classes_min, self.num_sample_classes+1)
      self.sample_classes_range_iterator = RandomIterator(len(self.sample_classes_range), rng_fn)

  def _fill_class_indices(self):
    for i, class_idx in enumerate(self.y):
      self.class_indices[class_idx].append(i)

  @property
  def _next_class(self):
    indices = self.image_iterator.getNextIndices(num = 1)
    class_idx = self.y[indices[0]]
    return class_idx

  def _next_class_data(self, class_idx):
    current_class_index = self.class_indices[class_idx]
    idx = self.class_iterators[class_idx].getNextIndices(num = 1)
    current_idx = current_class_index[idx[0]]
    image = self.x[current_idx]
    label = self.y[current_idx]
    original_image = None
    if self.original_imgs is not None:
      original_image = self.original_imgs[current_idx]
    return image, label, original_image

  def sample_bag(self, img_iterator, size, class_idx, is_positive=True, sampled_classes=None):
    imgs = []
    labels = []
    original_images = None
    if self.original_imgs is not None:
      original_images = []
    while len(imgs) < size:
      index = img_iterator.getNextIndices(num=1)[0]
      label = self.y[index]
      if sampled_classes is not None and not label in sampled_classes:
        continue
      if label == class_idx:
        if not is_positive or self.has_single_target:
          continue
      if self.one_example_per_class and label in labels:
        continue
      labels.append(label)
      imgs.append(self.x[index])
      if self.original_imgs is not None:
        original_images.append(self.original_imgs[index])
    return imgs, labels, original_images

  def sample_other_classes(self, class_idx):
    sampled_classes = [] #list(set(self.sample_class_iterator.getNextIndices(self.num_sample_classes-1)))
    next_range_index = self.sample_classes_range_iterator.getNextIndices(1)[0]
    cur_num_sample_classes = self.sample_classes_range[next_range_index]

    while len(sampled_classes) != cur_num_sample_classes - 1:
      #THIS IS DUE TO AN ERORR (same class can come in sampled_class twice)
      other_class = self.sample_class_iterator.getNextIndices(1)[0]
      if other_class not in sampled_classes:
        sampled_classes.append(other_class)
    if class_idx in sampled_classes:
      #class_idx is there. 
      #so, we should sample another class which is not there
      other_class = self.sample_class_iterator.getNextIndices(1)[0]
      while other_class in sampled_classes:
        other_class = self.sample_class_iterator.getNextIndices(1)[0]
      sampled_classes.append(other_class)
    else:
      #class_idx is not there.
      #So, we should add it
      sampled_classes.append(class_idx)
    assert len(sampled_classes) == cur_num_sample_classes,'sampled_classes does not have correct len'
    assert class_idx in sampled_classes, 'class_idx should be in sampled_classes'
    return sampled_classes


  def next(self):
    """
    define:
      total_bags: k_shot + num_negative_bags
      positive_bags: k_shot
      num_images: total_bags*bag_size
    Returns:
      imgs: list of num_images numpy arrays.
        if use_features is True it has the shape (1,1, D) where D is
        the feature dimension
      labels: list of num_images integer labels where 1s correspond the common object
        and 0s correspond to non-common objects
      original_labels: list of num_images integers corresponding to each image label
        starting from zero
      original_imgs:
        - if use_features and add_images is True represents a list of
        num_images numpy arrays of shape (W,H,C) and type uint8 representing the
        original images.
        - if add_images is False, it will be an empty list
      class_idx: id of the common class (starts from 0)
    """
    #print('next called')
    class_idx = self._next_class
    sampled_classes = None
    if self.num_sample_classes > 1:
      sampled_classes = self.sample_other_classes(class_idx)
    #print('Sampled classes: {}'.format(sampled_classes))
    imgs = []
    labels = []
    original_imgs = []
    for i in range(self.total_bags):
      img, label, original_img = self._next_class_data(class_idx)
      if self.bag_size > 1:
        bag_imgs = [img]
        bag_labels = [label]
        bag_original_imgs = [original_img]
        img_iterator = self.bag_image_iterator if self.has_bag_image_iterator else self.image_iterator
        if i < self.k_shot: #positive part
          (sampled_imgs, sampled_labels,
              sampled_original_imgs) = self.sample_bag( img_iterator,
                  self.bag_size - 1, class_idx,
                  is_positive=True, sampled_classes=sampled_classes)
          bag_imgs.extend(sampled_imgs)
          bag_labels.extend(sampled_labels)
          if self.original_imgs is not None:
            bag_original_imgs.extend(sampled_original_imgs)
        else: #negative part
          (bag_imgs, bag_labels,
              bag_original_imgs) = self.sample_bag( img_iterator, self.bag_size,
                class_idx, is_positive=False, sampled_classes=sampled_classes)

        indices = self.bag_iterator.getNextIndices(num=self.bag_size)
        imgs.extend([bag_imgs[j] for j in indices])
        labels.extend([bag_labels[j] for j in indices])
        if self.original_imgs is not None:
          original_imgs.extend([bag_original_imgs[j] for j in indices])
      else:
        if i < self.k_shot: #positive part
          imgs.append(img)
          labels.append(label)
          if self.original_imgs is not None:
            original_imgs.append(original_img)
        else: #negative part
          img_iterator = self.bag_image_iterator if self.has_bag_image_iterator else self.image_iterator
          (neg_img, neg_label,
              neg_original_img) = self.sample_bag(img_iterator, 1, class_idx,
                  is_positive=False, sampled_classes=sampled_classes)
          imgs.append(neg_img)
          labels.append(neg_label)
          if self.original_imgs is not None:
            original_imgs.append(neg_original_img)
    original_labels = [l for l in labels]
    labels = [int(l==class_idx) for l in labels]
    return imgs, labels, original_labels, original_imgs, class_idx

