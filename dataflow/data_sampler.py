from rcnn_attention.dataflow.data_reader import DataReader, RandomIterator
import numpy as np
from itertools import compress

class DataSampler(object):
  def __init__(self,
               classification_root,
               split, rng_fn,
               k_shot=5,
               noise_rate=0.0,
               class_agnostic=False,
               min_nclasses_in_positive_images=0,
               mode=None,
               folds=None,
               feas_key=None):
    self.cls_reader = DataReader(root_path=classification_root,
                                 rng_fn=rng_fn, split=split,
                                 folds=folds, feas_key=feas_key)
    self.pos_cls_reader = self.cls_reader

    ## Create a fucntion that randomly picks a class
    all_classes = range(1, self.cls_reader.nr_classes)
    class_chooser = RandomIterator(len(all_classes), rng_fn)
    self.pick_a_class = lambda: all_classes[class_chooser.getNextIndices(num=1)[0]]

    self.k_shot = k_shot
    self.noise_rate = noise_rate
    self.class_agnostic = class_agnostic
    self.rng_fn = rng_fn


    if min_nclasses_in_positive_images > 0:
      def _roi_filter(roi):
        return len(set(roi.classes)) >= min_nclasses_in_positive_images
      create_seprate_pos_reader = True
    else:
      _roi_filter = None
      create_seprate_pos_reader = False


    # Default target class retriver
    self._retrive_next_class = self._next_class

    #### Other data reader modes
    assert(mode in [None, 'normal', 'aggregate', 'test', 'aggregate,overlap'])
    if mode is not None and 'aggregate' in mode:
      self._retrive_next_class = self._next_class_aggregate_mode
      self._epoch = 0
      ## class_idx 0 is reservsed for bg
      self._class_idx = 1
      create_seprate_pos_reader = True
    elif mode == 'test':
      self._retrive_next_class = self._next_class_test_mode
      self._last_pick = None


    self._overlap_dp = mode is not None and 'overlap' in mode
    self._last_dp = [None]

    ## if necessary creates a separate data reader for positive images
    if create_seprate_pos_reader:
      self.pos_cls_reader = DataReader(root_path=classification_root,
                                       rng_fn=rng_fn, split=split,
                                       roi_filter_fn=_roi_filter,
                                       folds=folds, feas_key=feas_key)

  @property
  def meta(self):
    if 'meta' in self.cls_reader.data:
      return self.cls_reader.data.meta
    else:
      return 'y0x0y1x1,absolute'

  def _next_class_aggregate_mode(self):
    def _increase_class_idx():
      return self.pos_cls_reader.class_iterators[self._class_idx].epoch > self._epoch

    ## keep using the current class_idx until all the datapoints are seen once
    if _increase_class_idx():
     print('Read all the images in class {}'.format(self._class_idx))
     self._class_idx += 1

     # Seen all the classes once; go to next epoch
     if self._class_idx >= self.pos_cls_reader.nr_classes:
       self._class_idx = 1
       self._epoch += 1
       assert(not _increase_class_idx())
       print('Read the dataset once. Starting epoch {} now.'.format(self._epoch))

    ## Returning img and roi is optional
    return None, None, self._class_idx

  def _next_class_test_mode(self):
    class_idx = None
    while class_idx is None:
      if self._last_pick is not None:
        imgs, rois, idx, unique_classes = self._last_pick
        idx += 1
        if idx < len(unique_classes):
          self._last_pick[-2] = idx
          return imgs[0], rois[0], unique_classes[idx]
        else:
          self._last_pick = None

      imgs, rois = self.pos_cls_reader.next_images_data(1)
      roi = rois[0]
      unique_classes = list(set(roi.classes))
      if len(unique_classes) > 1:
        idx = 0
        class_idx = unique_classes[idx]
    self._last_pick = [imgs, rois, idx, unique_classes]
    return imgs[0], roi, class_idx

  def _next_class(self):
    class_idx = None
    while class_idx is None:
      imgs, rois = self.pos_cls_reader.next_images_data(1)
      roi = rois[0]
      if len(roi.classes) == 0:
        if 'image_class' in roi.keys():
          class_idx = roi.image_class
      else:
        class_idx = self.rng_fn().choice(roi.classes)
      if class_idx is not None and self.pos_cls_reader.get_class_len(class_idx) < self.k_shot:
        class_idx = None
    return imgs[0], roi, class_idx

  def _retrive_targets(self, rois, class_idx):
    boxes, classes = [], []
    for roi in rois:
      b, c = [], []
      for cls, box in zip(roi['classes'], roi['boxes']):
        if self.class_agnostic:
          if cls == class_idx:
            b.append(box)
            c.append(1.0)
        else:
          b.append(box)
          c.append(cls)

      b = np.array(b, dtype=np.float32)
      # Reshape is only necessary for empty b
      boxes.append(np.reshape(b, [-1, 4]))
      classes.append(np.array(c, dtype=np.float32))
    return boxes, classes

  def _get_noisy_pos_bags(self):
    n_out_of_class = 0
    if self.noise_rate > 0.0:
      n_out_of_class = self.rng_fn().binomial(self.k_shot,
                                        self.noise_rate)

    pimgs, prois, class_idx = self._get_pos_bags(self.k_shot - n_out_of_class)
    nimgs, nrois = self._get_neg_bags(n_out_of_class, class_idx)
    pimgs.extend(nimgs)
    prois.extend(nrois)
    return pimgs, prois, class_idx

  def _get_pos_bags(self, n):
    img, roi, class_idx = self._retrive_next_class()
    if n == 0:
      return [], [], class_idx


    if class_idx == self._last_dp[0]:
      assert(n > 1)
      imgs, rois = self.pos_cls_reader.next_class_data(class_idx, n-1)
      _, pre_img, pre_roi = self._last_dp
      imgs.insert(0, pre_img)
      rois.insert(0, pre_roi)
    else:
      imgs, rois = self.pos_cls_reader.next_class_data(class_idx, n)

    ## put img in the set
    if img is not None and img not in imgs:
      imgs[0], rois[0] = img, roi

    if self._overlap_dp:
      self._last_dp = [class_idx, imgs[-1], rois[-1]]

    return imgs, rois, class_idx

  def _get_neg_bags(self, n, class_idx, prefered_classes=None):
    imgs, rois = [], []
    retry = 1000
    for i in range(n):
        while True:
          assert(retry), 'Could not find negative images.'
          retry -= 1
          if prefered_classes is not None and len(prefered_classes) > 0:
            nclass_idx = self.rng_fn().choice(prefered_classes)
          else:
            nclass_idx = self.pick_a_class()
          if nclass_idx == class_idx:
            continue
          nimgs, nrois = self.cls_reader.next_class_data(nclass_idx, 1)
          if class_idx in nrois[0]['classes']:
            continue
          if nimgs[0] in imgs:
            continue
          break
        imgs.append(nimgs[0])
        rois.append(nrois[0])
    return imgs, rois

  def next(self):
    imgs, rois, class_idx = self._get_noisy_pos_bags()
    boxes, classes = self._retrive_targets(rois, class_idx)
    return imgs, boxes, classes, class_idx
