from rcnn_attention.dataflow.data_sampler import DataSampler
import numpy as np
from object_detection.utils.np_box_ops import iou as np_iou

class DataSamplerDet(DataSampler):
  def __init__(self,
               classification_root,
               split, rng_fn,
               k_shot=1,
               num_negative_bags=0,
               noise_rate=0.0,
               class_agnostic=False,
               min_nclasses_in_positive_images=0,
               smart_neg_bag_sampler=False,
               pos_threshold = 0.5,
               objectness_key='objectness',
               mode=None,
               folds=None,
               feas_key=None):

    super(DataSamplerDet, self).__init__(classification_root,
                                         split, rng_fn, k_shot,
                                         noise_rate=noise_rate,
                                         class_agnostic=class_agnostic,
        min_nclasses_in_positive_images=min_nclasses_in_positive_images,
                                         mode=mode,
                                         folds=folds,
                                         feas_key=feas_key)
    self.num_negative_bags = num_negative_bags
    self._smart_neg_bag_sampler = smart_neg_bag_sampler
    self.pos_threshold = pos_threshold
    self.objectness_key = objectness_key

  def _retrive_targets(self, rois, class_idx):
    boxes, classes, fea_boxes, fea_classes, objectness = [], [], [], [], []
    for roi in rois:
      b, c, fea_c = [], [], []
      for cls, box in zip(roi['classes'], roi['boxes']):
        if self.class_agnostic:
          if cls == class_idx:
            b.append(box)
            c.append(1.0)
        else:
          b.append(box)
          c.append(cls)
      overlaps = np_iou(roi['fea_boxes'], roi['boxes'].astype(np.float32))
      class_idx = float(class_idx)
      for ov in overlaps:
        fea_cls = np.unique(roi['classes'][ov >= self.pos_threshold]
                                                ).astype(np.float32)
        pos = class_idx in fea_cls
        if self.class_agnostic:
          fea_c.append(1.0 if pos else 0.0)
        else:
          if len(fea_cls) == 0:
            fea_c.append(0.0)
          else:
            fea_c.append(class_idx if pos else self.rng_fn().choice(fea_cls))

      b = np.array(b, dtype=np.float32)
      fea_b = roi['fea_boxes'].astype(np.float32)
      # Reshape is only necessary for empty b
      boxes.append(np.reshape(b, [-1, 4]))
      classes.append(np.array(c, dtype=np.float32))
      fea_boxes.append(np.reshape(fea_b,[-1,4]))
      fea_classes.append(np.array(fea_c, dtype=np.float32))

      if roi.has_key(self.objectness_key):
        objectness.append(roi[self.objectness_key].astype(np.float32))
      else:
        objectness.append(np.zeros((fea_b.shape[0],), dtype=np.float32))
        # Only for debuging
        #objectness.append((np.array(fea_c) == class_idx).astype(np.float32))
    return boxes, classes, fea_boxes, fea_classes, objectness

  def next(self):
    """
    define:
      total_bags = k_shot + num_negative_bags
    Returns:
      imgs: list of size total_bags of image addresses (strings) relative to cls_root
      classes: list of size total_bags of [N_k,4] numpy arrays representing absolute box
        coordinates where N_k is the number of groundtruth objects in k-th image.
      classes: list of size total_bags of [N_k,] numpy arrays representing class of each
        object in k-th image.
      fea_boxes: list of size total_bags of [300,4] numpy arrays representing absolute box
        coordinates of region proposals of k-th image
      fea_classes: list of size total_bags of [300,] numpy arrays which represents the
        class of each box in fea_boxes. Class zero represents the background class
      class_idx: the selected common class id (starts from 1)
    """
    imgs, rois, class_idx = self._get_noisy_pos_bags()
    prefered_classes = None
    if self._smart_neg_bag_sampler:
      prefered_classes = []
      for roi in rois:
        prefered_classes.extend(roi['classes'])
      prefered_classes = list(np.unique(prefered_classes))
      if class_idx in prefered_classes:
        prefered_classes.remove(class_idx)
    nimgs, nrois = self._get_neg_bags(self.num_negative_bags, class_idx,
                                      prefered_classes=prefered_classes)
    imgs.extend(nimgs)
    rois.extend(nrois)
    (boxes, classes, fea_boxes,
        fea_classes, objectness) = self._retrive_targets(rois, class_idx)
    return imgs, boxes, classes, fea_boxes, fea_classes, objectness, class_idx
