from data_sampler import DataSampler
from tensorpack import *
from tensorpack.dataflow.imgaug.transform import CropTransform, ResizeTransform
import os.path as osp
import numpy as np
from skimage.transform import resize
import copy as copy_mod
import cv2

# imgs: k_shot,height,width,3
# boxes: k_shot, n_boxes, 4 -> this is list of numpy arrays

class TensorpackReader(RNGDataFlow):
  def __init__(self, root, split,
               k_shot=1, shuffle=False,
               noise_rate=0.0,
               mask_dir=None,
               class_agnostic=False,
               folds=None):
    self.root = root
    self.k_shot = k_shot
    self.split = split
    self.mask_dir = mask_dir
    self.folds = folds
    rng = np.random.RandomState(1357)
    def _rng_fn():
      return rng
    self.sampler = DataSampler(root, split,
                               _rng_fn, k_shot,
                               noise_rate,
                               class_agnostic,
                               folds=folds)

    self._shuffle = shuffle
    self._is_shuffled = False

  def size(self):
    return self.sampler.cls_reader.nr_images

  def _load_images(self, img_fns):
    imgs = []
    for fn in img_fns:
      img_path = osp.join(self.root, 'Images', fn)
      img = cv2.imread(img_path, cv2.IMREAD_COLOR)

      if self.mask_dir:
        mask_fn = osp.splitext(fn)[0] + '.png'
        mask_path = osp.join(self.root, self.mask_dir, mask_fn)
        if osp.exists(mask_path):
          mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
          img[mask>0, :] = 0
      imgs.append(img)
    return imgs

  def get_data(self):
    if not self._is_shuffled and self._shuffle:
      def _rng_fn():
        return self.rng
      self.sampler = DataSampler(self.root,
          self.split, _rng_fn, self.k_shot, folds=self.folds)

    for _ in list(xrange(self.size())):
      img_fns, boxes, classes, target_class = self.sampler.next()
      imgs = self._load_images(img_fns)
      yield [imgs, boxes, classes, target_class]

class GoogleNetResize(imgaug.ImageAugmentor):
  """
  crop 8%~100% of the original image
  See `Going Deeper with Convolutions` by Google.
  """
  def __init__(self, crop_area_fraction=0.08,
              aspect_ratio_low=0.75, aspect_ratio_high=1.333):
    self._init(locals())

  def _get_augment_params(self, img):
    h, w = img.shape[:2]
    area = h * w
    for _ in range(10):
      targetArea = self.rng.uniform(self.crop_area_fraction, 1.0) * area
      aspectR = self.rng.uniform(self.aspect_ratio_low, self.aspect_ratio_high)
      ww = int(np.sqrt(targetArea * aspectR) + 0.5)
      hh = int(np.sqrt(targetArea / aspectR) + 0.5)
      if self.rng.uniform() < 0.5:
        ww, hh = hh, ww
      if hh <= h and ww <= w:
        x1 = 0 if w == ww else self.rng.randint(0, w - ww)
        y1 = 0 if h == hh else self.rng.randint(0, h - hh)
        return [CropTransform(y1, x1, hh, ww)]

  def _augment(self, img, params):
    if params:
      for param in params:
        img = param.apply_image(img)
    return img

  def _augment_coords(self, coords, params):
    if params:
      for param in params:
        coords = param.apply_coords(coords)
    return coords

def box_to_point8(boxes):
  """
  Args:
    boxes: nx4
  Returns:
    (nx4)x2
  """
  b = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]]
  b = b.reshape((-1, 2))
  return b

def point8_to_box(points):
  """
  Args:
    points: (nx4)x2
  Returns:
    nx4 boxes (x1y1x2y2)
  """
  p = points.reshape((-1, 4, 2))
  minxy = p.min(axis=1)   # nx2
  maxxy = p.max(axis=1)   # nx2
  return np.concatenate((minxy, maxxy), axis=1)

def normalize_boxes(boxes, classes, img_shape):
  """
    Args:
      nx4 boxes (x1y1x2y2)
      n classes
      img_shape (h,w)
    Returns:
      nx4 normalized, cliped boxes
      in (y1x1y2x2) format
      as it is required by the object detection
      pipeline
  """
  # Clip and normalize the values
  boxes = np.maximum(boxes, 0.0)
  h, w = map(float, img_shape)
  boxes[:, [0,2]] = np.minimum(boxes[:, [0,2]], w)/w
  boxes[:, [1,3]] = np.minimum(boxes[:, [1,3]], h)/h

  # Remove the empty boxes
  hs = boxes[:, 3] - boxes[:, 1]
  ws = boxes[:, 2] - boxes[:, 0]
  assert(all(hs >= 0) and all(ws >= 0))
  assert(len(boxes) == len(classes))
  not_empty = (hs > 0) & (ws > 0)
  boxes = boxes[not_empty]
  classes = classes[not_empty]

  # Convert the format (x1y1x2y2) ==> (y1x1y2x2)
  return boxes[:, [1, 0, 3, 2]], classes

def augmentors(is_training, im_size):
  if is_training:
    return [
      imgaug.Resize((im_size[0], im_size[1]), interp=cv2.INTER_CUBIC),
      imgaug.Flip(horiz=True)]
  else:
    return [imgaug.Resize((im_size[0], im_size[1]), cv2.INTER_CUBIC)]


def augment_supportset(ds, is_training, k_shot, im_size):
  def mapf(ds):
    img_list, points_list, class_list, target_class = ds
    points8_list = []
    for points in points_list:
      if points.dtype != np.float32:
        points = np.array(points, dtype=np.float32)
      points8_list.append(box_to_point8(points))
    return img_list + points8_list + class_list + [target_class]
  ds = MapData(ds, mapf)
  augs = augmentors(is_training, im_size)

  for i in range(k_shot):
    ds = AugmentImageComponents(ds, augs, index=(i,), coords_index=(i+k_shot,))

  def unmapf(ds):
    assert(len(ds) == 3*k_shot+1)
    points8_list = ds[k_shot:2*k_shot]
    class_list = ds[2*k_shot:3*k_shot]
    target_class = ds[-1]
    img_shape = ds[0].shape[:2]
    points_list = []
    nclass_list = []
    for classes, points in zip(class_list, points8_list):
      boxes = point8_to_box(points)
      nboxes, nclasses = normalize_boxes(boxes, classes, img_shape)
      points_list.append(nboxes)
      nclass_list.append(nclasses)
    try:
      imgs = np.stack(ds[:k_shot]).astype(np.float32)
      #convert BGR --> RGB
      imgs = imgs[..., ::-1]
      return [imgs, target_class] + points_list + nclass_list
    except:
      from IPython import embed;embed()
  ds = MapData(ds, unmapf)
  return ds

