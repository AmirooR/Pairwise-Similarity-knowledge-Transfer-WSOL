import tensorflow as tf
import os
import numpy as np
import pickle
from collections import defaultdict
import operator


flags = tf.app.flags
flags.DEFINE_string('ds_root', None, 'Path to dataset root')
flags.DEFINE_string('train_split', None, 'train_split')
flags.DEFINE_string('gt_split', None, 'gt_split')
flags.DEFINE_integer('num_folds', 10, 'num folds')
flags.DEFINE_integer('min_class_counts', 32, 'min class count in a fold')

FLAGS= flags.FLAGS

def read_pickle(fn):
  assert(os.path.exists(fn)), fn
  with open(fn, 'rb') as f:
    return pickle.load(f)

def write_pickle(d, fn):
  with open(fn, 'wb') as f:
    pickle.dump(d,f)

def get_class_indices_and_counts_sorted(d):
  class_counts = defaultdict(int)
  class_indices = defaultdict(list)
  for i, roi in enumerate(d['rois']):
    for k in np.unique(roi['classes']):
      class_counts[k] += 1
      class_indices[k].append(i)
  # sorted from largest number of images to lowest
  # This is because we easily rewrite by the lowest values if multiple classes appear in
  # a single image
  sorted_class_counts = sorted(class_counts.items(), key=operator.itemgetter(1))[::-1]
  return sorted_class_counts, class_indices

def get_folds(train_ds, num_folds):
  sorted_class_counts, class_indices = get_class_indices_and_counts_sorted(train_ds)
  folds = np.ones(len(train_ds['rois']), dtype=np.int64) * -1

  for k,_ in sorted_class_counts:
    start_fold = np.random.randint(0, num_folds)
    indices_k = np.array(class_indices[k])
    folds_k = (np.random.permutation(len(indices_k)) + start_fold) % num_folds
    folds[indices_k] = folds_k
  return folds

def check_fold_counts(train_ds, folds, num_folds, min_class_count_in_fold=32):
  assert(folds.min() >= 0)
  for fold in range(num_folds):
    fold_class_count = defaultdict(int)
    for i, f in enumerate(folds):
      if f == fold:
        for k in np.unique(train_ds['rois'][i]['classes']):
          fold_class_count[k] += 1
    assert(min(fold_class_count.values()) >= min_class_count_in_fold), "fold class counts: {}\
        ".format(fold_class_count)


def main(_):
  train_ds = read_pickle(os.path.join(FLAGS.ds_root, 'ImageSet', FLAGS.train_split)+'.pkl')
  gt_ds = read_pickle(os.path.join(FLAGS.ds_root, "ImageSet", FLAGS.gt_split)+'.pkl')
  assert(len(gt_ds['rois']) == len(train_ds['rois']))
  folds = get_folds(train_ds, FLAGS.num_folds)
  check_fold_counts(train_ds, folds, FLAGS.num_folds, FLAGS.min_class_counts)

  gt_folds = np.zeros_like(folds)
  for i, fold in enumerate(folds):
    train_img = train_ds['images'][i]
    gt_ind = gt_ds['images'].index(train_img)
    gt_folds[gt_ind] = fold

  train_ds['folds'] = folds
  gt_ds['folds'] = gt_folds
  write_pickle(train_ds, os.path.join(FLAGS.ds_root, 'ImageSet', FLAGS.train_split)+'.pkl')
  write_pickle(gt_ds, os.path.join(FLAGS.ds_root, 'ImageSet', FLAGS.gt_split)+'.pkl')


if __name__ == '__main__':
  tf.app.run()

