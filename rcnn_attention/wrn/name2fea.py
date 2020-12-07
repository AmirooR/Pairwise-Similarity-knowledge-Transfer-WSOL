import pickle
import os
import numpy as np
import tensorflow as  tf
from collections import defaultdict
from object_detection.utils.np_box_ops import iou

flags = tf.app.flags
flags.DEFINE_string('ds_root', "/mnt/scratch/amir/detection/amirreza/data/coco", 'dataset root')
flags.DEFINE_string('split', 'trws_k8_aggregate_init', 'name of the init split')
flags.DEFINE_string('save_name', 'k8_init_labelling_info', 'name of save split')
flags.DEFINE_boolean('doublefeas', False, 'use feas as well as det_feas')
flags.DEFINE_boolean('random', False, 'init with random feas')
flags.DEFINE_boolean('objectness', False, 'init with highest objectness fea (which is the '+
                                          'first proposal in our pickle)')
flags.DEFINE_integer('eval_fold', None, 'evaluation fold')
FLAGS= flags.FLAGS

def read_pickle(path):
  with open(path, 'rb') as f:
    return pickle.load(f)

def write_pickle(d, path):
  with open(path, 'wb') as f:
    pickle.dump(d, f)

def main(_):
  ds = read_pickle(os.path.join(FLAGS.ds_root, 'ImageSet', FLAGS.split + '.pkl'))
  name2class2feas = defaultdict(dict)
  class2names = defaultdict(set)
  for i, img in enumerate(ds['images']):
    if (i+1) % 1000 == 0:
      print("Processing image {}/{}".format(i+1, len(ds['images'])))

    if FLAGS.eval_fold is not None and ds['folds'][i] != FLAGS.eval_fold:
      continue
    roi = ds['rois'][i]
    fea_path = ds['det_feas'][i]
    fea = np.load(os.path.join(FLAGS.ds_root,'Feas', fea_path))
    if FLAGS.doublefeas:
      fea_path = ds['feas'][i]
      alexfea = np.load(os.path.join(FLAGS.ds_root, 'Feas', fea_path))
      fea = np.concatenate((alexfea, fea), axis=3)

    assert(np.all(np.unique(roi['classes']) == roi['classes']))
    #TODO last is removed because it is zero
    fea_indices = np.argmax(iou(roi['boxes'], roi['fea_boxes'][:-1]), axis=1)
    for j, cls in enumerate(roi['classes']):
      class2names[cls].add(img)
      name2class2feas[img][cls] = fea[fea_indices[j]]
      if FLAGS.random:
        name2class2feas[img][cls] = fea[np.random.choice(np.arange(fea.shape[0]))]
      if FLAGS.objectness:
        name2class2feas[img][cls] = fea[0]

  save_dict = {'name2class2feas': name2class2feas,
               'class2names': class2names}
  write_pickle(save_dict, os.path.join(FLAGS.ds_root,'ImageSet', FLAGS.save_name +'.pkl'))


if __name__ == '__main__':
  tf.app.run()
