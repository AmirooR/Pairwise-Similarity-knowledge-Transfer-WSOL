import tensorflow as tf
import os
import cPickle as pickle
import glob
import opengm
import time
import numpy as np
from object_detection.utils.np_box_ops import iou

flags = tf.app.flags
flags.DEFINE_string('gm_db_root', 'gm_db/k4', 'Path to graphical model stored dataset')
flags.DEFINE_string('dataset_root', '../../data/coco', 'Path to the original dataset root')
flags.DEFINE_string('groundtruth_split', 'alexfea301_normalized_unseen', 'name of the groundtruth split')
flags.DEFINE_string('inf_type', 'trws', 'Inference type')
FLAGS= flags.FLAGS

def read_pickle(path):
  with open(path,'rb') as f:
    return pickle.load(f)

def merge_gm_dbs(gm_db_paths):
  print("Mergin gm dbs")
  print(gm_db_paths[0]+' ...')
  gm_db = read_pickle(gm_db_paths[0])
  for db_path in gm_db_paths[1:]:
    print(db_path+' ...')
    db = read_pickle(db_path)
    if len(gm_db['unaries']) > 0:
      gm_db['unaries'].update(db['unaries'].copy())
    gm_db['pairwises'].update(db['pairwises'].copy())
  return gm_db

def create_graph(gm_db):
  print("Creating Graph ...")
  all_name_pairs = gm_db['pairwises'].keys()
  all_names = set()
  for n1,n2 in all_name_pairs:
    all_names.add(n1)
    all_names.add(n2)
  node2id = {}
  id2node = {}
  num_labels = gm_db['pairwises'][all_name_pairs[0]].shape[0]
  for i,n in enumerate(all_names):
    node2id[n] = i
    id2node[i] = n
  num_nodes = len(all_names)
  print("Num nodes: {}".format(num_nodes))
  gm = opengm.gm([num_labels]*num_nodes, operator='adder')
  if len(gm_db['unaries']) > 0:
    for i in range(num_nodes):
      gm.addFactor(gm.addFunction(gm_db['unaries'][id2node[i]]), [i])
  for n1,n2 in all_name_pairs:
    id1 = node2id[n1]
    id2 = node2id[n2]
    t = [id1,id2]
    p = gm_db['pairwises'][(n1,n2)]
    if id2 < id1:
      t = [id2,id1]
      p = p.T
    if id1 == id2:
      continue
    gm.addFactor(gm.addFunction(p), t)
  print(" - Done")
  return gm, node2id, id2node

def run_inference(gm):
  print("Running Inference...")
  if FLAGS.inf_type == 'trws':
    inf = opengm.inference.TrwsExternal(gm, accumulator='minimizer')
  else:
    raise ValueError("Inference type is not supported")
  t0 = time.time()
  inf.infer()
  argmin = inf.arg()
  t1 = time.time() - t0
  energy = inf.value()
  print(" - Inference done: {}sec, energy: {}".format(t1,energy))
  return argmin, t1, energy

def evaluate_class(argmin, id2node, node2id, gt_ds, cls, thresh=0.5):
  total = 0
  correct = 0
  for i, imgname in enumerate(gt_ds['images']):
    roi = gt_ds['rois'][i]
    if cls in roi['classes']:
      total += 1
      boxes = roi['boxes'][roi['classes'] == cls]
      node_id = node2id[imgname]
      argmin_box = roi['fea_boxes'][argmin[node_id]][None]
      if iou(argmin_box, boxes).max() > thresh:
        correct += 1

  print("Class: {}, Total: {}, correct: {}, CorLoc: {}".format(
    cls, total, correct, float(correct)/total))

  return float(correct)/total

def main(_):
  print("Reading groundtruth dataset")
  gt_ds = read_pickle(os.path.join(FLAGS.dataset_root,
                                   'ImageSet',
                                   FLAGS.groundtruth_split+'.pkl'))

  all_class_corlocs = []
  for cls in range(1, len(gt_ds['synsets'])):
    gm_db_path_list = glob.glob(os.path.join(FLAGS.gm_db_root, 'mrf_class_{}_*.pkl'.format(cls)))
    gm_db = merge_gm_dbs(gm_db_path_list)
    gm, node2id, id2node = create_graph(gm_db)
    argmin, t1, energy = run_inference(gm)
    class_corloc = evaluate_class(argmin, id2node, node2id, gt_ds, cls, thresh=0.5)
    all_class_corlocs.append(class_corloc)

  print("Mean CorLoc: {}".format(np.mean(all_class_corlocs)))

if __name__ == '__main__':
  tf.app.run()

