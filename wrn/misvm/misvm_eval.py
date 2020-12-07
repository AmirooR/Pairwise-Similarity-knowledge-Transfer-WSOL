import argparse
import os, sys
import logging
import numpy as np
from tqdm import tqdm
import data_reader
import utils
from IPython import embed
import json
import svm_factory
import time

parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', default='experiments/', help='experiments folder')
#reader params
parser.add_argument('--k', default=4, type=int, help="Number of positive bags")
parser.add_argument('--neg', default=2, type=int, help="Number of negative bags")
parser.add_argument('--b', default=5, type=int,
    help="Number of image proposals per bag")
parser.add_argument('--split', default="test", help="split to use for evaluation")
parser.add_argument('--dataset', default='miniimagenet', help="name of the dataset")
#coco only
parser.add_argument('--data_root',
  default='/home/amir/CoLoc/rcnn_attention/feature_extractor/logs/coco/k1/extract{}',
  help="path to features")
# svm params
parser.add_argument('--balance', default=True, type=bool,
    help="balances the negative bags by copying them " \
        "to be have equal number of positive and negative bags")

parser.add_argument('--whiten', default=False, type=bool,
    help="If True, whitens the features of each problem ")
parser.add_argument('--normalize', default='nonorm', help='one of nonorm, l1, l2 norms')
# eval params
parser.add_argument('--eval_num', default=500, type=int,
    help="number of evaluation problems")
parser.add_argument('--print_every', default=100, type=int,
    help="print evaluation every number of evaluated problems")


def add_results(result_dict, result_lists):
  for key,val in result_dict.items():
    if isinstance(val, dict):
      if key not in result_lists:
        result_lists[key] = dict()
      add_results(val, result_lists[key])
    else:
      if key not in result_lists:
        result_lists[key] = []
      result_lists[key].extend(val)

def evaluate(reader, params, name, classifiers):
  #imported here in order to have correct logging functionality
  from rcnn_attention import evaluator
  k_shot = params.k
  num_negative_bags = params.neg
  total_bags = k_shot + num_negative_bags
  result_lists = {}
  start_time = time.time()
  for i, data in enumerate(reader.get_data()):
    time_elapsed = time.time() - start_time
    if (i+1) % params.print_every == 0:
      logging.info('Evaluating problem number {}/{} ({:.3f}s)'.format(i+1,
        params.eval_num,
        time_elapsed))
    [feas, fea_boxes, fea_target_classes,
        fea_classes, imgs, target_class] = data[0:6]
    boxes_list = data[6:6+total_bags]
    class_list = data[6+total_bags:]

    # (total_bags,1,1,ndim) -> (total_bags, ndim)
    bags = np.squeeze(feas)
    bag_labels = 2*np.max(fea_target_classes, axis=1)-1
    if params.whiten:
      b_mean = np.mean(bags, axis=(0,1))
      b_std = np.std(bags, axis=(0,1)) + 1e-18
      bags = (bags - b_mean)/b_std
    if params.normalize == 'l2':
      bags /= np.linalg.norm(bags, axis=2, keepdims=True)
    elif params.normalize == 'l1':
      raise NotImplemented('l1 is not implemented yet')
    elif params.normalize == 'nonorm':
      pass
    else:
      raise ValueError('normalize param {} is not defined'.format(params.normalize))
    if params.balance:
      num_to_add = k_shot - num_negative_bags
      if num_to_add > 0:
        bags = np.vstack((bags, bags[-num_to_add:]))
        bag_labels = np.hstack((bag_labels, bag_labels[-num_to_add:]))

    for method, classifier_fn in classifiers.items():
      classifier = classifier_fn(verbose=False)
      t0 = time.time()
      classifier.fit(bags, bag_labels)
      _, predictions = classifier.predict(bags, instancePrediction=True)
      inference_time = time.time() - t0
      predictions = predictions.reshape((-1, params.b))[:params.k] #positive bags
      result_dict = {}
      res = {'boxes': fea_boxes[:params.k],
                     'classes': np.ones_like(fea_classes[:params.k]),
                     'scores': predictions,
                     'class_agnostic': True}
      result_dict[method] = res
      gt = {}
      gt['boxes'] = boxes_list[:params.k]
      gt['classes'] = class_list[:params.k]
      gt['target_class'] = target_class
      result_dict['groundtruth'] = gt
      evaluator._postprocess_result_dict(result_dict)
      meta_info = {}
      meta_info[method+'_inference_time'] = [inference_time]
      result_dict['meta_info'] = meta_info
      result_dict.pop('groundtruth')
      add_results(result_dict, result_lists)

    if i+1 == params.eval_num:
      break
  meta_info = result_lists.pop('meta_info', None)
  metrics = {}
  #imported here to have correct logging functionality
  from rcnn_attention import eval_util
  for method, result_list in result_lists.items():
    m = eval_util.evaluate_coloc_results(result_list, None)
    metrics[method] = m
  metrics.update(eval_util.process_meta_info(meta_info))
  for k,v in metrics.items():
    logging.info('{}: {}'.format(k,v))

if __name__ == '__main__':
  args = parser.parse_args()
  args.data_root = args.data_root.format(args.b)
  name = 'dataset_{}-k_{}-neg_{}-b_{}-split_{}'.format(args.dataset,
                                                       args.k,
                                                       args.neg,
                                                       args.b,
                                                       args.split)
  classifier_params_path = os.path.join(args.exp_dir, 'params.json')
  assert os.path.isfile(classifier_params_path), "No json configuration for " \
      "classifiers in {}".format(classifier_params_path)
  with open(classifier_params_path, 'r') as f:
    classifier_params = json.load(f)
  classifiers = svm_factory.get_classifiers(classifier_params)
  utils.set_logger(os.path.join(args.exp_dir, name+'.log'))
  logging.info('loading datasets for {}'.format(name))
  reader = data_reader.get_data_reader(args)
  logging.info(' - done.')
  evaluate(reader, args, name, classifiers)

