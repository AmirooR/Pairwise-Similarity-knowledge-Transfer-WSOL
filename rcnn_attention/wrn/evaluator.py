"""CoLocDetection model evaluator.

This file provides a generic evaluation method that can be used to evaluate a
CoLocDetectionModel.
"""
import copy
import functools
import logging
import tensorflow as tf
from rcnn_attention.attention import util
from rcnn_attention import eval_util
from rcnn_attention import evaluator as det_evaluator
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import prefetcher
from rcnn_attention.coloc import standard_fields
fields = standard_fields.CoLocInputDataFields
from object_detection.utils import ops
import numpy as np
from IPython import embed
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

import itertools
from tensorflow.python.platform import flags
from calibration.calibration_utils import add_calibration_functions
import time

FLAGS = flags.FLAGS

slim = tf.contrib.slim

EVAL_METRICS_FN_DICT = det_evaluator.EVAL_METRICS_FN_DICT

def _extract_prediction_tensors(model,
                                input_k_shot,
                                create_input_dict_fn,
                                ignore_groundtruth=False,
                                use_target_class_in_predictions=False):
  """Restores the model in a tensorflow session.

  Args:
    model: model to perform predictions with.
    create_input_dict_fn: function to create input tensor dictionaries.
    ignore_groundtruth: whether groundtruth should be ignored.

  Returns:
    tensor_dict: A tensor dictionary with evaluations.
  """
  tree_k_shot = model._k_shot
  bag_size = model._bag_size
  num_classes = model._num_classes
  num_negative_bags = model._num_negative_bags
  total_images = num_negative_bags + input_k_shot
  input_dict, thread = create_input_dict_fn()
  prefetch_queue = prefetcher.prefetch(input_dict, capacity=3)
  input_dict = prefetch_queue.dequeue()
  def _r(t):
    return tf.reshape(t, [-1,]+t.shape[2:].as_list())

  def reorder_for_k2(t):
    '''
    when tree_k_shot is 2, and we have negative bags and input_k_shot is greater than 2,
    this function reorders the list of inputs in a way that it puts the negative elements
    after every 2*bag_size elements.
    Notes:
      - assumes positive elements are in first bag_size*input_k_shot elements of the input list
      and the rest of the elements are negative bags i.e., num_negative_bags*bag_size last
      elements of the list.
    '''
    if num_negative_bags > 0 and tree_k_shot == 2 and input_k_shot != 2:
      #TDOO: 'implemented, but not tested yet! use embed to check'
      pos_part = t[:bag_size*input_k_shot]
      neg_part = t[bag_size*input_k_shot:]
      assert input_k_shot % 2 == 0, 'input_k_shot should be multiple of 2'
      l = [pos_part[bag_size*2*i:bag_size*2*(i+1)] + neg_part for i in range(input_k_shot/2)]
      t = [i for x in l for i in x]
    return t

  def _s(t):
    return reorder_for_k2(tf.split(_r(t), total_images*bag_size))

  images_list = _s(input_dict[fields.supportset])
  groundtruth_classes_list = _s(input_dict[fields.groundtruth_classes])
  original_groundtruth_classes = input_dict[fields.original_groundtruth_classes]
  original_groundtruth_classes_list = _s(original_groundtruth_classes)
  one_hot_gt_classes_list = _s(tf.one_hot(tf.to_int32(original_groundtruth_classes), num_classes + 1))
  boxes_list = _s(input_dict[fields.groundtruth_boxes])

  images = tf.concat(images_list,0)
  float_images = tf.to_float(images)
  input_dict[fields.supportset] = float_images
  preprocessed_images = [model.preprocess(float_image)
                          for float_image in
                          tf.split(float_images, len(images_list))]
  model.provide_groundtruth(boxes_list, one_hot_gt_classes_list, None)

  if fields.proposal_objectness in input_dict:
    model._groundtruth_lists['objectness_score'] = input_dict[fields.proposal_objectness]

  if use_target_class_in_predictions:
    target_class = input_dict[fields.groundtruth_target_class]
    model._groundtruth_lists['target_class'] = target_class[tf.newaxis]

  preprocessed_image = tf.concat(preprocessed_images, 0)
  prediction_dict = model.predict(preprocessed_image)
  detections = model.postprocess(prediction_dict)

  tensor_dict = dict()
  tensor_dict.update(model._tree_debug_tensors(
      include_k2_cross_similarty=False)) #FLAGS.add_mrf

  tensor_dict['input_k_shot'] = tf.constant(input_k_shot)
  tensor_dict['num_negative_bags'] = tf.constant(num_negative_bags)

  label_id_offset = 1
  # Convert to the absolute coordinates
  for key,val in tensor_dict.items():
    if 'multiclass' in key:
        tensor_dict.pop(key)
    if isinstance(val, dict):
      for mkey in val:
        if 'classes' in mkey:
          val[mkey] = val[mkey] + label_id_offset

  groundtruth_classes = tf.concat(groundtruth_classes_list[:input_k_shot*bag_size], 0)
  groundtruth_classes = tf.reshape(groundtruth_classes, [input_k_shot, bag_size])
  original_groundtruth_classes = tf.concat(original_groundtruth_classes_list[:input_k_shot*bag_size],0)
  original_groundtruth_classes = tf.reshape(original_groundtruth_classes,
                                                      [input_k_shot, bag_size])
  if not fields.original_images in input_dict:
    raise Exception('Please add this tensors to mini-imagenet model, guzu.' \
                    ' It is added but ridam passe kallat')


  tensor_dict['original_image'] = input_dict[fields.original_images]
  tensor_dict['filenames'] = input_dict[fields.filename]
  ndict = dict(boxes=[], classes=[])
  ndict['target_class'] = input_dict[fields.groundtruth_target_class]
  for i in range(input_k_shot):
    ndict['boxes'].append(
        input_dict[fields.groundtruth_image_boxes+'_{}'.format(i)])
    ndict['classes'].append(input_dict[
        fields.groundtruth_image_classes+'_{}'.format(i)])
  #ndict['fea_groundtruth_classes'] = groundtruth_classes
  #ndict['fea_original_groundtruth_classes'] = original_groundtruth_classes
  #ndict['fea_boxes'] = input_dict[fields.groundtruth_boxes]
  tensor_dict['groundtruth'] = ndict

  ####
  #tensor_dict['meta_info']['extra_tensors'] = util.get_extra_tensors_dict()
  ####
  return tensor_dict, thread

def get_k(str_list):
   for i in range(10):
       k = 2**(i+1)
       if not 'Tree_K{}'.format(k) in str_list:
           return k//2
   return -1

def evaluate(create_input_dict_fn_list, input_config_names,
             create_model_fn, eval_config, categories,
             checkpoint_dir, eval_dir, input_k_shot):
  """Evaluation function for detection models.

  Args:
    create_input_dict_fn: a function to create a tensor input dictionary.
    create_model_fn: a function that creates a DetectionModel.
    eval_config: a eval_pb2.EvalConfig protobuf.
    categories: a list of category dictionaries. Each dict in the list should
                have an integer 'id' field and string 'name' field.
    checkpoint_dir: directory to load the checkpoints to evaluate from.
    eval_dir: directory to write evaluation metrics summary to.
  """
  if eval_config.ignore_groundtruth and not eval_config.export_path:
    logging.fatal('If ignore_groundtruth=True then an export_path is '
                  'required. Aborting!!!')

  extract_prediction_tensors_fn_list = [
      lambda create_input_dict_fn=create_input_dict_fn:
      _extract_prediction_tensors(
        model=create_model_fn(),
        input_k_shot=input_k_shot,
        create_input_dict_fn=create_input_dict_fn,
        ignore_groundtruth=eval_config.ignore_groundtruth,
        use_target_class_in_predictions=eval_config.use_target_class_in_predictions)
      for create_input_dict_fn in create_input_dict_fn_list]

  required_batch_indices = []
  #required_batch_indices = [293, 341, 432, 710, 940, 959, 961, 980]
  #all
  #required_batch_indices = [1,18,67,170, 204,226,261,274,278,306,361,375,385,485,488,760,445,503,531,573,582,583,618,630,671,744,863,896,903,906,936,952,970]
  #required_batch_indices = range(1000)
  #cup
  #required_batch_indices = range(1000)
  #good test_mode=True
  #required_batch_indices = [126,127,154,155,207,208,295,296,318,319,429,430,512,513,639,640,655,656,803,804,832,833,890,891,894,895]
  #required_batch_indices = [306, 488, 744]
  saved_results = []
  saved_indices = []
  def _process_batch(tensor_dict, sess, batch_index, counters,
                     update_op, model_name=None):
    """Evaluates tensors in tensor_dict, visualizing the first K examples.

    This function calls sess.run on tensor_dict, evaluating the original_image
    tensor only on the first K examples and visualizing detections overlaid
    on this original_image.

    Args:
      tensor_dict: a dictionary of tensors
      sess: tensorflow session
      b#atch_index: the index of the batch amongst all batches in the run.
      counters: a dictionary holding 'success' and 'skipped' fields which can
        be updated to keep track of number of successful and failed runs,
        respectively.  If these fields are not updated, then the success/skipped
        counter values shown at the end of evaluation will be incorrect.
      update_op: An update op that has to be run along with output tensors. For
        example this could be an op to compute statistics for slim metrics.

    Returns:
      result_dict: a dictionary of numpy arrays
    """
    tensor_dict = dict(tensor_dict)
    if batch_index >= eval_config.num_visualizations:
      if not batch_index in required_batch_indices:
        tensor_dict.pop('original_image', None)
    def filter_nones(d):
      d = dict(filter(lambda item: item[1] is not None, d.items()))
      for k, v in d.iteritems():
        if type(v) == dict:
          d[k] = filter_nones(d[k])
      return d
    tensor_dict = filter_nones(tensor_dict)
    t0 = time.time()

    (result_dict, _) = sess.run([tensor_dict, update_op])
    filenames = result_dict.pop('filenames', None)
    gt = result_dict['groundtruth']
    k = get_k(result_dict.keys())
    forward_time = time.time() - t0
    meta_info = result_dict.pop('meta_info', dict())
    meta_info['forward_time'] = [forward_time]
    meta_info['positive_images_name'] = filenames[:k]
    input_k_shot = result_dict.pop('input_k_shot')
    num_negative_bags = result_dict.pop('num_negative_bags')
    det_evaluator._postprocess_result_dict(result_dict, compute_upper_bound=True)
    counters['success'] += 1
    global_step = tf.train.global_step(sess, slim.get_global_step())
    if batch_index < eval_config.num_visualizations:
        if False:
          method0 = 'Tree_K1'
          method1 = 'Tree_K8'
          corloc0 = eval_util.corloc(result_dict[method0])[0]
          corloc1 = eval_util.corloc(result_dict[method1])[0]
          condition =  (corloc1 >= corloc0) + 2/8.0
          condition = (corloc1 == 1.0) and condition
          vis_dict = dict()
          if condition:
            print('Improvment: ', corloc1 - corloc0)
            vis_dict[method0] = result_dict[method0]
            vis_dict[method1] = result_dict[method1]
            vis_dict['original_image'] = result_dict['original_image']
        else:
          vis_dict = result_dict
        vis_dict['original_image'] = vis_dict['original_image'][:,:,:,::-1]
        eval_util.visualize_detection_results(
                vis_dict, batch_index, global_step,
                summary_dir=eval_dir,
                export_dir=eval_config.visualization_export_dir,
                show_groundtruth=True,
                max_num_predictions=10,
                min_score_thresh=-10000.0,
                model_name=model_name)

    result_dict['meta_info'] = meta_info
    if (batch_index+1) % 100 == 0:
      print('#total images added: {}'.format(len(saved_results)))
    if batch_index in required_batch_indices:
      method1 = 'Tree_K8'
      method0 = 'Tree_K1'
      corloc0 = eval_util.corloc(result_dict[method0])[0]
      corloc1 = eval_util.corloc(result_dict[method1])[0]
      condition = corloc1 == 1.0
      condition = condition and (corloc0 <= 6/8.0)
      res = dict(result_dict)
      res.pop('meta_info', None)
      res.pop('groundtruth', None)
      if condition:
        saved_indices.append(batch_index)
        saved_results.append(res)
      else:
        result_dict.pop('original_image', None)
      if batch_index == required_batch_indices[-1]:
        name = res.keys()[0]
        import pickle
        logging.info('Saving {} results'.format(len(saved_results)))
        with open('{}_all.pkl'.format(name), 'wb') as f:
          pickle.dump([saved_results, saved_indices], f)
    return result_dict

  def _process_aggregated_results(result_lists):
    eval_metric_fn_key = eval_config.metrics_set
    if eval_metric_fn_key not in EVAL_METRICS_FN_DICT:
      raise ValueError('Metric not found: {}'.format(eval_metric_fn_key))
    return det_evaluator.EVAL_METRICS_FN_DICT[eval_metric_fn_key](result_lists,
                                                         categories=categories)

  eval_util.repeated_checkpoint_run(model_names=input_config_names,
      aggregation_params=eval_config.aggregation_params,
      extract_prediction_tensors_fn_list=extract_prediction_tensors_fn_list,
      summary_dir=eval_dir,
      aggregated_result_processor=_process_aggregated_results,
      batch_processor=_process_batch,
      checkpoint_dirs=[checkpoint_dir],
      variables_to_restore=None,
      num_batches=eval_config.num_examples,
      eval_interval_secs=eval_config.eval_interval_secs,
      max_number_of_evaluations=1,
      master=eval_config.eval_master,
      save_graph=eval_config.save_graph,
      save_graph_dir=(eval_dir if eval_config.save_graph else ''),
      use_moving_averages=eval_config.use_moving_averages,
      load_from_vanilla_into_doubled_net=eval_config.load_from_vanilla_into_doubled_net,
      transfered_network_checkpoint=eval_config.transfered_network_checkpoint)
