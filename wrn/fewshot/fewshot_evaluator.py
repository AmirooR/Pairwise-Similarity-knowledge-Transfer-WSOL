"""CoLocDetection model evaluator.

This file provides a generic evaluation method that can be used to evaluate a
CoLocDetectionModel.
"""
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

import itertools
from tensorflow.python.platform import flags
import time

FLAGS = flags.FLAGS

slim = tf.contrib.slim

EVAL_METRICS_FN_DICT = det_evaluator.EVAL_METRICS_FN_DICT

def _extract_prediction_tensors(model,
                                input_k_shot,
                                create_input_dict_fn,
                                ignore_groundtruth=False):
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
  prefetch_queue = prefetcher.prefetch(input_dict, capacity=100)
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
  preprocessed_image = tf.concat(preprocessed_images, 0)
  prediction_dict = model.predict(preprocessed_image)
  detections = model.postprocess(prediction_dict)

  tensor_dict = dict()
  tensor_dict.update(model._tree_debug_tensors(
      include_k2_cross_similarty=False)) #FLAGS.add_mrf

  tensor_dict['input_k_shot'] = tf.constant(input_k_shot)
  tensor_dict['num_negative_bags'] = tf.constant(num_negative_bags)
  tensor_dict['bag_size'] = tf.constant(bag_size)

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
  ndict = dict(boxes=[], classes=[])
  ndict['target_class'] = input_dict[fields.groundtruth_target_class]
  for i in range(total_images):
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
        ignore_groundtruth=eval_config.ignore_groundtruth)
      for create_input_dict_fn in create_input_dict_fn_list]

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
    if batch_index >= eval_config.num_visualizations:
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
    meta_info = result_dict.pop('meta_info', dict())
    input_k_shot = result_dict.pop('input_k_shot')
    num_negative_bags = result_dict.pop('num_negative_bags')
    bag_size = result_dict.pop('bag_size')
    scores = result_dict['Tree_K2']['scores'][0].reshape((bag_size, bag_size))
    sim = np.argmax(scores, axis=1)
    gt_classes = result_dict['groundtruth']['classes']
    acc = np.mean(gt_classes[1][sim] == gt_classes[0])
    counters['success'] += 1
    result_dict['mean_acc'] = [acc]
    result_dict.pop('original_image', None)
    result_dict.pop('Tree_K1')
    result_dict.pop('Tree_K2')
    result_dict.pop('groundtruth')
    global_step = tf.train.global_step(sess, slim.get_global_step())

    return result_dict

  def _process_aggregated_results(result_lists):
    stds = np.std(result_lists)
    ci95 = 1.96*stds/np.sqrt(len(result_lists))
    mean_acc = np.mean(result_lists)
    print('MEAN ACCURACY: {}, ci95: {}'.format(mean_acc, ci95))
    return {'mean_acc':mean_acc, 'ci95': ci95}

  eval_util.repeated_checkpoint_run(model_names=input_config_names,
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
      use_moving_averages=eval_config.use_moving_averages)
