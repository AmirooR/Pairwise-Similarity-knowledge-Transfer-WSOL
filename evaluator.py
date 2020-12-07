"""CoLocDetection model evaluator.

This file provides a generic evaluation method that can be used to evaluate a
CoLocDetectionModel.
"""
import functools
import logging
import tensorflow as tf

from rcnn_attention import eval_util
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import prefetcher
from rcnn_attention.coloc import standard_fields as fields
from object_detection.utils import ops
import numpy as np
from IPython import embed
from rcnn_attention.eval_util import normalize_problem_labels, get_k_shot
from object_detection.utils.np_box_ops import iou as np_iou
import copy

slim = tf.contrib.slim

EVAL_METRICS_FN_DICT = {
    'pascal_voc_metrics': eval_util.evaluate_detection_results_pascal_voc,
    'coloc_metrics': eval_util.evaluate_coloc_results
}


def _extract_prediction_tensors(model,
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
  k_shot = model._k_shot
  input_dict, thread = create_input_dict_fn()
  prefetch_queue = prefetcher.prefetch(input_dict, capacity=100)
  input_dict = prefetch_queue.dequeue()
  images = input_dict[fields.CoLocInputDataFields.supportset]
  images_list = [image for image in tf.split(images, k_shot)]
  float_images = tf.to_float(images)
  input_dict[fields.CoLocInputDataFields.supportset] = float_images
  preprocessed_images = [model.preprocess(float_image)
                          for float_image in
                          tf.split(float_images, k_shot)]
  preprocessed_image = tf.concat(preprocessed_images, 0)
  prediction_dict = model.predict(preprocessed_image)
  detections = model.postprocess(prediction_dict)
  original_image_shape = tf.shape(images)

  def _absolute_boxes(normalized_boxes):
    absolute_detection_boxlist_list = [box_list_ops.to_absolute_coordinates(
      box_list.BoxList(tf.squeeze(k, axis=0)),
      original_image_shape[1],
      original_image_shape[2]
      ) for k in tf.split(normalized_boxes,k_shot)]
    return tf.stack([db.get() for db in absolute_detection_boxlist_list])

  tensor_dict = {'original_image': images}

  if detections.has_key('rpn_detection_boxes'):
    tensor_dict['rpn'] = {'boxes': detections['rpn_detection_boxes'],
                          'scores': detections['rpn_detection_scores'],
                          'classes': detections['rpn_detection_classes'],
                          'class_agnostic': tf.constant(True)}

  if detections.has_key('detection_boxes'):
    tensor_dict['detection'] = {'boxes': detections['detection_boxes'],
                                'scores': detections['detection_scores'],
                                'classes': detections['detection_classes'],
                                'class_agnostic': tf.constant(False)}
  label_id_offset = 1
  if hasattr(model, '_tree_debug_tensors'):
    tensor_dict.update(model._tree_debug_tensors())

  # Convert to the absolute coordinates
  for key,val in tensor_dict.items():
    if isinstance(val, dict):
      for mkey in val:
        if 'boxes' in mkey:
          val[mkey] = _absolute_boxes(val[mkey])
        if 'classes' in mkey:
          val[mkey] = val[mkey] + label_id_offset

  if not ignore_groundtruth:
    groundtruth_boxes_list = []
    groundtruth_classes_list = []
    groundtruth_target_class = input_dict[fields.CoLocInputDataFields.groundtruth_target_class]
    for k in xrange(k_shot):
      normalized_gt_boxlist = box_list.BoxList(
          input_dict[fields.CoLocInputDataFields.groundtruth_boxes+'_{}'.format(k)])
      gt_boxlist = box_list_ops.scale(normalized_gt_boxlist,
                                    original_image_shape[1],
                                    original_image_shape[2])
      groundtruth_boxes = gt_boxlist.get()
      groundtruth_classes = input_dict[fields.CoLocInputDataFields.groundtruth_classes+'_{}'.format(k)]
      groundtruth_boxes_list.append(groundtruth_boxes)
      groundtruth_classes_list.append(groundtruth_classes)
    ndict = dict()
    ndict['boxes'] = groundtruth_boxes_list
    ndict['classes'] = groundtruth_classes_list
    ndict['target_class'] = groundtruth_target_class
    tensor_dict['groundtruth'] = ndict
  return tensor_dict, thread


def _postprocess_result_dict(result_dict, compute_upper_bound=False):
  # For now I remove the images and groundtruth data of
  # the negative bags
  #npositive_bags = result_dict['Tree_K1']['boxes'].shape[0]

  gt = result_dict['groundtruth']
  gt_classes = gt['classes']
  gt_boxes   = gt['boxes']
  target_class = gt.pop('target_class')
  agnostic_normalized_gt = normalize_problem_labels(gt_classes, gt_boxes,
                                                    False, target_class)
  multiclass_normalized_gt = normalize_problem_labels(gt_classes, gt_boxes,
                                                      True, target_class)
  if compute_upper_bound and result_dict.has_key('Tree_K1'):
    upper_bound = copy.deepcopy(result_dict['Tree_K1'])
    scores = upper_bound['scores']
    bb0_list = upper_bound['boxes']
    bb1_list = agnostic_normalized_gt['boxes']
    for i in range(len(bb0_list)):
      iou = np_iou(bb0_list[i], bb1_list[i])
      max_iou = iou.max(axis=-1)
      scores[i, ...] = np.array(max_iou == np.max(max_iou), dtype=np.float32)
    result_dict['upper_bound'] = upper_bound
  for key in result_dict:
    if key in ['original_image'] or 'groundtruth' in key:
      continue
    ndict = result_dict[key]

    # Sort base on scores 
    scores = ndict['scores']
    inds = np.argsort(-scores)
    for nkey, nval in ndict.items():
      if nkey is not 'class_agnostic':
        ndict[nkey] = nval[np.arange(scores.shape[0])[:,np.newaxis],
                         inds]
      else:
        ndict[nkey] = [ndict[nkey]]

    # Add groundtruth to the results
    #rpn does not have class_agnostic field
    #so we assume agnostic to be the default
    #normalized_gt = agnostic_normalized_gt
    # check if it is not class_agnostic (we can check by multiclass in key)
    #if ndict.has_key('class_agnostic') and not ndict['class_agnostic'][0]:
    #    normalized_gt = multiclass_normalized_gt
    #ndict['groundtruth'] = normalized_gt #each key has its own gt now

    normalized_gt = multiclass_normalized_gt
    if not ndict.has_key('class_agnostic') or ndict['class_agnostic'][0]:
      ndict['classes'][...] = target_class
    ndict['groundtruth'] = normalized_gt

def evaluate(create_input_dict_fn_list, input_config_names,
             create_model_fn, eval_config, categories,
             checkpoint_dir, eval_dir):
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
      batch_index: the index of the batch amongst all batches in the run.
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
      if 'original_image' in tensor_dict:
        tensor_dict = {k: v for (k, v) in tensor_dict.items()
                       if k != 'original_image'}
    (result_dict, _) = sess.run([tensor_dict, update_op])
    meta_info = result_dict.pop('meta_info', None)
    _postprocess_result_dict(result_dict, compute_upper_bound=True)
    counters['success'] += 1
    global_step = tf.train.global_step(sess, slim.get_global_step())
    if batch_index < eval_config.num_visualizations:
        eval_util.visualize_detection_results(
            result_dict, batch_index, global_step,
            summary_dir=eval_dir,
            export_dir=eval_config.visualization_export_dir,
            show_groundtruth=True,
            max_num_predictions=10,
            min_score_thresh=-10000.0,
            model_name=model_name)
    if 'original_image' in result_dict:
      result_dict.pop('original_image')
    return result_dict

  def _process_aggregated_results(result_lists):
    eval_metric_fn_key = eval_config.metrics_set
    if eval_metric_fn_key not in EVAL_METRICS_FN_DICT:
      raise ValueError('Metric not found: {}'.format(eval_metric_fn_key))
    return EVAL_METRICS_FN_DICT[eval_metric_fn_key](result_lists,
                                                    categories=categories)

  # It seems tf is not releasing the memory after reset.
  # Do it more than 1 tiems and you will get out of memory.
  max_number_of_evaluations = (1 if eval_config.ignore_groundtruth else
                               eval_config.max_evals if eval_config.max_evals else
                               1)

  eval_util.repeated_checkpoint_run(model_names=input_config_names,
      extract_prediction_tensors_fn_list=extract_prediction_tensors_fn_list,
      summary_dir=eval_dir,
      aggregated_result_processor=_process_aggregated_results,
      batch_processor=_process_batch,
      checkpoint_dirs=[checkpoint_dir],
      variables_to_restore=None,
      num_batches=eval_config.num_examples,
      eval_interval_secs=eval_config.eval_interval_secs,
      max_number_of_evaluations=max_number_of_evaluations,
      master=eval_config.eval_master,
      save_graph=eval_config.save_graph,
      save_graph_dir=(eval_dir if eval_config.save_graph else ''),
      use_moving_averages=eval_config.use_moving_averages)
