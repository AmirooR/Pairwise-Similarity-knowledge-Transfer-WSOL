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
from object_detection.utils import np_box_ops
import os, sys
import cv2
import cPickle as pickle
from hashlib import sha1

slim = tf.contrib.slim

def mkdir_if_not_exists(*dir_names):
  for dir_name in dir_names:
    if not os.path.exists(dir_name):
      os.makedirs(dir_name)

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
                          'classes': detections['rpn_detection_classes']}

  if detections.has_key('detection_boxes'):
    tensor_dict['detection'] = {'boxes': detections['detection_boxes'],
                                'scores': detections['detection_scores'],
                                'classes': detections['detection_classes'],
                                'feas': detections['detection_feas']}
  label_id_offset = 1
  if hasattr(model, '_tree_debug_tensors'):
    tensor_dict.update(model._tree_debug_tensors(include_feas=True))

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


def extract(create_input_dict_fn_list, input_config_names,
             create_model_fn, eval_config, categories,
             checkpoint_dir, eval_dir):
  """
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

    (result_dict, _) = sess.run([tensor_dict, update_op])
    key = 'detection' #'Tree_K1' use detection for box predictor
    boxes = result_dict[key]['boxes'][0]
    gt_boxes = result_dict['groundtruth']['boxes'][0]
    gt_classes = result_dict['groundtruth']['classes'][0]

    iou = np_box_ops.iou(boxes, gt_boxes)
    box_classes = gt_classes[iou.argmax(axis=1)]
    box_classes[iou.max(axis=1) < 0.5] = 0
    fea = result_dict[key]['feas'][0][...,:640]
    img = result_dict['original_image'][0]
    mkdir_if_not_exists(os.path.join(eval_dir, 'Images', model_name),
                        os.path.join(eval_dir, 'ImageSet'),
                        os.path.join(eval_dir, 'Feas', model_name))
    name = '{:08d}'.format(batch_index)
    image_path = os.path.join(eval_dir, 'Images', model_name, name+'.jpg')
    fea_path   = os.path.join(eval_dir, 'Feas', model_name, name+'.npy')
    info_path  = os.path.join(eval_dir, 'Feas', model_name, name+'.pkl')
    img_info = {'gt_boxes': gt_boxes, 'gt_classes': gt_classes,
                'fea_boxes': boxes, 'fea_classes': box_classes,
                'name': name}
    #TODO skipping this since I will link it
    cv2.imwrite(image_path, img[:,:,::-1]) #H,W,3
    np.save(fea_path, fea) #300,1,1,640
    with open(info_path, 'wb') as f:
      pickle.dump(img_info, f) #types are a bit different from the original
    ##
    counters['success'] += 1
    global_step = tf.train.global_step(sess, slim.get_global_step())
    return dict(hash=[sha1(img).hexdigest()])

  def aggregated_result_processor(hash_values):
    return {'DB_len': len(hash_values), 'DB_unique_elems': len(set(hash_values))}

  eval_util.repeated_checkpoint_run(model_names=input_config_names,
      extract_prediction_tensors_fn_list=extract_prediction_tensors_fn_list,
      summary_dir=eval_dir,
      aggregated_result_processor=aggregated_result_processor,
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
