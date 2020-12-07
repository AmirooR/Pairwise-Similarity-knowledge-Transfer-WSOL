# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Common functions for repeatedly evaluating a checkpoint.
"""
import copy
from collections import defaultdict
import logging
import os
import time

import numpy as np
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import object_detection_evaluation
from rcnn_attention import visualization_utils as vis_utils
from rcnn_attention.result_list_util import ResultList, get_k_shot

slim = tf.contrib.slim
from threading import Thread
import pickle as pkl
from PIL import Image, ImageDraw, ImageFont
import sys
from functools import reduce
from IPython import embed
from object_detection.utils.np_box_ops import iou as np_iou
from scipy import stats

try:
  from joblib import Parallel, delayed
  import multiprocessing
  joblib_available = True
except:
  joblib_available = False

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
def write_metrics(metrics, global_step, summary_dir):
  """Write metrics to a summary directory.

  Args:
    metrics: A dictionary containing metric names and values.
    global_step: Global step at which the metrics are computed.
    summary_dir: Directory to write tensorflow summaries to.
  """
  logging.info('Writing metrics to tf summary.')
  summary_writer = tf.summary.FileWriter(summary_dir)
  for key in sorted(metrics):
    summary = tf.Summary(value=[
        tf.Summary.Value(tag=key, simple_value=metrics[key]),
    ])
    summary_writer.add_summary(summary, global_step)
    logging.info('%s: %f', key, metrics[key])
  summary_writer.close()
  logging.info('Metrics written to tf summary.')


def load_dataset(root, split):
  logging.info('Loading dataset {}'.format(split))
  pkl_file = os.path.join(root, 'ImageSet', split + '.pkl')
  assert os.path.exists(pkl_file), 'File {} not found'.format(pkl_file)
  with open(pkl_file, 'rb') as f:
    return pkl.load(f)

def check_dataset(ds):
  assert('meta' in ds.keys())
  xy_format, coordinates_format = ds['meta'].split(',')
  assert(xy_format == 'y0x0y1x1')
  assert(coordinates_format in ['relative'])


def save_dataset(root, split, ds):
  logging.info('Saving dataset {}'.format(split))
  pkl_file = os.path.join(root, 'ImageSet', split + '.pkl')
  assert not os.path.exists(pkl_file), 'File {} already exists'.format(pkl_file)
  with open(pkl_file, 'wb') as f:
    pkl.dump(ds, f)

def normalize_problem_labels(classes, boxes, is_multiclass, target_class=None):
  """ classes and boxes are gt classes and boxes. Groundtruth boxes and classes
      for multiclass methods remain the same. For agnostic methods the groundtruth
      gets updated with respect to the target_class.
  """
  if is_multiclass:
    return {'classes':classes, 'boxes': boxes}

  if target_class is None:
    raise NotImplemented('Sorry guzu.')

  def gather(array, gather_indices):
    return map(lambda arr,indices: arr[indices], array, gather_indices)

  is_target = map(np.equal, classes, [target_class]*len(classes))
  classes = map(lambda arr:np.ones_like(arr), classes)

  nclasses = gather(classes, is_target)
  nboxes = gather(boxes, is_target)

  return {'classes':nclasses, 'boxes':nboxes}

def has_box(roi, b, c, single_class_per_image, iou_threshold):
  if len(roi['classes']) == 0:
    return False
  if single_class_per_image and c in roi['classes']:
    return True
  if np_iou(np.array(roi['boxes']), b[None]).max() > iou_threshold:
    return True
  return False

def reduce_results(result_dict, single_class_per_image=False, iou_threshold=0.5):
  for fname, pred in result_dict.items():
    roi = {'classes':[], 'boxes':[], 'scores':[]}
    indices = np.argsort(pred['scores'])[::-1]
    for i in indices:
      cls = pred['classes'][i]
      box = pred['boxes'][i]
      score = pred['scores'][i]
      if not has_box(roi, box, cls, single_class_per_image, iou_threshold):
        roi['classes'].append(cls)
        roi['boxes'].append(box)
        roi['scores'].append(score)
    result_dict[fname] = roi


def process_gholi(g_list):
  results = dict()
  for g in g_list:
    gt, res, filenames = g
    for i, fn in enumerate(filenames):
      cur_res = results.get(fn, defaultdict(list))
      ngt = dict(classes=gt['classes'][i], boxes=gt['boxes'][i])
      cur_res['groundtruth'] = ngt
      cur_res['scores'].extend(res['scores'][i])
      cur_res['boxes'].extend(res['boxes'][i])
      cur_res['classes'].extend(gt['target_class']*res['classes'][i])
      results[fn] = cur_res
  name = 'gholi_pseudo_gt300/'+str(int(time.time())) + '.pkl'
  reduce_results(results, single_class_per_image=False, iou_threshold=0.7)
  with open(name, 'wb') as f:
    pkl.dump(results, f)
  return {}
  result_lists = {'groundtruth': defaultdict(list), 'scores': [], 'boxes': [], 'classes': []}

  for r in results.values():
    for key in result_lists.keys():
      if key == 'groundtruth':
          result_lists[key]['classes'].append(r['groundtruth']['classes'])
          result_lists[key]['boxes'].append(r['groundtruth']['boxes'])
      else:
        result_lists[key].append(np.array(r[key]))

  #creating catgories
  all_classes = [r['classes'] for r in results.values()]
  all_classes = np.unique(map(sum, all_classes))
  categories = []
  for i, cls in enumerate(all_classes):
    categories.append({'id': i, 'name': str(int(cls))})
  categories = [{'id': i, 'name': i} for i in range(201)]
  return evaluate_detection_results_pascal_voc(result_lists, categories)

def process_meta_info(meta_info):
    ret = dict()
    valid_keys = ['time', 'k2_scores_ratio', 'energy']
    if 'gholi' in meta_info.keys():
      ret.update(process_gholi(meta_info['gholi']))
    for k, v in meta_info.items():
      if np.any([vk in k for vk in valid_keys]):
        ret[k] = np.mean(v)
    return ret

def corloc(result_lists, iou_thres=0.5,
           confidence_interval=0.95,
           only_consider_present_classes=True):
    num_exp = len(result_lists['scores'])
    num_classes = int(np.concatenate(result_lists['groundtruth']['classes']).max())
    correct_list = []
    num_corrects = np.zeros(num_classes)
    num_elements = np.zeros(num_classes)

    for i in range(num_exp):
      gt_classes = result_lists['groundtruth']['classes'][i]
      classes = result_lists['classes'][i]
      if only_consider_present_classes:
        target_classes = np.unique(classes)
      else:
        target_classes = np.unique(gt_classes)
      for target_class in target_classes:
        assert(target_class >= 1.0 and target_class <= num_classes), target_class
        cls_box = result_lists['boxes'][i][classes == target_class]
        cls_scores = result_lists['scores'][i][classes == target_class]
        if len(cls_box) > 0:
          top_box = cls_box[np.argmax(cls_scores)]
          inds = gt_classes == target_class
          gt_boxes = result_lists['groundtruth']['boxes'][i][inds]
          max_iou = np_iou(gt_boxes, top_box[None,:]).max()
          success = max_iou >= iou_thres
        else:
          success = False
          #print('No prediction for image id {}, class {}'.format(i, target_class))
        correct_list.append(success)
        target_ind = int(target_class) - 1
        if success:
          num_corrects[target_ind] += 1
        num_elements[target_ind] += 1

    mean = np.mean(correct_list)
    stdev = np.std(correct_list)
    test_stat = stats.t.ppf((confidence_interval + 1)/2, num_exp)
    interval = test_stat * stdev / np.sqrt(num_exp)
    class_corloc = num_corrects/num_elements
    multiclass_corloc = np.nanmean(class_corloc)
    return mean, interval, multiclass_corloc, class_corloc

def evaluate_coloc_results(result_lists,
                           categories,
                           label_id_offset=0,
                           iou_thres=0.5,
                           mAP_summery=True,
                           corloc_summary=True,
                           recall_summary=False,
                           max_proposals=[None]):

  def fn(n):
    res0 = dict()
    #res0 = evaluate_detection_results_pascal_voc(result_lists,
    #                                             categories,
    #                                             label_id_offset,
    #                                             iou_thres,
    #                                             corloc_summary,
    #                                             recall_summary,
    #                                             n)
    mean, interval, multiclass_corloc, per_class_corlocs = corloc(result_lists, iou_thres)
    res0['interval_top_percent_cor'] = interval
    res0['mean_top_percent_cor'] = mean
    res0['multiclass_corloc'] = multiclass_corloc
    for i, cls_corloc in enumerate(per_class_corlocs):
      res0['class %02d corloc'%(i+1)] = cls_corloc
    return res0

  metrics = dict()
  for n in max_proposals:
    new_metrics = fn(n)
    if n is None:
      metrics.update(new_metrics)
    else:
      for key, val in new_metrics.items():
        metrics[key + '/{}Proposals'.format(n)] = val
  return metrics


def evaluate_detection_results_pascal_voc(result_lists,
                                          categories,
                                          label_id_offset=0,
                                          iou_thres=0.5,
                                          corloc_summary=True,
                                          recall_summary=False,
                                          max_proposals=None):
  """Computes Pascal VOC detection metrics given groundtruth and detections.

  This function computes Pascal VOC metrics. This function by default
  takes detections and groundtruth boxes encoded in result_lists and writes
  evaluation results to tf summaries which can be viewed on tensorboard.

  Args:
    result_lists: a dictionary holding lists of groundtruth and detection
      data corresponding to each image being evaluated.  The following keys
      are required:
        'boxes': a list of float32 numpy arrays of shape [N, 4]
        'scores': a list of float32 numpy arrays of shape [N]
        'classes': a list of int32 numpy arrays of shape [N]
        'groundtruth':
          'boxes': a list of float32 numpy arrays of shape [M, 4]
          'classes': a list of int32 numpy arrays of shape [M]
        and the remaining fields below are optional:
          'difficult': a list of boolean arrays of shape [M] indicating the
            difficulty of groundtruth boxes. Some datasets like PASCAL VOC provide
            this information and it is used to remove difficult examples from eval
            in order to not penalize the models on them.
    categories: a list of dictionaries representing all possible categories.
      Each dict in this list has the following keys:
          'id': (required) an integer id uniquely identifying this category
          'name': (required) string representing category name
            e.g., 'cat', 'dog', 'pizza'
    label_id_offset: an integer offset for the label space.
    iou_thres: float determining the IoU threshold at which a box is considered
        correct. Defaults to the standard 0.5.
    corloc_summary: boolean. If True, also outputs CorLoc metrics.

  Returns:
    A dictionary of metric names to scalar values.

  Raises:
    ValueError: if the set of keys in result_lists is not a superset of the
      expected list of keys.  Unexpected keys are ignored.
    ValueError: if the lists in result_lists have inconsistent sizes.
  """
  # check for expected keys in result_lists
  expected_keys = [
      'boxes', 'scores', 'classes', 'groundtruth'
  ]
  if not set(expected_keys).issubset(set(result_lists.keys())):
    raise ValueError('result_lists does not have expected key set.')

  groundtruth_lists = result_lists['groundtruth']
  expected_keys = expected_keys[:-1]
  gt_expected_keys = ['boxes', 'classes']
  if not set(gt_expected_keys).issubset(set(groundtruth_lists.keys())):
    raise ValueError('groundtruth does not have expected key set.')

  num_results = len(result_lists[expected_keys[0]])
  for key in expected_keys:
    if len(result_lists[key]) != num_results:
      raise ValueError('Inconsistent list sizes in result_lists')

  for key in gt_expected_keys:
    if len(groundtruth_lists[key]) != num_results:
      raise ValueError('Inconsistent list sizes in groundtruth_lists')

  if categories is None:
    labs = []
    for i in range(num_results):
      labs.extend(groundtruth_lists['classes'][i])
      labs.extend(result_lists['classes'][i])
    labs = np.unique(np.array(labs, dtype=int))
    categories = [dict(id=i+1, name='{}'.format(i+1)
                  ) for i in range(len(labs))]
  categories = copy.deepcopy(categories)
  for idx in range(len(categories)):
    categories[idx]['id'] -= label_id_offset

  # num_classes (maybe encoded as categories)
  num_classes = max([cat['id'] for cat in categories]) + 1
  logging.info('Computing Pascal VOC metrics on results.')
  image_ids = range(num_results)

  evaluator = object_detection_evaluation.ObjectDetectionEvaluation(
      num_classes, matching_iou_threshold=iou_thres)

  difficult_lists = None
  if 'difficult' in result_lists and result_lists['difficult']:
    difficult_lists = result_lists['difficult']
  for idx, image_id in enumerate(image_ids):
    difficult = None
    if difficult_lists is not None and difficult_lists[idx].size:
      difficult = difficult_lists[idx].astype(np.bool)
      difficult = difficult[:max_proposals]

    evaluator.add_single_ground_truth_image_info(
        image_id, groundtruth_lists['boxes'][idx],
        groundtruth_lists['classes'][idx] - label_id_offset,
        difficult)
    evaluator.add_single_detected_image_info(
        image_id, result_lists['boxes'][idx][:max_proposals],
        result_lists['scores'][idx][:max_proposals],
        result_lists['classes'][idx][:max_proposals] - label_id_offset)
  per_class_ap, mean_ap, _, per_class_recall, per_class_corloc, mean_corloc = (
      evaluator.evaluate())
  metrics = {'Precision/mAP@{}IOU'.format(iou_thres): mean_ap}
  category_index = label_map_util.create_category_index(categories)
  #for idx in range(per_class_ap.size):
  #  if idx in category_index:
  #    display_name = ('PerformanceByCategory/mAP@{}IOU/{}'
  #                    .format(iou_thres, category_index[idx]['name']))
  #    metrics[display_name] = per_class_ap[idx]

  if corloc_summary:
    metrics['corloc/corloc@{}iou'.format(iou_thres)] = mean_corloc
  #  for idx in range(per_class_corloc.size):
  #    if idx in category_index:
  #      display_name = (
  #          'performancebycategory/corloc@{}iou/{}'.format(
  #              iou_thres, category_index[idx]['name']))
  #      metrics[display_name] = per_class_corloc[idx]

  if recall_summary:
    raise Exception('Not working yet. per_class_recall is not in the right format')
    metrics['recall/recall@{}iou'.format(iou_thres)] = np.nanmean(per_class_recall)
  #  for idx in range(per_class_recall.size):
  #    if idx in category_index:
  #      display_name = (
  #          'performancebycategory/recall@{}iou/{}'.format(
  #              iou_thres, category_index[idx]['name']))
  #      metrics[display_name] = per_class_recall[idx]

  return metrics


# TODO: Add tests.
def visualize_detection_results(result_dict,
                                batch_index,
                                global_step,
                                summary_dir='',
                                export_dir='',
                                agnostic_mode=False,
                                show_groundtruth=False,
                                min_score_thresh=.5,
                                max_num_predictions=3,
                                model_name=None):
  """Visualizes detection results and writes visualizations to image summaries.

  This function visualizes an image with its detected bounding boxes and writes
  to image summaries which can be viewed on tensorboard.  It optionally also
  writes images to a directory. In the case of missing entry in the label map,
  unknown class name in the visualization is shown as "N/A".

  Args:
    result_dict: a dictionary holding groundtruth and detection
      data corresponding to each image being evaluated.  The following keys
      are required:
        'original_image': a numpy array representing the image with shape
          [1, height, width, 3]
        'detection_boxes': a numpy array of shape [N, 4]
        'detection_scores': a numpy array of shape [N]
        'detection_classes': a numpy array of shape [N]
      The following keys are optional:
        'groundtruth_boxes': a numpy array of shape [N, 4]
        'groundtruth_keypoints': a numpy array of shape [N, num_keypoints, 2]
      Detections are assumed to be provided in decreasing order of score and for
      display, and we assume that scores are probabilities between 0 and 1.
    tag: tensorboard tag (string) to associate with image.
    global_step: global step at which the visualization are generated.
    categories: a list of dictionaries representing all possible categories.
      Each dict in this list has the following keys:
          'id': (required) an integer id uniquely identifying this category
          'name': (required) string representing category name
            e.g., 'cat', 'dog', 'pizza'
          'supercategory': (optional) string representing the supercategory
            e.g., 'animal', 'vehicle', 'food', etc
    summary_dir: the output directory to which the image summaries are written.
    export_dir: the output directory to which images are written.  If this is
      empty (default), then images are not exported.
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.
    show_groundtruth: boolean (default: False) controlling whether to show
      groundtruth boxes in addition to detected boxes
    min_score_thresh: minimum score threshold for a box to be visualized
    max_num_predictions: maximum number of detections to visualize
  Raises:
    ValueError: if result_dict does not contain the expected keys (i.e.,
      'original_image', 'detection_boxes', 'detection_scores',
      'detection_classes')
  """
  if not result_dict:
    return
  if not set([
      'original_image']
      ).issubset(set(result_dict.keys())):
    raise ValueError('result_dict does not contain all expected keys.')
  logging.info('Creating detection visualizations.')

  result_dict = dict(result_dict)
  result_dict.pop('groundtruth', None)
  def _tile_images(images, tag=None):
    nimages, h, w = images.shape[:3]
    l = int(np.ceil(np.sqrt(nimages)))
    m = int(np.ceil(nimages/float(l)))
    image = np.zeros((h*m, w*l, 3), dtype=images.dtype)
    for i in range(nimages):
      row = int(i/l)*h
      col = int(i%l)*w
      image[row:(row+h), col:(col+w), ...] = images[i]
    if tag:
      # get a drawing context
      image = np.array(image, dtype=np.uint8)
      pil_img = Image.fromarray(image)
      d = ImageDraw.Draw(pil_img)
      fnt = ImageFont.load_default()
      d.text((10,10), tag, fill=(0, 0, 0))
      image = np.array(pil_img)
    return image

  def _plot_boxes(images,
                  res,
                  max_num_predictions=None,
                  inplace=False,
                  show_groundtruth=True,
                  tag_boxes=True,
                  color_offset=0,
                  default_tag=None):

    nprops = res['scores'][0].shape[0]
    detection_scores = res['scores']
    detection_boxes = res['boxes']
    if res.has_key('groundtruth'):
      groundtruth = res['groundtruth']
    else:
      show_groundtruth = False

    if 'training_classes' in res:
      detection_classes = res['training_classes']
    else:
      detection_classes = res['classes']

    detection_keypoints = None
    detection_masks = None
    if not inplace:
      images = np.copy(images)
    for i in range(detection_classes.shape[0]):
      # Plot groundtruth underneath detections
      if show_groundtruth:
        groundtruth_boxes = groundtruth['boxes'][i]
        groundtruth_keypoints = groundtruth.get('keypoints', None)
        if groundtruth_keypoints is not None:
          groundtruth_keypoints = groundtruth_keypoints[i]

        vis_utils.visualize_boxes_and_labels_on_image_array(
            images[i],
            groundtruth_boxes,
            None,
            None,
            None,
            keypoints=groundtruth_keypoints,
            use_normalized_coordinates=False,
            max_boxes_to_draw=None)
      top_scores_ind = np.argsort(detection_scores[i])[::-1]

      # Assign different classes to get different colors
      classes = np.zeros(detection_classes[i].shape, dtype=np.int32)
      classes[top_scores_ind] = np.arange(1, len(classes)+1) + color_offset

      if tag_boxes:
        category_index = dict()
        for j, ind in enumerate(top_scores_ind):
          id = j+1+color_offset
          cls = int(detection_classes[i, ind])
          if default_tag:
            category_index[id] = dict(id=id, name=default_tag)
          else:
            category_index[id] = dict(id=id, name='i{},c{}'.format(id, cls))
      else:
        category_index = None

      vis_utils.visualize_boxes_and_labels_on_image_array(
          images[i],
          detection_boxes[i],
          classes,
          detection_scores[i],
          category_index,
          instance_masks= detection_masks[i] if detection_masks else None,
          keypoints=detection_keypoints[i] if detection_keypoints else None,
          use_normalized_coordinates=False,
          max_boxes_to_draw=max_num_predictions,
          min_score_thresh=min_score_thresh,
          agnostic_mode=agnostic_mode)

    return images

  images = result_dict.pop('original_image')
  tag = '{}'.format(batch_index)
  if model_name:
    tag = model_name + '_' + tag

  # Write Tree images
  depth = int(np.log2(images.shape[0]))
  imgs = []
  top_pred_images = np.copy(images)

  for i, (method, res) in enumerate(result_dict.items()):
    imgs_boxes = _plot_boxes(images,
                             res,
                             max_num_predictions=max_num_predictions,
                             show_groundtruth=show_groundtruth)

    imgs.append(_tile_images(imgs_boxes,
                             tag=method.upper()))

    ## Draw top boxes on top of each other
    _plot_boxes(top_pred_images,
                res,
                max_num_predictions=1,
                inplace=True,
                show_groundtruth=show_groundtruth if i == 0 else False,
                color_offset=i,
                tag_boxes=True,
                default_tag=method)



  top_pred_images = _tile_images(top_pred_images)
  image = np.concatenate((top_pred_images,) + tuple(imgs))

  if export_dir:
    export_path = os.path.join(export_dir, 'export-{}.jpg'.format(tag))
    vis_utils.save_image_array(image, export_path, 'JPEG')
  else:
    summary = tf.Summary(value=[
          tf.Summary.Value(tag=tag, image=tf.Summary.Image(
              encoded_image_string=vis_utils.encode_image_array_as_png_str(
                  image)))
    ])
    summary_writer = tf.summary.FileWriter(summary_dir)
    summary_writer.add_summary(summary, global_step)
    summary_writer.close()

  logging.info('Detection visualizations written to summary with tag %s.', tag)

# TODO: Add tests.
# TODO: Have an argument called `aggregated_processor_tensor_keys` that contains
# a whitelist of tensors used by the `aggregated_result_processor` instead of a
# blacklist. This will prevent us from inadvertently adding any evaluated
# tensors into the `results_list` data structure that are not needed by
# `aggregated_result_preprocessor`.
def run_checkpoint_once(tensor_dict,
                        update_op,
                        summary_dir,
                        aggregated_result_processor=None,
                        batch_processor=None,
                        checkpoint_dirs=None,
                        variables_to_restore=None,
                        restore_fn=None,
                        num_batches=1,
                        master='',
                        save_graph=False,
                        save_graph_dir='',
                        metric_names_to_values=None,
                        keys_to_exclude_from_results=(),
                        enqueue_thread=None,
                        model_name=None,
                        aggregation_params=None,
                        override_elems=False):
  """Evaluates both python metrics and tensorflow slim metrics.

  Python metrics are processed in batch by the aggregated_result_processor,
  while tensorflow slim metrics statistics are computed by running
  metric_names_to_updates tensors and aggregated using metric_names_to_values
  tensor.

  Args:
    tensor_dict: a dictionary holding tensors representing a batch of detections
      and corresponding groundtruth annotations.
    update_op: a tensorflow update op that will run for each batch along with
      the tensors in tensor_dict..
    summary_dir: a directory to write metrics summaries.
    aggregated_result_processor: a function taking one arguments:
      1. result_lists: a dictionary with keys matching those in tensor_dict
        and corresponding values being the list of results for each tensor
        in tensor_dict.  The length of each such list is num_batches.
    batch_processor: a function taking four arguments:
      1. tensor_dict: the same tensor_dict that is passed in as the first
        argument to this function.
      2. sess: a tensorflow session
      3. batch_index: an integer representing the index of the batch amongst
        all batches
      4. update_op: a tensorflow update op that will run for each batch.
      and returns result_dict, a dictionary of results for that batch.
      By default, batch_processor is None, which defaults to running:
        return sess.run(tensor_dict)
      To skip an image, it suffices to return an empty dictionary in place of
      result_dict.
    checkpoint_dirs: list of directories to load into an EnsembleModel. If it
      has only one directory, EnsembleModel will not be used -- a DetectionModel
      will be instantiated directly. Not used if restore_fn is set.
    variables_to_restore: None, or a dictionary mapping variable names found in
      a checkpoint to model variables. The dictionary would normally be
      generated by creating a tf.train.ExponentialMovingAverage object and
      calling its variables_to_restore() method. Not used if restore_fn is set.
    restore_fn: None, or a function that takes a tf.Session object and correctly
      restores all necessary variables from the correct checkpoint file. If
      None, attempts to restore from the first directory in checkpoint_dirs.
    num_batches: the number of batches to use for evaluation.
    master: the location of the Tensorflow session.
    save_graph: whether or not the Tensorflow graph is stored as a pbtxt file.
    save_graph_dir: where to store the Tensorflow graph on disk. If save_graph
      is True this must be non-empty.
    metric_names_to_values: A dictionary containing metric names to tensors
      which will be evaluated after processing all batches
      of [tensor_dict, update_op]. If any metrics depend on statistics computed
      during each batch ensure that `update_op` tensor has a control dependency
      on the update ops that compute the statistics.
    keys_to_exclude_from_results: keys in tensor_dict that will be excluded
      from results_list. Note that the tensors corresponding to these keys will
      still be evaluated for each batch, but won't be added to results_list.

  Raises:
    ValueError: if restore_fn is None and checkpoint_dirs doesn't have at least
      one element.
    ValueError: if save_graph is True and save_graph_dir is not defined.
  """
  if save_graph and not save_graph_dir:
    raise ValueError('`save_graph_dir` must be defined.')

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
  config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
  sess = tf.Session(master, graph=tf.get_default_graph(), config=config)
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  if restore_fn:
    restore_fn(sess)
  else:
    if not checkpoint_dirs:
      raise ValueError('`checkpoint_dirs` must have at least one entry.')
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dirs[0])
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, checkpoint_file)

  if False:
    unit0_vars = [v for v in tf.global_variables() if 'Unit0' in v.name]
    unit1_vars = [v for v in tf.global_variables() if 'Unit1' in v.name]
    sess.run(tf.group(*[unit0_vars[i].assign(unit1_vars[i]
                    ) for i in range(len(unit0_vars)-1)]))
  if save_graph:
    tf.train.write_graph(sess.graph_def, save_graph_dir, 'eval.pbtxt')

  is_aggregate = aggregation_params is not None and aggregation_params.aggregate
  result_lists = ResultList(remove_duplicates=is_aggregate, override=override_elems)
  counters = {'skipped': 0, 'success': 0}
  other_metrics = None
  if is_aggregate:
    gt_ds = load_dataset(aggregation_params.dataset_root,
                         aggregation_params.groundtruth_split)
    check_dataset(gt_ds)
    assert(aggregation_params.update_split is not None), 'update_split should not be None'
    update_ds = load_dataset(aggregation_params.dataset_root,
                             aggregation_params.update_split)
    check_dataset(update_ds)

  def _save_pseudo_ds(name=None, partial=True):
    pseudo_ds, stats = result_lists.update_pseudo_dataset(update_ds)
    metrics = result_lists.evaluate_created_dataset(pseudo_ds, gt_ds,
                              #evaluate_detection_results_pascal_voc)
                              evaluate_coloc_results,
                              partial=partial)
    metrics.update(stats)
    for key in sorted(metrics):
      logging.info('{}: {}'.format(key, metrics[key]))
    if name:
      save_dataset(aggregation_params.dataset_root,
                   name, pseudo_ds)


  with sess.as_default():
      enqueue_thread.start()
  try:
    with tf.contrib.slim.queues.QueueRunners(sess):
      for batch in range(int(num_batches)):
        if (batch + 1) % 100 == 0:
          logging.info('Running eval ops batch %d/%d',
                               batch + 1, num_batches)
        result_dict = batch_processor(
                       tensor_dict, sess, batch,
                       counters, update_op, model_name)
        if result_dict is None:
          continue
        result_lists.add_results(result_dict)
        if is_aggregate and (batch+1) % aggregation_params.save_eval_freq == 0:
          logging.info('Evaluating dataset at iter {}'.format(batch + 1))
          save_name = None
          if override_elems: #bcd saves the dataset in order to compute energy
            save_name=aggregation_params.save_split + '_epoch_{:06d}'.format((batch+1)/aggregation_params.save_eval_freq)
          _save_pseudo_ds(name=save_name, partial=True)
      if metric_names_to_values is not None:
        other_metrics = sess.run(metric_names_to_values)
      logging.info('Running eval batches done.')

    #result_lists._merged_inds = range(len(result_lists._result_lists))
    #result_lists.apply_nms(max_proposal=1)
    #from IPython import embed;embed()
    if is_aggregate:
      logging.info('Creating final dataset')
      _save_pseudo_ds(aggregation_params.save_split, partial=False)

    result_lists = result_lists.get_packed_results()
    metrics = dict()
    k_shot = get_k_shot(result_lists)

    meta_info = result_lists.pop('meta_info', dict())
    metrics.update(process_meta_info(meta_info))

    methods, results = [], []
    for method, result in result_lists.items():
      if method not in ['original_image'] and 'groundtruth' not in method:# and not 'Tree' in method:
        results.append(result)
        methods.append(method)
    #if joblib_available:
    #  num_cores = multiprocessing.cpu_count()
    #  new_metrics_list = Parallel(n_jobs=num_cores)(
    #              delayed(aggregated_result_processor)(r) for r in results)
    #else:
    #  new_metrics_list = map(aggregated_result_processor, results)

    new_metrics_list = []
    for i,r in enumerate(results):
      #print('METHOD: {}'.format(methods[i]))
      new_metrics_list.append(aggregated_result_processor(r))
    for method, new_metrics in zip(methods, new_metrics_list):
     if other_metrics is not None:
        new_metrics.update(other_metrics)
     for metric_keys in new_metrics:
        if 'category' in metric_keys:
          continue
        metrics[method+'/'+metric_keys] = new_metrics[metric_keys]
    global_step = tf.train.global_step(sess, slim.get_global_step())
    if model_name:
      metrics = {model_name + '/' + key: val
                 for key, val in metrics.items()}

    write_metrics(metrics, global_step, summary_dir)
    logging.info('# success: %d', counters['success'])
    logging.info('# skipped: %d', counters['skipped'])
    return metrics
  finally:
    enqueue_thread._Thread__stop()
    Thread.join(enqueue_thread)
    sess.close()

# TODO: Add tests.
def repeated_checkpoint_run(model_names,
                            aggregation_params,
                            extract_prediction_tensors_fn_list,
                            summary_dir,
                            override_elems=False,
                            aggregated_result_processor=None,
                            batch_processor=None,
                            checkpoint_dirs=None,
                            variables_to_restore=None,
                            num_batches=1,
                            eval_interval_secs=120,
                            max_number_of_evaluations=None,
                            master='',
                            save_graph=False,
                            save_graph_dir='',
                            metric_names_to_values=None,
                            keys_to_exclude_from_results=(),
                            use_moving_averages=False,
                            load_from_vanilla_into_doubled_net = True,
                            transfered_network_checkpoint=''):
  """Periodically evaluates desired tensors using checkpoint_dirs or restore_fn.

  This function repeatedly loads a checkpoint and evaluates a desired
  set of tensors (provided by tensor_dict) and hands the resulting numpy
  arrays to a function result_processor which can be used to further
  process/save/visualize the results.

  Args:
    tensor_dict: a dictionary holding tensors representing a batch of detections
      and corresponding groundtruth annotations.
    update_op: a tensorflow update op that will run for each batch along with
      the tensors in tensor_dict.
    summary_dir: a directory to write metrics summaries.
    aggregated_result_processor: a function taking one argument:
      1. result_lists: a dictionary with keys matching those in tensor_dict
        and corresponding values being the list of results for each tensor
        in tensor_dict.  The length of each such list is num_batches.
    batch_processor: a function taking three arguments:
      1. tensor_dict: the same tensor_dict that is passed in as the first
        argument to this function.
      2. sess: a tensorflow session
      3. batch_index: an integer representing the index of the batch amongst
        all batches
      4. update_op: a tensorflow update op that will run for each batch.
      and returns result_dict, a dictionary of results for that batch.
      By default, batch_processor is None, which defaults to running:
        return sess.run(tensor_dict)
    checkpoint_dirs: list of directories to load into a DetectionModel or an
      EnsembleModel if restore_fn isn't set. Also used to determine when to run
      next evaluation. Must have at least one element.
    variables_to_restore: None, or a dictionary mapping variable names found in
      a checkpoint to model variables. The dictionary would normally be
      generated by creating a tf.train.ExponentialMovingAverage object and
      calling its variables_to_restore() method. Not used if restore_fn is set.
    restore_fn: a function that takes a tf.Session object and correctly restores
      all necessary variables from the correct checkpoint file.
    num_batches: the number of batches to use for evaluation.
    eval_interval_secs: the number of seconds between each evaluation run.
    max_number_of_evaluations: the max number of iterations of the evaluation.
      If the value is left as None the evaluation continues indefinitely.
    master: the location of the Tensorflow session.
    save_graph: whether or not the Tensorflow graph is saved as a pbtxt file.
    save_graph_dir: where to save on disk the Tensorflow graph. If store_graph
      is True this must be non-empty.
    metric_names_to_values: A dictionary containing metric names to tensors
      which will be evaluated after processing all batches
      of [tensor_dict, update_op]. If any metrics depend on statistics computed
      during each batch ensure that `update_op` tensor has a control dependency
      on the update ops that compute the statistics.
    keys_to_exclude_from_results: keys in tensor_dict that will be excluded
      from results_list. Note that the tensors corresponding to these keys will
      still be evaluated for each batch, but won't be added to results_list.

  Raises:
    ValueError: if max_num_of_evaluations is not None or a positive number.
    ValueError: if checkpoint_dirs doesn't have at least one element.
  """
  if max_number_of_evaluations and max_number_of_evaluations <= 0:
    raise ValueError(
        '`number_of_steps` must be either None or a positive number.')

  if not checkpoint_dirs:
    raise ValueError('`checkpoint_dirs` must have at least one entry.')

  number_of_evaluations = 0
  while True:
    start = time.time()
    logging.info('Starting evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                           time.gmtime()))
    model_path = tf.train.latest_checkpoint(checkpoint_dirs[0])
    if not model_path:
      logging.info('No model found in %s. Will try again in %d seconds',
                   checkpoint_dirs[0], eval_interval_secs)
    else:
      backup_dir = os.path.join(checkpoint_dirs[0], 'top')
      tf.gfile.MkDir(backup_dir)
      backup_prefix = os.path.join(backup_dir, os.path.split(model_path)[1])
      logging.info("model_path: {}".format(model_path))
      logging.info("backup_prefix: {}".format(backup_prefix))
      backup_model(model_path, backup_prefix)
      model_path = backup_prefix
      for model_name, extract_prediction_tensors_fn in zip(model_names,
          extract_prediction_tensors_fn_list):
        with tf.Graph().as_default():
          tensor_dict, enqueue_thread = extract_prediction_tensors_fn()
          update_op = tf.no_op()

          def _vanilla_to_doubled(var_name, unit):
            assert(unit == 'A' or unit == 'B')
            unit = '/' + unit
            s = 'attention_tree%s/Preprocess'
            var_name = var_name.replace(s % unit, s % '', 1)
            s = 'attention_tree/Unit1/cross_similarity/similarity_matrix%s'
            var_name = var_name.replace(s % unit, s % '', 1)
            return var_name

          variables_to_restore = tf.global_variables()
          global_step = slim.get_or_create_global_step()
          variables_to_restore.append(global_step)
          if use_moving_averages:
            variable_averages = tf.train.ExponentialMovingAverage(0.0)
            variables_to_restore = variable_averages.variables_to_restore()

          found_in_checkpoint = []

          def _get_saver(model_path, unit=None, load_from_vanilla_into_doubled_net=False):
            reader = tf.pywrap_tensorflow.NewCheckpointReader(model_path)
            checkpoint_var_shape = reader.get_variable_to_shape_map()
            available_var_map = {}
            for var in variables_to_restore:
              var_name = var.name[:-2]
              if load_from_vanilla_into_doubled_net:
                var_name = _vanilla_to_doubled(var_name, unit)
              if var_name in checkpoint_var_shape:
                if checkpoint_var_shape[var_name] == var.shape.as_list():
                  available_var_map[var_name] = var
                  found_in_checkpoint.append(var)

            return tf.train.Saver(available_var_map)


          saver = _get_saver(model_path, 'A', load_from_vanilla_into_doubled_net)

          # Initialize the transfered network
          transfered_saver = None
          if transfered_network_checkpoint:
            transfered_saver = _get_saver(transfered_network_checkpoint, 'B', True)

          # Print variables that don't get intialized.
          for var in variables_to_restore:
            if var not in found_in_checkpoint:
              logging.info('%s not found in the checkpoint.', var.name)

          def _restore_latest_checkpoint(sess):
            if transfered_saver is not None:
              transfered_saver.restore(sess, transfered_network_checkpoint)
            saver.restore(sess, model_path)

          metrics = run_checkpoint_once(tensor_dict, update_op, summary_dir,
                                        aggregated_result_processor,
                                        batch_processor, checkpoint_dirs,
                                        variables_to_restore, _restore_latest_checkpoint, num_batches, master,
                                        save_graph, save_graph_dir, metric_names_to_values,
                                        keys_to_exclude_from_results, enqueue_thread=enqueue_thread,
                                        model_name=model_name,
                                        aggregation_params=aggregation_params,
                                        override_elems=override_elems)
          update_best_model(metrics, backup_prefix, backup_dir, keep_topk=5)

    number_of_evaluations += 1

    if (max_number_of_evaluations and
        number_of_evaluations >= max_number_of_evaluations):
      logging.info('Finished evaluation!')
      break
    time_to_next_eval = start + eval_interval_secs - time.time()
    if time_to_next_eval > 0:
      time.sleep(time_to_next_eval)


def backup_model(src_prefix, dest_prefix):
  src_files = tf.gfile.Glob(src_prefix + '*')
  assert(all([x[:len(src_prefix)] == src_prefix]) for x in src_files)
  dest_files = [dest_prefix + src[len(src_prefix):] for src in src_files]
  [tf.gfile.Remove(dest) for dest in dest_files if tf.gfile.Exists(dest)]
  [tf.gfile.Copy(src, dest) for src, dest in zip(src_files, dest_files)]

def update_best_model(matrics, model_prefix, backup_dir, keep_topk=1):
  #target = 'os_test/Tree_K{}/corloc/corloc@0.5iou'
  #target = 'val_unseen/Tree_K{}/corloc/corloc@0.5iou'
  target = 'train_unseen/Tree_K{}_nmsed/Precision/mAP@0.5IOU'

  k = 1
  while True:
    if target.format(k) not in matrics:
      break
    k *= 2

  if k == 1:
    return
  k /= 2

  val = matrics[target.format(k)]
  vals = [0.0]*keep_topk
  model_paths = [None]*keep_topk

  table_fn = os.path.join(backup_dir, 'table.pkl')
  # Read current table
  if tf.gfile.Exists(table_fn):
    with open(table_fn, 'rb') as f:
      table = pkl.load(f)
      limit = min(keep_topk, len(table['val']))
      vals[:limit] = table['val'][:limit]
      model_paths[:limit] = table['path'][:limit]

  def _normpath(path_list):
    return [os.path.normpath(p) for p in path_list]

  if os.path.normpath(model_prefix) in _normpath([x for x in model_paths if x]):
    return

  # Update the table
  comp = [val > v for v in vals]
  if any(comp):
    ind = np.where(comp)[0][0]
    vals[ind+1:] = vals[ind:-1]
    model_paths[ind+1:] = model_paths[ind:-1]
    vals[ind] = val
    model_paths[ind] = model_prefix
  # Write the updated table
  with open(table_fn, 'wb') as f:
    pkl.dump(dict(val=vals, path=model_paths), f)

  # Remove unreferenced model paths
  keep_files = [table_fn]
  for prefix in model_paths:
    if prefix:
      keep_files.extend(tf.gfile.Glob(prefix + '*'))

  all_files = [os.path.join(backup_dir, x)
                for x in tf.gfile.ListDirectory(backup_dir)]

  all_files, keep_files = _normpath(all_files), _normpath(keep_files)
  for fn in all_files:
    if fn not in keep_files:
      tf.gfile.Remove(fn)

