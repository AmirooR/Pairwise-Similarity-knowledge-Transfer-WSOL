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

"""Post-processing operations on detected boxes."""

import tensorflow as tf

from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import standard_fields as fields
from object_detection.core.post_processing import multiclass_non_max_suppression
import util

def batch_multiclass_non_max_suppression(boxes,
                                         scores,
                                         score_thresh,
                                         iou_thresh,
                                         max_size_per_class,
                                         max_total_size=0,
                                         clip_window=None,
                                         change_coordinate_frame=False,
                                         num_valid_boxes=None,
                                         masks=None,
                                         scope=None,
                                         parallel_iterations=32):
  """Multi-class version of non maximum suppression that operates on a batch.

  This op is similar to `multiclass_non_max_suppression` but operates on a batch
  of boxes and scores and also clip or resample boxes to max_total_size value.
  See documentation for `multiclass_non_max_suppression`
  for details.

  Args:
    boxes: A [batch_size, num_anchors, q, 4] float32 tensor containing
      detections. If `q` is 1 then same boxes are used for all classes
        otherwise, if `q` is equal to number of classes, class-specific boxes
        are used.
    scores: A [batch_size, num_anchors, num_classes] float32 tensor containing
      the scores for each of the `num_anchors` detections.
    score_thresh: scalar threshold for score (low scoring boxes are removed).
    iou_thresh: scalar threshold for IOU (new boxes that have high IOU overlap
      with previously selected boxes are removed).
    max_size_per_class: maximum number of retained boxes per class.
    max_total_size: maximum number of boxes retained over all classes. By
      default returns all boxes retained after capping boxes per class.
    clip_window: A float32 tensor of the form [y_min, x_min, y_max, x_max]
      representing the window to clip boxes to before performing non-max
      suppression.
    change_coordinate_frame: Whether to normalize coordinates after clipping
      relative to clip_window (this can only be set to True if a clip_window
      is provided)
    num_valid_boxes: (optional) a Tensor of type `int32`. A 1-D tensor of shape
      [batch_size] representing the number of valid boxes to be considered
        for each image in the batch.  This parameter allows for ignoring zero
        paddings.
    masks: (optional) a [batch_size, num_anchors, q, mask_height, mask_width]
      float32 tensor containing box masks. `q` can be either number of classes
      or 1 depending on whether a separate mask is predicted per class.
    scope: tf scope name.
    parallel_iterations: (optional) number of batch items to process in
      parallel.

  Returns:
    'nmsed_boxes': A [batch_size, max_detections, 4] float32 tensor
      containing the non-max suppressed boxes.
    'nmsed_scores': A [batch_size, max_detections] float32 tensor containing
      the scores for the boxes.
    'nmsed_classes': A [batch_size, max_detections] float32 tensor
      containing the class for boxes.
    'nmsed_masks': (optional) a
      [batch_size, max_detections, mask_height, mask_width] float32 tensor
      containing masks for each selected box. This is set to None if input
      `masks` is None.
    'num_detections': A [batch_size] int32 tensor indicating the number of
      nmsed boxes per batch item before resampling.
  Raises:
    ValueError: if `q` in boxes.shape is not 1 or not equal to number of
      classes as inferred from scores.shape.
  """
  q = boxes.shape[2].value
  num_classes = scores.shape[2].value
  if q != 1 and q != num_classes:
    raise ValueError('third dimension of boxes must be either 1 or equal '
                     'to the third dimension of scores')

  original_masks = masks
  with tf.name_scope(scope, 'BatchMultiClassNonMaxSuppression'):
    boxes_shape = boxes.shape
    batch_size = boxes_shape[0].value
    num_anchors = boxes_shape[1].value

    if batch_size is None:
      batch_size = tf.shape(boxes)[0]
    if num_anchors is None:
      num_anchors = tf.shape(boxes)[1]

    # If num valid boxes aren't provided, create one and mark all boxes as
    # valid.
    if num_valid_boxes is None:
      num_valid_boxes = tf.ones([batch_size], dtype=tf.int32) * num_anchors

    # If masks aren't provided, create dummy masks so we can only have one copy
    # of single_image_nms_fn and discard the dummy masks after map_fn.
    if masks is None:
      masks_shape = tf.stack([batch_size, num_anchors, 1, 0, 0])
      masks = tf.zeros(masks_shape)

    def single_image_nms_fn(args):
      """Runs NMS on a single image and returns padded output."""
      (per_image_boxes, per_image_scores, per_image_masks,
       per_image_num_valid_boxes) = args
      per_image_boxes = tf.reshape(
          tf.slice(per_image_boxes, 3 * [0],
                   tf.stack([per_image_num_valid_boxes, -1, -1])), [-1, q, 4])
      per_image_scores = tf.reshape(
          tf.slice(per_image_scores, [0, 0],
                   tf.stack([per_image_num_valid_boxes, -1])),
          [-1, num_classes])

      per_image_masks = tf.reshape(
          tf.slice(per_image_masks, 4 * [0],
                   tf.stack([per_image_num_valid_boxes, -1, -1, -1])),
          [-1, q, per_image_masks.shape[2].value,
           per_image_masks.shape[3].value])
      nmsed_boxlist = multiclass_non_max_suppression(
          per_image_boxes,
          per_image_scores,
          score_thresh,
          iou_thresh,
          max_size_per_class,
          max_total_size,
          masks=per_image_masks,
          clip_window=clip_window,
          change_coordinate_frame=change_coordinate_frame)


      num_detections = nmsed_boxlist.num_boxes()
      nmsed_boxes = nmsed_boxlist.get()
      nmsed_scores = nmsed_boxlist.get_field(fields.BoxListFields.scores)
      nmsed_classes = nmsed_boxlist.get_field(fields.BoxListFields.classes)
      nmsed_masks = nmsed_boxlist.get_field(fields.BoxListFields.masks)

      resampling_inds = util.topk_or_pad_inds_with_resampling(
                              tf.ones((num_detections,), dtype=tf.bool),
                              nmsed_scores, max_total_size)

      tensor_list = [nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks]
      tensor_list = [tf.gather(tensor, resampling_inds) for tensor in tensor_list]
      return tensor_list + [num_detections, max_total_size]


    (batch_nmsed_boxes, batch_nmsed_scores,
     batch_nmsed_classes, batch_nmsed_masks,
     batch_num_detections,
     resamped_batch_num_detections) = tf.map_fn(
         single_image_nms_fn,
         elems=[boxes, scores, masks, num_valid_boxes],
         dtype=[tf.float32, tf.float32, tf.float32, tf.float32, tf.int32, tf.int32],
         parallel_iterations=parallel_iterations)
    tf.summary.scalar('FirstStageActualNumOfProposalsAfterNMS',
                      tf.reduce_mean(batch_num_detections))
    if original_masks is None:
      batch_nmsed_masks = None

    return (batch_nmsed_boxes, batch_nmsed_scores, batch_nmsed_classes,
            batch_nmsed_masks, resamped_batch_num_detections)
