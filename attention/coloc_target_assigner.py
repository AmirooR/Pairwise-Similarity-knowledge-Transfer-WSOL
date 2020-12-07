import tensorflow as tf

from object_detection.box_coders import faster_rcnn_box_coder
from object_detection.box_coders import mean_stddev_box_coder
from object_detection.core import box_coder as bcoder
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import matcher as mat
from object_detection.core import region_similarity_calculator as sim_calc
from object_detection.matchers import argmax_matcher
from object_detection.matchers import bipartite_matcher
from object_detection.core import matcher
from object_detection.core import target_assigner as target_assign

class KShotMatcher(matcher.Matcher):
  def __init__(self, similarity_calc, matcher, k_shot,
                independent_matching=False):
    self._similarity_calc = similarity_calc
    self._matcher = matcher
    self._k_shot = k_shot
    self._independent_matching = independent_matching

  def _join_match_list(self, match_list):
    unmatched = reduce(tf.logical_or, [tf.equal(m, -1) for m in match_list])
    ignored = reduce(tf.logical_or, [tf.equal(m, -2) for m in match_list])

    all_unmatched = tf.fill(tf.shape(unmatched), -1)
    all_ignored = tf.fill(tf.shape(unmatched), -2)

    # If an item is not unmatched or ignored in any of match list elements
    # then it is a match
    joined_match_list = []
    for match in match_list:
      joined_match = tf.where(ignored, all_ignored,
                                    tf.where(unmatched,
                                      all_unmatched, match))
      joined_match_list.append(joined_match)


    #joined_match_list[0] = tf.Print(joined_match_list[0],
    #    [tf.reduce_max(joined_match) for joined_match in joined_match_list]+
    #    [tf.reduce_max(match) for match in match_list])
    return joined_match_list

  def match_k_shot(self, anchors_batch, gt_box_batch,
                  gt_class_targets_batch, **params):

    # Will be used for testing
    if self._independent_matching:
      k_shot = 1
    else:
      k_shot = self._k_shot
    independent_match_list = []
    # First do the matching for each image independently
    for anchors, gt_boxes, gt_class_targets in zip(
        anchors_batch, gt_box_batch, gt_class_targets_batch):
      match_quality_matrix = self._similarity_calc.compare(gt_boxes,
                                                           anchors)
      match = self._matcher.match(match_quality_matrix, **params)
      independent_match_list.append(match._match_results)

    # Then "and" the match results of k_shot images
    self._match_list = []
    batch_size = len(anchors_batch)
    for i in range(0, batch_size, k_shot):
      k_shot_match_list = independent_match_list[i:i+k_shot]
      self._match_list.extend(self._join_match_list(k_shot_match_list))

  def _match(self, similarity_matrix, **params):
    return self._match_list[params['batch_ind']]

def create_coloc_target_assigner(reference, stage=None,
                                 positive_class_weight=1.0,
                                 negative_class_weight=1.0,
                                 unmatched_cls_target=None,
                                 k_shot=1,
                                 independent_matching=False):
  """Factory function for creating coloc target assigners.

  Args:
    reference: string referencing the type of TargetAssigner.
    stage: string denoting stage: {proposal, detection}.
    positive_class_weight: classification weight to be associated to positive
      anchors (default: 1.0)
    negative_class_weight: classification weight to be associated to negative
      anchors (default: 1.0)
    unmatched_cls_target: a float32 tensor with shape [d_1, d_2, ..., d_k]
      which is consistent with the classification target for each
      anchor (and can be empty for scalar targets).  This shape must thus be
      compatible with the groundtruth labels that are passed to the Assign
      function (which have shape [num_gt_boxes, d_1, d_2, ..., d_k]).
      If set to None, unmatched_cls_target is set to be 0 for each anchor.

  Returns:
    TargetAssigner: desired target assigner.

  Raises:
    ValueError: if combination reference+stage is invalid.
  """
  if reference == 'RCNNAttention' and stage == 'coloc':
    k_shot_similarity_calc = sim_calc.IouSimilarity()

    # Uses all proposals with IOU < 0.5 as candidate negatives.
    k_shot_matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.5,
                                    negatives_lower_than_unmatched=True)
    matcher = KShotMatcher(k_shot_similarity_calc, k_shot_matcher,
                            k_shot, independent_matching)

    # We do not use the similarity calculator inside the target
    # assigner. Matches will be decided before calling the 
    # target_assigner.
    class _DummySimilarity(sim_calc.RegionSimilarityCalculator):
      def _compare(self, boxlist1, boxlist2):
        return None

    similarity_calc = _DummySimilarity()

    box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder(
        scale_factors=[10.0, 10.0, 5.0, 5.0])
  else:
    raise ValueError('No valid combination of reference and stage.')

  return target_assign.TargetAssigner(similarity_calc, matcher, box_coder,
                        positive_class_weight=positive_class_weight,
                        negative_class_weight=negative_class_weight,
                        unmatched_cls_target=unmatched_cls_target)


def batch_assign_targets(target_assigner,
                         anchors_batch,
                         gt_box_batch,
                         gt_class_targets_batch):
  """Batched assignment of classification and regression targets.

  Args:
    target_assigner: a target assigner.
    anchors_batch: BoxList representing N box anchors or list of BoxList objects
      with length batch_size representing anchor sets.
    gt_box_batch: a list of BoxList objects with length batch_size
      representing groundtruth boxes for each image in the batch
    gt_class_targets_batch: a list of tensors with length batch_size, where
      each tensor has shape [num_gt_boxes_i, classification_target_size] and
      num_gt_boxes_i is the number of boxes in the ith boxlist of
      gt_box_batch.

  Returns:
    batch_cls_targets: a tensor with shape [batch_size, num_anchors,
      num_classes],
    batch_cls_weights: a tensor with shape [batch_size, num_anchors],
    batch_reg_targets: a tensor with shape [batch_size, num_anchors,
      box_code_dimension]
    batch_reg_weights: a tensor with shape [batch_size, num_anchors],
    match_list: a list of matcher.Match objects encoding the match between
      anchors and groundtruth boxes for each image of the batch,
      with rows of the Match objects corresponding to groundtruth boxes
      and columns corresponding to anchors.
  Raises:
    ValueError: if input list lengths are inconsistent, i.e.,
      batch_size == len(gt_box_batch) == len(gt_class_targets_batch)
        and batch_size == len(anchors_batch) unless anchors_batch is a single
        BoxList.
  """
  if not isinstance(anchors_batch, list):
    anchors_batch = len(gt_box_batch) * [anchors_batch]
  if not all(
      isinstance(anchors, box_list.BoxList) for anchors in anchors_batch):
    raise ValueError('anchors_batch must be a BoxList or list of BoxLists.')
  if not (len(anchors_batch)
          == len(gt_box_batch)
          == len(gt_class_targets_batch)):
    raise ValueError('batch size incompatible with lengths of anchors_batch, '
                     'gt_box_batch and gt_class_targets_batch.')

  # Compute k_shot maches first
  target_assigner._matcher.match_k_shot(anchors_batch, gt_box_batch,
                                    gt_class_targets_batch)

  cls_targets_list = []
  cls_weights_list = []
  reg_targets_list = []
  reg_weights_list = []
  match_list = []
  for i, (anchors, gt_boxes, gt_class_targets) in enumerate(zip(
      anchors_batch, gt_box_batch, gt_class_targets_batch)):
    (cls_targets, cls_weights, reg_targets,
     reg_weights, match) = target_assigner.assign(
         anchors, gt_boxes, gt_class_targets, batch_ind=i)
    cls_targets_list.append(cls_targets)
    cls_weights_list.append(cls_weights)
    reg_targets_list.append(reg_targets)
    reg_weights_list.append(reg_weights)
    match_list.append(match)
  batch_cls_targets = tf.stack(cls_targets_list)
  batch_cls_weights = tf.stack(cls_weights_list)
  batch_reg_targets = tf.stack(reg_targets_list)
  batch_reg_weights = tf.stack(reg_weights_list)
  return (batch_cls_targets, batch_cls_weights, batch_reg_targets,
          batch_reg_weights, match_list)
