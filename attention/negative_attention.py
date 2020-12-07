import tensorflow as tf
import object_detection.utils.shape_utils as shape_utils
from abc import abstractmethod
from util import add_extra_tensor
from rcnn_attention.attention.cross_similarity import CosineCrossSimilarity, EuclideanCrossSimilarity

slim = tf.contrib.slim

class NegativeAttention(object):
  def __init__(self, convline=None,
      concat_type="NEGATIVE_IN_ORIG",
      similarity_type="COSINE",
      use_gt_labels=False,
      add_loss=False):
    self._convline = convline
    self._similarity_cal = None
    self._concat_type = concat_type
    if similarity_type == 'COSINE':
      self._similarity_cal = CosineCrossSimilarity()
    elif similarity_type == 'EUCLIDEAN':
      self._similarity_cal = EuclideanCrossSimilarity()
    else:
      raise ValueError('similarity_type: {} is not defined'.format(similarity_type))
    self._use_gt_labels = use_gt_labels
    self._add_loss = add_loss

  def build(self, fea, neg_fea, matched_class0, neg_matched_class, reuse_vars=False, scope=None):
    '''
      Args:
        fea: feature tensor of size [MBS*K, 1, B, 1, d] of positive bags
        neg_fea: feature tensor of size [MBS, N*B, 1, 1, d] of negative bags
        matched_class0: [MBS*K, B, num_classes+1]
        neg_matched_class: [MBS*N, B, num_classes+1]
    '''
    with tf.variable_scope(scope, 'negative_attention',
        reuse=reuse_vars, values=[fea, neg_fea]):
      neg_fea_shape = shape_utils.combined_static_and_dynamic_shape(neg_fea)
      orig_neg_fea = neg_fea
      #[MBS*K, 1, B, 1, d] -> [MBS*K, B, d]
      orig_fea_shape = shape_utils.combined_static_and_dynamic_shape(fea)
      orig_fea = fea
      fea = tf.squeeze(fea, [1,3])
      fea_shape = shape_utils.combined_static_and_dynamic_shape(fea)
      #[MBS*K, B, d] -> [MBS, K*B, d]
      fea = tf.reshape(fea, [neg_fea_shape[0], -1] + fea_shape[2:])
      add_extra_tensor('input_fea', fea)
      add_extra_tensor('input_neg_fea', neg_fea)
      #[MBS, K*B, 1, 1, d]
      fea = fea[:, :, tf.newaxis, tf.newaxis, :]

      if self._convline is not None:
        #[MBS, (K+N)*B, 1, 1, d]
        combined_fea = tf.concat([fea, neg_fea], axis=1)
        combined_fea = self._convline.build(combined_fea, scope='combined_convline')
        #combined_fea = tf.squeeze(combined_fea, [2,3])
        fea = combined_fea[:,:-neg_fea_shape[1], ...]
        neg_fea = combined_fea[:, -neg_fea_shape[1]:, ...]
        add_extra_tensor('conv_fea', fea)
        add_extra_tensor('conv_neg_fea', neg_fea)

      fea = tf.squeeze(fea, [2,3])
      neg_fea = tf.squeeze(neg_fea, [2,3])

      gt_alphas = None
      loss = None
      if neg_matched_class is not None and matched_class0 is not None:
        # [MBS, N*B, num_class+1]
        neg_matched_class = tf.reshape(neg_matched_class, neg_fea_shape[:2]+ [-1])
        neg_cls_shape = shape_utils.combined_static_and_dynamic_shape(neg_matched_class)
        # [MBS, K*B, num_class+1] (k is 1 always...)
        matched_class0 = tf.reshape(matched_class0, [neg_cls_shape[0], -1, neg_cls_shape[2]])
        # [MBS, K*B, N*B]
        gt_alphas = tf.matmul(matched_class0[...,1:], neg_matched_class[...,1:], transpose_b=True)

      #[MBS, N*B+1, d]
      neg_fea = tf.pad(neg_fea, [[0,0],[1,0],[0,0]])
      if self._use_gt_labels and gt_alphas is not None:
        #[MBS, K*B, N*B]
        alphas = gt_alphas / (tf.reduce_sum(gt_alphas, axis=-1, keep_dims=True) + 1e-7)
        #[MBS, K*B, N*B+1] TODO: not sure about this
        alphas = tf.pad(alphas, [[0,0],[0,0],[1,0]])
      else:
        scores, pairs, joined_fea, _ = self._similarity_cal.build(fea, neg_fea,
                                                                  None, None,
                                                                  2, None, reuse_vars)
        #[MBS, K*B, N*B+1] scores
        scores = tf.reshape(scores[...,1], [neg_fea_shape[0], -1, neg_fea_shape[1]+1])
        add_extra_tensor('scores', scores)
        if self._add_loss and gt_alphas is not None:
          gt_alphas = tf.cast(gt_alphas, tf.bool)
          is_in_neg = tf.reduce_any(gt_alphas, axis=-1, keep_dims=True)
          not_in_neg = tf.logical_not(is_in_neg)
          labels = tf.to_float(tf.concat([not_in_neg, gt_alphas], axis=-1))
          labels = labels/tf.reduce_sum(labels, axis=-1, keep_dims=True)
          logits = scores
          loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
          loss = tf.reduce_mean(loss)
        #[MBS, K*B, N*B+1]
        alphas = tf.nn.softmax(scores, dim=2)
      add_extra_tensor('alphas', alphas)
      neg_fea_to_attend, pos_fea = None, None
      if self._concat_type == 'NEGATIVE_IN_ORIG':
        # orig_neg_fea is [MBS, N*B, 1, 1, d] -> [MBS, N*B, d]
        orig_neg_fea = tf.squeeze(orig_neg_fea, [2,3])
        # [MBS, N*B+1, d]
        orig_neg_fea = tf.pad(orig_neg_fea, [[0,0],[1,0],[0,0]])
        neg_fea_to_attend = orig_neg_fea
        pos_fea = orig_fea
      elif self._concat_type == 'NEGATIVE_IN_FEA':
        neg_fea_to_attend = neg_fea
        pos_fea = orig_fea
      elif self._concat_type == 'CONCAT_IN_FEA':
        neg_fea_to_attend = neg_fea
        pos_fea = tf.reshape(fea, orig_fea_shape[:4]+[-1])
      else:
        raise ValueError('concat type {} is not defined'.format(self._concat_type))
      # [MBS, K*B, N*B+1] * [MBS, N*B+1, d] -> [MBS, K*B, d]
      attended_neg_feas = tf.matmul(alphas, neg_fea_to_attend)
      add_extra_tensor('attended_neg_feas', attended_neg_feas)
      attended_neg_feas = tf.reshape(attended_neg_feas, orig_fea_shape[:4]+[-1])
      fea01 = tf.concat((pos_fea, attended_neg_feas), axis=-1)
      return fea01, loss


