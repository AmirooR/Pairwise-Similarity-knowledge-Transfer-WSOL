import tensorflow as tf
from attention_tree import *
from cross_similarity import CrossSimilarity
import numpy as np
import object_detection.utils.shape_utils as shape_utils
from IPython import embed

slim = tf.contrib.slim
PROPOSALS_OFFSETS = 100

class EqualCrossSimilarity(CrossSimilarity):
  def __init__(self):
    super(EqualCrossSimilarity, self).__init__()
    self._loss_fn = self._loss_fn2

  def _loss_fn2(self, scores, labels):
    # Check if the label is one-hot
    unique, _ = tf.unique(tf.reshape(tf.to_int32(labels), [-1]))
    vals, _ = tf.nn.top_k(unique, k=2)
    assert0 = tf.Assert(tf.logical_and(tf.equal(tf.shape(unique)[0], 2),
                      tf.reduce_all(tf.equal(vals, [1, 0]))), [unique])

    # Check if the label is computed correctly
    is_fg = tf.cast(1 - labels[..., 0], tf.bool)
    fg_scores = tf.boolean_mask(scores, is_fg)
    fg_labels = tf.boolean_mask(tf.cast(labels, tf.bool), is_fg)
    sum_scores = tf.reduce_sum(fg_scores, axis=-1)[..., tf.newaxis]
    fg_scores_onehot = tf.equal(fg_scores, sum_scores)

    assert1 = tf.Assert(tf.reduce_all(tf.equal(fg_scores_onehot, fg_labels)),
                        [labels])
    with tf.control_dependencies([assert0, assert1]):
      loss = tf.zeros_like(scores[..., 0])
    return loss

  def normalize_scores(self, scores):
    return scores, tf.reduce_max(scores, axis=-1)

  def _build(self, fea0, fea1, score_size, reuse_vars):
    fea0_shape = shape_utils.combined_static_and_dynamic_shape(fea0)
    m = fea0_shape[1]

    if fea1 is not None:
      fea0 = tf.tile(fea0[:, :, tf.newaxis], [1,1,m,1])
      fea1 = tf.tile(fea1[:, tf.newaxis], [1,m,1,1])
      fea01 = tf.concat((fea0, fea1), axis=-1)
    else:
      fea01 = fea0[:, :, tf.newaxis]
    self._joined_fea = fea01
    fea01 = tf.mod(fea01, PROPOSALS_OFFSETS)
    shape = shape_utils.combined_static_and_dynamic_shape(fea01)
    fea01 = tf.to_int32(tf.reshape(fea01, [-1] + [shape[-1]]))
    fea01 = tf.map_fn(lambda fea: tf.bincount(fea, minlength=score_size,
                                              maxlength=score_size),
                      fea01, dtype=tf.int32)
    fea01 = tf.to_float(tf.reshape(fea01, shape[:-1] + [-1]))
    return fea01

  @property
  def support_pre_joining(self):
    return True

def attention_unit(ncobj_proposals, max_ncobj_proposals,
                   num_classes, loss_minibatch_size, is_training):
  def _fn(k, parallel_iterations):
    return AttentionUnit(ncobj_proposals, max_ncobj_proposals,
                         positive_balance_fraction=.5,
                         k=k,
                         pre_match_convline=None,
                         post_match_convline=None,
                         cross_similarity_cal=EqualCrossSimilarity(),
                         is_training=is_training,
                         use_orig_feas_in_post_match_convline=False,
                         loss_positive_balance_fraction=0.5,
                         loss_minibatch_size=loss_minibatch_size,
                         loss_weight=1.0,
                         negative_example_weight=1.0,
                         compute_scores_after_matching=True,
                         num_classes=num_classes,
                         class_agnostic=False,
                         parallel_iterations=parallel_iterations)
  return _fn

def attention_tree(k, num_classes, is_training):
  units = [attention_unit(8, 8, num_classes, 0, is_training),
           attention_unit(4, 4, num_classes, 4, is_training)]

  return AttentionTree(units,
                       k_shot=k,
                       is_training=is_training,
                       stop_features_gradient=False,
                       preprocess_convline=None,
                       num_classes=num_classes)

def inputs():
  is_training = True
  batch_size = 1
  k = 4
  proposal = 8
  num_classes = proposal/2

  matched_class = np.arange(batch_size*k*proposal) % proposal + 1
  matched_class = np.reshape(matched_class, (batch_size*k, proposal))

  fea = np.reshape(np.float32(matched_class),
            (batch_size*k, proposal, 1, 1, 1))
  for i in range(batch_size*k):
    fea[i] += i * PROPOSALS_OFFSETS

  matched_class[matched_class > num_classes] = 0
  matched_class = tf.one_hot(tf.constant(matched_class), num_classes + 1)
  fea = tf.constant(fea)
  return attention_tree(k, num_classes, is_training), fea, matched_class, num_classes


if __name__ == '__main__':
  attention_tree, fea, matched_class, num_classes = inputs()
  res = attention_tree.build(fea, matched_class, 16)
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  sess.run(attention_tree.loss())
  embed()

