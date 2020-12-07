"""Function to build attention tree from configuration."""

from rcnn_attention.attention import attention_tree
from rcnn_attention.builders import (cross_similarity_builder,
                                     convline_builder,
                                     attention_loss_builder)

from object_detection.protos import attention_tree_pb2
from functools import partial
from collections import namedtuple
import tensorflow as tf
from object_detection.utils import shape_utils

slim = tf.contrib.slim

def _post_convline_builder(argscope_fn,
                           convline,
                           use_tanh_sigmoid,
                           res_type,
                           res_fc_hyperparams,
                           split_fea_in_res,
                           is_training):

  def _tanh_sigmoid(fea, scope=None):
    if scope:
      scope_tanh = scope + '_tanh'
      scope_sigmoid = scope + '_sigmoid'
    else:
      scope_tanh = 'tanh'
      scope_sigmoid = 'sigmoid'
    tanh = tf.tanh(convline.build(fea, scope=scope_tanh))
    sigmoid = tf.sigmoid(convline.build(fea, scope=scope_sigmoid))
    return tf.multiply(tanh, sigmoid)

  def _res_block(fea, scope=None):
    def _split_fea(fea):
      shape = shape_utils.combined_static_and_dynamic_shape(fea)
      dim = int(shape[-1]/2)
      fea = tf.reshape(fea, shape[:-1] + [2, dim])
      return fea

    with tf.variable_scope('res_block'):
      if split_fea_in_res:
        # fea has shape [..., 2*fea_size] we have
        # to reshape it to [..., 2, fea_size] and
        # apply fc and reshape it back.
        fea = _split_fea(fea)

      if res_fc_hyperparams is not None:
        with slim.arg_scope(res_fc_hyperparams):
          with slim.arg_scope([slim.fully_connected],
                              activation_fn=None,
                              normalizer_fn=None,
                              normalizer_params=None):
            shape = shape_utils.combined_static_and_dynamic_shape(fea)
            dim = shape[-1]
            if split_fea_in_res:
              dim /= shape[-1]
            fea = slim.fully_connected(fea, dim)
      if split_fea_in_res:
        if res_type == 'sum':
          fea = tf.reduce_mean(fea, axis=-2)
        else:
          shape = shape_utils.combined_static_and_dynamic_shape(fea)
          nshape = shape[:-2] + [-1]
          fea = tf.reshape(fea, nshape)
      return fea

  def _func(fea, scope=None):
    if convline is None:
      out = _res_block(fea)
    else:
      if use_tanh_sigmoid:
        out = _tanh_sigmoid(fea, scope=scope)
      else:
        out = convline.build(fea, scope=scope)
      if res_type == 'sum':
        out += _res_block(fea)
      elif res_type == 'concat':
        out = tf.concat([out, _res_block(fea)], axis=-1)
    return out

  if convline is None and res_type == 'none':
    return None
  return namedtuple('PostConvlineBlock', ['build'])(_func)

def build_attention_unit(argscope_fn, unit_config,
                         num_classes, k_shot,
                         is_training, calibration_type,
                         is_calibration):

  def _fn(k, tree, parall_iterations):
    pre_convline, post_convline, negative_convline = None, None, None
    if unit_config.HasField('pre_convline'):
      pre_convline = convline_builder.build(argscope_fn, None,
                                            unit_config.pre_convline,
                                            is_training)

    if unit_config.HasField('post_convline'):
      post_convline = convline_builder.build(argscope_fn, None,
                                            unit_config.post_convline,
                                            is_training)
    if unit_config.HasField('negative_convline'):
      negative_convline = convline_builder.build(argscope_fn, None,
                                            unit_config.negative_convline,
                                            is_training)

    res_fc_hyperparams = None
    if unit_config.HasField('res_fc_hyperparams'):
      res_fc_hyperparams = argscope_fn(unit_config.res_fc_hyperparams,
                                       is_training)

    post_convline = _post_convline_builder(argscope_fn,
                                           post_convline,
                                           unit_config.use_tanh_sigmoid_in_post_convline,
                                           unit_config.post_convline_res,
                                           res_fc_hyperparams,
                                           unit_config.split_fea_in_res,
                                           is_training)

    cross_similarity = cross_similarity_builder.build(argscope_fn,
                                                      unit_config.cross_similarity,
                                                      tree,
                                                      k, is_training)
    loss = attention_loss_builder.build(unit_config.loss, num_classes, k)

    max_ncobj_proposals = unit_config.ncobj_proposals
    if is_training:
      ncobj_proposals = unit_config.training_subsampler.ncobj_proposals
      if unit_config.training_subsampler.HasField('topk'):
        max_ncobj_proposals = unit_config.training_subsampler.topk
    else:
      ncobj_proposals = max_ncobj_proposals

    res_fc_hyperparams = None
    if unit_config.HasField('res_fc_hyperparams'):
      res_fc_hyperparams = argscope_fn(unit_config.res_fc_hyperparams,
                                       is_training)

    positive_balance_fraction = None
    if unit_config.training_subsampler.HasField('positive_balance_fraction'):
      positive_balance_fraction = unit_config.training_subsampler.positive_balance_fraction

    return attention_tree.AttentionUnit(ncobj_proposals, max_ncobj_proposals,
                                        positive_balance_fraction,
                                        k, pre_convline, post_convline,
                                        cross_similarity,
                                        loss,
                                        is_training,
                                        unit_config.orig_fea_in_post_convline,
                                        unit_config.training_subsampler.sample_hard_examples,
                                        unit_config.training_subsampler.stratified,
                                        unit_config.loss.positive_balance_fraction,
                                        unit_config.loss.minibatch_size,
                                        parall_iterations,
                                        unit_config.loss.weight,
                                        unit_config.negative_example_weight,
                                        unit_config.compute_scores_after_matching,
                                        unit_config.overwrite_fea_by_scores,
                                        negative_convline,
                                        is_calibration,
                                        calibration_type,
                                        unit_config.unary_energy_scale,
                                        unit_config.transfered_objectness_weight)
  return _fn


def build(argscope_fn, attention_tree_config,
          k_shot, num_classes, num_negative_bags, is_training, is_calibration):
  """Builds attention_tree based on the configuration.
  Args:
    argscope_fn: A function that takes the following inputs:
        * hyperparams_pb2.Hyperparams proto
        * a boolean indicating if the model is in training mode.
      and returns a tf slim argscope for Conv and FC hyperparameters.
    attention_tree_config:
    k_shot:
    is_training: Whether the models is in training mode.
  Returns:
    attention_tree: attention.attention_tree.AttentionTree object.
  Raises:
    ValueError: On unknown parameter learner.
  """
  if not isinstance(attention_tree_config, attention_tree_pb2.AttentionTree):
    raise ValueError('attention_tree_config not of type '
                     'attention_tree_pb2.AttentionTree.')

  fea_split_ind = None
  if attention_tree_config.HasField('fea_split_ind'):
    fea_split_ind = attention_tree_config.fea_split_ind

  preprocess_convline = None
  if attention_tree_config.HasField('preprocess_convline'):
    preprocess_convline = convline_builder.build(argscope_fn,
                                          None,
                                          attention_tree_config.preprocess_convline,
                                          is_training)
  rescore_convline = None
  rescore_fc_hyperparams = None
  if attention_tree_config.rescore_instances:
    if attention_tree_config.HasField('rescore_convline'):
      rescore_convline = convline_builder.build(argscope_fn,
                                              None,
                                              attention_tree_config.rescore_convline,
                                              is_training)
    if attention_tree_config.HasField('rescore_fc_hyperparams'):
      rescore_fc_hyperparams = argscope_fn(
            attention_tree_config.rescore_fc_hyperparams,
            is_training)

  negative_preprocess_convline = None
  if attention_tree_config.HasField('negative_preprocess_convline'):
    negative_preprocess_convline = convline_builder.build(argscope_fn,
        None,
        attention_tree_config.negative_preprocess_convline,
        is_training)

  negative_postprocess_convline = None
  if attention_tree_config.HasField('negative_postprocess_convline'):
    negative_postprocess_convline = convline_builder.build(argscope_fn,
        None,
        attention_tree_config.negative_postprocess_convline,
        is_training)
  calibration_type = attention_tree_config.CalibrationType.Name(attention_tree_config.calibration_type)
  units = [build_attention_unit(argscope_fn,
                                unit_config,
                                num_classes,
                                k_shot,
                                is_training,
                                calibration_type,
                                is_calibration)
                                for unit_config in attention_tree_config.unit]

  subsampler_ncobj, subsampler_pos_frac = None, None
  subsampler_agnostic = False
  if attention_tree_config.HasField('training_subsampler'):
      subsampler = attention_tree_config.training_subsampler
      if subsampler.HasField('positive_balance_fraction'):
        subsampler_pos_frac = subsampler.positive_balance_fraction
      if subsampler.HasField('ncobj_proposals'):
        subsampler_ncobj = subsampler.ncobj_proposals
      if subsampler.HasField('agnostic'):
        subsampler_agnostic = subsampler.agnostic

  return attention_tree.AttentionTree(units, k_shot,
                                      num_negative_bags, is_training,
                                      attention_tree_config.stop_features_gradient,
                                      preprocess_convline,
                                      num_classes,
                                      attention_tree_config.rescore_instances,
                                      attention_tree_config.rescore_min_match_frac,
                                      rescore_convline,
                                      rescore_fc_hyperparams,
                                      negative_preprocess_convline,
                                      negative_postprocess_convline,
                                      subsampler_ncobj,
                                      subsampler_pos_frac,
                                      subsampler_agnostic,
                                      fea_split_ind)
