"""Builder function to construct tf-slim arg_scope for convolution, fc ops."""
import tensorflow as tf

from object_detection.protos import hyperparams_pb2
from object_detection.builders.hyperparams_builder import _build_batch_norm_params, _build_activation_fn
from rcnn_attention.coloc.util import dynamic_conv2d
slim = tf.contrib.slim


def build(dynamic_hyperparams_config, is_training):
  """
  Returns an arg_scope to use for dynamic convolution ops containing activation
  function, batch norm function and batch norm parameters based on the
  configuration.
  Note that if the batch_norm parameteres are not specified in the config
  (i.e. left to default) then batch norm is excluded from the arg_scope.
  The batch norm parameters are set for updates based on `is_training` argument
  and conv_hyperparams_config.batch_norm.train parameter. During training, they
  are updated only if batch_norm.train parameter is true. However, during eval,
  no updates are made to the batch norm variables. In both cases, their current
  values are used during forward pass.
  Args:
    dynamic_hyperparams_config: hyperparams.proto object containing
      hyperparameters.
    is_training: Whether the network is in training mode.
  Returns:
    arg_scope: tf-slim arg_scope containing hyperparameters for ops.
  Raises:
    ValueError: if dynamic_hyperparams_config is not of type hyperparams.Hyperparams.
  """
  if not isinstance(dynamic_hyperparams_config,
                    hyperparams_pb2.Hyperparams):
    raise ValueError('dynamic_hyperparams_config not of type '
                     'hyperparams_pb.Hyperparams.')

  batch_norm = None
  batch_norm_params = None
  if dynamic_hyperparams_config.HasField('batch_norm'):
    batch_norm = slim.batch_norm
    batch_norm_params = _build_batch_norm_params(
        dynamic_hyperparams_config.batch_norm, is_training)

  affected_ops = [dynamic_conv2d]
  with slim.arg_scope(
      affected_ops,
      activation_fn=_build_activation_fn(dynamic_hyperparams_config.activation),
      normalizer_fn=batch_norm,
      normalizer_params=batch_norm_params) as sc:
    return sc

