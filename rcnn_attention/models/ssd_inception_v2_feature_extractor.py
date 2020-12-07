
"""SSDFeatureExtractor for InceptionV2 features."""
import tensorflow as tf

from coloc_ssd.coloc import coloc_ssd_meta_arch
from coloc_ssd.models import feature_map_generators
from nets import inception_v2

slim = tf.contrib.slim


class SSDInceptionV2FeatureExtractor(coloc_ssd_meta_arch.CoLocSSDFeatureExtractor):
  """SSD Feature Extractor using InceptionV2 features."""

  def __init__(self,
               depth_multiplier,
               min_depth,
               conv_hyperparams,
               reuse_weights=None,
               endpoints_indicator=None):
    """InceptionV2 Feature Extractor for SSD Models.

    Args:
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      conv_hyperparams: tf slim arg_scope for conv2d and separable_conv2d ops.
      reuse_weights: Whether to reuse variables. Default is None.
    """
    super(SSDInceptionV2FeatureExtractor, self).__init__(
        depth_multiplier, min_depth, conv_hyperparams,
        reuse_weights, endpoints_indicator)
    self._endpoints_indicator = endpoints_indicator or '111111'
    assert(len(self._endpoints_indicator) == 6)

  def preprocess(self, resized_inputs):
    """SSD preprocessing.

    Maps pixel values to the range [-1, 1].

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    return (2.0 / 255.0) * resized_inputs - 1.0

  def extract_features(self, preprocessed_inputs):
    """Extract features from preprocessed inputs.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    """
    preprocessed_inputs.get_shape().assert_has_rank(4)
    shape_assert = tf.Assert(
        tf.logical_and(tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
                       tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
        ['image size must at least be 33 in both height and width.'])

    feature_map_layout = {
        'from_layer': ['Mixed_4c', 'Mixed_5c', '', '', '', ''],
        'layer_depth': [-1, -1, 512, 256, 256, 128],
    }

    with tf.control_dependencies([shape_assert]):
      with slim.arg_scope(self._conv_hyperparams):
        with tf.variable_scope('InceptionV2',
                               reuse=self._reuse_weights) as scope:
          _, image_features = inception_v2.inception_v2_base(
              preprocessed_inputs,
              final_endpoint='Mixed_5c',
              min_depth=self._min_depth,
              depth_multiplier=self._depth_multiplier,
              scope=scope)
          feature_maps = feature_map_generators.multi_resolution_feature_maps(
              feature_map_layout=feature_map_layout,
              depth_multiplier=self._depth_multiplier,
              min_depth=self._min_depth,
              insert_1x1_conv=True,
              image_features=image_features)
    ret = []
    for endpoint_indicator, feature_map in zip(
          self._endpoints_indicator,
          feature_maps.values()):
      if endpoint_indicator == '1':
        ret.append(feature_map)
    return ret
