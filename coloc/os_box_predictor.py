import tensorflow as tf
from object_detection.utils import ops
from object_detection.utils import shape_utils
from object_detection.utils import static_shape
from object_detection.core.box_predictor import *
from util import (ConvLine, dynamic_conv2d,
                  custom_convline, dict_union,
                  collect_debug, overwrite_arg_scope)
from dynamic_function import DynamicFunction

slim = tf.contrib.slim


class OSBoxPredictor(DynamicFunction):
  """OSBoxPredictor."""

  def __init__(self, is_training,  num_classes):
    """Constructor.
    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
  """
    super(OSBoxPredictor, self).__init__()
    self._is_training = is_training
    self._num_classes = num_classes

  @property
  def num_classes(self):
    return self._num_classes

  def init_variables(self, **param):
    self._init_variables(**param)

  @abstractmethod
  def _init_variables(self, **param):
    pass

  @property
  def box_predictor_dynamic_param_scope(self):
      return 'BoxEncodingPredictor'

  @property
  def class_predictor_dynamic_param_scope(self):
      return 'ClassPredictor'

class OSMaskRCNNBoxPredictor(OSBoxPredictor):
  """Mask R-CNN Box Predictor.
  See Mask R-CNN: He, K., Gkioxari, G., Dollar, P., & Girshick, R. (2017).
  Mask R-CNN. arXiv preprint arXiv:1703.06870.
  This is used for the second stage of the Mask R-CNN detector where proposals
  cropped from an image are arranged along the batch dimension of the input
  image_features tensor. Notice that locations are *not* shared across classes,
  thus for each anchor, a separate prediction is made for each class.
  In addition to predicting boxes and classes, optionally this class allows
  predicting masks and/or keypoints inside detection boxes.
  Currently this box predictor makes per-class predictions; that is, each
  anchor makes a separate box prediction for each class.
  """

  def __init__(self,
               is_training,
               num_classes,
               fc_hyperparameters,
               use_dropout,
               dropout_keep_prob,
               dynamic_fc_hyperparameters=None,
               use_dynamic_box_predictor=False,
               use_dynamic_class_predictor=False):
    """Constructor.
    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      fc_hyperparameters: Slim arg_scope with hyperparameters for fully
        connected ops.
      use_dropout: Option to use dropout or not.  Note that a single dropout
        op is applied here prior to both box and class predictions, which stands
        in contrast to the ConvolutionalBoxPredictor below.
      dropout_keep_prob: Keep probability for dropout.
        This is only used if use_dropout is True.
      box_code_size: Size of encoding for each box.
        Raises:
      ValueError: if num_predictions_per_location is not 1.
    """

    super(OSMaskRCNNBoxPredictor, self).__init__(is_training, num_classes)

    self._use_dropout = use_dropout
    self._dropout_keep_prob = dropout_keep_prob

    self._fc_hyperparameters = fc_hyperparameters
    self._dynamic_fc_hyperparameters = dynamic_fc_hyperparameters
    self._use_dynamic_box_predictor  = use_dynamic_box_predictor
    self._use_dynamic_class_predictor = use_dynamic_class_predictor

  def _init_variables(self, box_code_size,
                      num_predictions_per_location=None,
                      dynamic_parameters_scope=None):
    self._box_code_size = box_code_size
    self.dynamic_parameters_scope = dynamic_parameters_scope

    if num_predictions_per_location != 1:
      raise ValueError('Currently FullyConnectedBoxPredictor only supports '
                       'predicting a single box per class per location.')
    fc_hyperparameters = self._fc_hyperparameters
    if fc_hyperparameters:
      fc_hyperparameters = overwrite_arg_scope(fc_hyperparameters,
                                               activation_fn=None)

    dynamic_fc_hyperparameters = self._dynamic_fc_hyperparameters
    if dynamic_fc_hyperparameters:
      dynamic_fc_hyperparameters = overwrite_arg_scope(dynamic_fc_hyperparameters,
                                                       activation_fn=None)

    self._box_encoding_predictor_convline = custom_convline(1,
                        self._num_classes * self._box_code_size,
                        fc_hyperparameters, dynamic_fc_hyperparameters,
                        self._use_dynamic_box_predictor,
                        dynamic_parameters_scope=self.box_predictor_dynamic_param_scope)

    self._class_predictor_convline = custom_convline(1,
                        self._num_classes + 1,
                        fc_hyperparameters, dynamic_fc_hyperparameters,
                        self._use_dynamic_class_predictor,
                        dynamic_parameters_scope=self.class_predictor_dynamic_param_scope)

  def _build(self, image_features, dynamic_parameters_map, scope=None):
    """Computes encoded object locations and corresponding confidences.
    Flattens image_features and applies fully connected ops (with no
    non-linearity) to predict box encodings and class predictions.  In this
    setting, anchors are not spatially arranged in any way and are assumed to
    have been folded into the batch dimension.  Thus we output 1 for the
    anchors dimension.
    Args:
      image_features: A float tensor of shape [meta_batch_size,
        batch_size, height, width, channels] containing features
        for a batch of images.
      dynamic_parameters_map:
      num_predictions_per_location: an integer representing the number of box
        predictions to be made per spatial location in the feature map.
        Currently, this must be set to 1, or an error will be raised.
    Returns:
      A dictionary containing the following tensors.
        box_encodings: A float tensor of shape
          [batch_size, 1, num_classes, code_size] representing the
          location of the objects.
        class_predictions_with_background: A float tensor of shape
          [batch_size, 1, num_classes + 1] representing the class
          predictions for the proposals.
   """

    spatial_averaged_image_features = tf.reduce_mean(image_features, [1, 2],
                                                     keep_dims=True,
                                                     name='AvgPool')
    if self._use_dropout:
      spatial_averaged_image_features = slim.dropout(spatial_averaged_image_features,
                                              keep_prob=self._dropout_keep_prob,
                                              is_training=self._is_training)
    box_encodings = self._box_encoding_predictor_convline.build(
            spatial_averaged_image_features,
            dynamic_parameters_map=dynamic_parameters_map,
            scope='BoxEncodingPredictor')
    class_predictions_with_background = self._class_predictor_convline.build(
            spatial_averaged_image_features,
            dynamic_parameters_map=dynamic_parameters_map,
            scope='ClassPredictor')

    box_encodings = tf.reshape(
        box_encodings, [-1, 1, self._num_classes, self._box_code_size])
    class_predictions_with_background = tf.reshape(
        class_predictions_with_background, [-1, 1, self._num_classes + 1])

    return {
        BOX_ENCODINGS: box_encodings,
        CLASS_PREDICTIONS_WITH_BACKGROUND: class_predictions_with_background
    }

  def _dynamic_parameters_shape_map(self, input_depth):
    shape_map = (
              self._box_encoding_predictor_convline.dynamic_parameters_shape_map(input_depth=input_depth))
    dict_union(shape_map,
              self._class_predictor_convline.dynamic_parameters_shape_map(input_depth=input_depth))
    return shape_map

class _NoopVariableScope(object):
  """A dummy class that does not push any scope."""

  def __enter__(self):
    return None

  def __exit__(self, exc_type, exc_value, traceback):
    return False

class OSConvolutionalBoxPredictor(OSBoxPredictor):
  """Convolutional Box Predictor.
  Optionally add an intermediate 1x1 convolutional layer after features and
  predict in parallel branches box_encodings and
  class_predictions_with_background.
  Currently this box predictor assumes that predictions are "shared" across
  classes --- that is each anchor makes box predictions which do not depend
  on class.
  """

  def __init__(self,
               is_training,
               num_classes,
               conv_hyperparameters,
               use_dropout,
               dropout_keep_prob,
               kernel_size,
               net_before_prediction_list=None,
               apply_sigmoid_to_scores=False,
               dynamic_conv_hyperparameters=None,
               use_dynamic_box_predictor=False,
               use_dynamic_class_predictor=False):
    """Constructor.
    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      use_dropout: Option to use dropout for class prediction or not.
      dropout_keep_prob: Keep probability for dropout.
        This is only used if use_dropout is True.
      kernel_size: Size of final convolution kernel.  If the
        spatial resolution of the feature map is smaller than the kernel size,
        then the kernel size is automatically set to be
        min(feature_width, feature_height).
      net_before_prediction_list: list of optional nets to be used before prediction
      apply_sigmoid_to_scores: if True, apply the sigmoid on the output
        class_predictions.
      use_dynamic_box_predictor: if True, use convolution with
        dynamic parameters for prediction.
      use_dynamic_class_predictor: if True, use convolution
        with dynamic parameters for class prediction.
    """
    super(OSConvolutionalBoxPredictor, self).__init__(
           is_training, num_classes)

    self._net_before_prediction_list = net_before_prediction_list

    self._use_dropout = use_dropout
    self._dropout_keep_prob = dropout_keep_prob
    self._apply_sigmoid_to_scores = apply_sigmoid_to_scores
    self._conv_hyperparameters = conv_hyperparameters
    self._dynamic_conv_hyperparameters = dynamic_conv_hyperparameters
    self._kernel_size = kernel_size
    self._use_dynamic_box_predictor = use_dynamic_box_predictor
    self._use_dynamic_class_predictor = use_dynamic_class_predictor

  def _init_variables(self, box_code_size, num_predictions_per_location_list,
      dynamic_parameters_scope=None):
    self._box_code_size = box_code_size
    self._num_predictions_per_location_list = num_predictions_per_location_list
    self.dynamic_parameters_scope = dynamic_parameters_scope

    conv_hyperparameters = self._conv_hyperparameters
    if conv_hyperparameters:
      conv_hyperparameters = overwrite_arg_scope(
                            conv_hyperparameters,
                            [slim.conv2d],
                            activation_fn=None,
                            normalizer_fn=None,
                            normalizer_params=None)

    dynamic_conv_hyperparameters = self._dynamic_conv_hyperparameters
    if dynamic_conv_hyperparameters:
      dynamic_conv_hyperparameters = overwrite_arg_scope(
                            dynamic_conv_hyperparameters,
                            [dynamic_conv2d],
                            activation_fn=None,
                            normalizer_fn=None,
                            normalizer_params=None)

    num_class_slots = self._num_classes + 1
    self._box_encoding_predictor_convline_list = []
    self._class_predictor_convline_list = []
    for i, num_predictions_per_location in enumerate(num_predictions_per_location_list):
      scope_ind = '_{}'.format(i)
      self._box_encoding_predictor_convline_list.append(
          custom_convline(self._kernel_size,
                          num_predictions_per_location*
                          self._box_code_size,
                          conv_hyperparameters,
                          dynamic_conv_hyperparameters,
                          self._use_dynamic_box_predictor,
                          dynamic_parameters_scope=self.box_predictor_dynamic_param_scope+scope_ind))

      self._class_predictor_convline_list.append(
          custom_convline(1,
                          num_predictions_per_location*
                          num_class_slots,
                          conv_hyperparameters,
                          dynamic_conv_hyperparameters,
                          self._use_dynamic_class_predictor,
                          dynamic_parameters_scope=self.class_predictor_dynamic_param_scope+scope_ind))
      if self._net_before_prediction_list:
        self._net_before_prediction_list[i].dynamic_parameters_scope = 'NetBeforePrediction'+scope_ind

  def _build(self, image_features, dynamic_parameters_map, scope=None):
    """Computes encoded object locations and corresponding confidences.
    Args:
      image_features: A list of float tensor of shape [meta_batch_size, batch_size, height, width,
        channels] containing features for a batch of images.
    Returns:
      A dictionary containing the following tensors.
        box_encodings: A float tensor of shape [batch_size, num_anchors, 1,
          code_size] representing the location of the objects, where
          num_anchors = feat_height * feat_width * num_predictions_per_location
        class_predictions_with_background: A float tensor of shape
          [batch_size, num_anchors, num_classes + 1] representing the class
          predictions for the proposals.
    """
    assert(len(self._num_predictions_per_location_list) == len(image_features))
    box_encodings_list = []
    class_predictions_list = []
    # TODO(rathodv): Come up with a better way to generate scope names
    # in box predictor once we have time to retrain all models in the zoo.
    # The following lines create scope names to be backwards compatible with the
    # existing checkpoints.
    box_predictor_scopes = [_NoopVariableScope()]
    if len(image_features) > 1:
      box_predictor_scopes = [
          tf.variable_scope('BoxPredictor_{}'.format(i))
          for i in range(len(image_features))
      ]

    for i, (image_feature,
            num_predictions_per_location,
            box_encoding_predictor_convline,
            class_predictor_convline,
            box_predictor_scope) in enumerate(zip(
             image_features,
             self._num_predictions_per_location_list,
             self._box_encoding_predictor_convline_list,
             self._class_predictor_convline_list,
             box_predictor_scopes)):
      with box_predictor_scope:
        net = image_feature
        collect_debug('boxpredictor/input_{}'.format(i), net)
        if self._net_before_prediction_list:
          net = self._net_before_prediction[i].build(net, dynamic_parameters_map)
        collect_debug('boxpredictor/beforeprediction_{}'.format(i), net)
        # Add a slot for the background class.
        num_class_slots = self._num_classes + 1

        with slim.arg_scope([slim.dropout], is_training=self._is_training):
          box_encodings = box_encoding_predictor_convline.build(net,
                                dynamic_parameters_map=dynamic_parameters_map,
                                scope='BoxEncodingPredictor')
          if self._use_dropout:
            net = slim.dropout(net, keep_prob=self._dropout_keep_prob)
          class_predictions_with_background = class_predictor_convline.build(
                                net,
                                dynamic_parameters_map=dynamic_parameters_map,
                                scope='ClassPredictor')
          collect_debug('boxpredictor/classes_{}'.format(i),
                        class_predictions_with_background)
          collect_debug('boxpredictor/boxes_{}'.format(i),
                        box_encodings)
          if self._apply_sigmoid_to_scores:
            class_predictions_with_background = tf.sigmoid(
                  class_predictions_with_background)

        combined_feature_map_shape = (shape_utils.
                                      combined_static_and_dynamic_shape(
                                      image_feature))
        box_encodings = tf.reshape(
            box_encodings, tf.stack([combined_feature_map_shape[0]*
                                     combined_feature_map_shape[1], #meta_batch_size*background
                                     combined_feature_map_shape[2] *
                                     combined_feature_map_shape[3] * #w*h
                                     num_predictions_per_location,
                                     1, self._box_code_size]))
        class_predictions_with_background = tf.reshape(
            class_predictions_with_background,
            tf.stack([combined_feature_map_shape[0] *
                      combined_feature_map_shape[1],
                      combined_feature_map_shape[2] *
                      combined_feature_map_shape[3] *
                      num_predictions_per_location,
                      num_class_slots]))
        box_encodings_list.append(box_encodings)
        class_predictions_list.append(class_predictions_with_background)

    return {BOX_ENCODINGS: box_encodings_list,
            CLASS_PREDICTIONS_WITH_BACKGROUND: class_predictions_list}

  def dynamic_parameters_shape_map_list(self, input_depth_list):
    assert(len(input_depth_list) == len(self._num_predictions_per_location_list))
    shape_map = dict()
    shape_map_list = []
    if self._shape_map is not None:
      raise Exception('dynamic_parameters_shape_map_list can not be called twice!')
    for i in range(len(input_depth_list)):
      nshape_map = self.partial_dynamic_parameters_shape_map(input_depth_list, i)
      nshape_map = dict([(self.get_var_global_name(name), shape) for name, shape in nshape_map.iteritems()])
      dict_union(shape_map, nshape_map)
      shape_map_list.append(nshape_map)
    self._shape_map = shape_map
    return shape_map_list

  def partial_dynamic_parameters_shape_map(self, input_depth_list, ind):
    shape_map=dict()
    box_encoding_predictor_convline = self._box_encoding_predictor_convline_list[ind]
    class_predictor_convline = self._class_predictor_convline_list[ind]
    input_depth = input_depth_list[ind]
    if self._net_before_prediction_list:
      dict_union(shape_map,
                 self._net_before_prediction_list[ind].dynamic_parameters_shape_map(
                 input_depth=input_depth))
      input_depth = self._net_before_prediction.depth_out()

    dict_union(shape_map,
                 box_encoding_predictor_convline.dynamic_parameters_shape_map(
                 input_depth=input_depth))

    dict_union(shape_map,
                 class_predictor_convline.dynamic_parameters_shape_map(
                 input_depth=input_depth))
    return shape_map

