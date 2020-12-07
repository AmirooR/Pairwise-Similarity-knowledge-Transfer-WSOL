from abc import abstractmethod
import tensorflow as tf
from object_detection.utils import ops
from object_detection.utils import shape_utils
from object_detection.utils import static_shape

slim = tf.contrib.slim

from object_detection.core.box_predictor import *

class RCNNAttentionBoxPredictor(BoxPredictor):
  """R-CNN Attention Box Predictor.

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
               fc_hyperparams,
               use_dropout,
               dropout_keep_prob,
               box_code_size,
               conv_hyperparams=None,
               predict_instance_masks=False,
               mask_prediction_conv_depth=256,
               predict_keypoints=False,
               agnostic_box_regressor=True):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      fc_hyperparams: Slim arg_scope with hyperparameters for fully
        connected ops.
      use_dropout: Option to use dropout or not.  Note that a single dropout
        op is applied here prior to both box and class predictions, which stands
        in contrast to the ConvolutionalBoxPredictor below.
      dropout_keep_prob: Keep probability for dropout.
        This is only used if use_dropout is True.
      box_code_size: Size of encoding for each box.
      conv_hyperparams: Slim arg_scope with hyperparameters for convolution
        ops.
      predict_instance_masks: Whether to predict object masks inside detection
        boxes.
      mask_prediction_conv_depth: The depth for the first conv2d_transpose op
        applied to the image_features in the mask prediciton branch.
      predict_keypoints: Whether to predict keypoints insde detection boxes.


    Raises:
      ValueError: If predict_instance_masks or predict_keypoints is true.
    """
    super(RCNNAttentionBoxPredictor, self).__init__(is_training, num_classes)
    self._fc_hyperparams = fc_hyperparams
    self._use_dropout = use_dropout
    self._box_code_size = box_code_size
    self._dropout_keep_prob = dropout_keep_prob
    self._conv_hyperparams = conv_hyperparams
    self._predict_instance_masks = predict_instance_masks
    self._mask_prediction_conv_depth = mask_prediction_conv_depth
    self._predict_keypoints = predict_keypoints
    if self._predict_instance_masks:
      raise ValueError('Mask prediction is unimplemented.')
    if self._predict_keypoints:
      raise ValueError('Keypoint prediction is unimplemented.')
    if ((self._predict_instance_masks or self._predict_keypoints) and
        self._conv_hyperparams is None):
      raise ValueError('`conv_hyperparams` must be provided when predicting '
                       'masks.')
    self._agnostic_box_regressor = agnostic_box_regressor

  @property
  def num_classes(self):
    return self._num_classes

  def _predict(self, image_features, num_predictions_per_location):
    """Computes encoded object locations and corresponding confidences.

    Flattens image_features and applies fully connected ops (with no
    non-linearity) to predict box encodings and class predictions.  In this
    setting, anchors are not spatially arranged in any way and are assumed to
    have been folded into the batch dimension.  Thus we output 1 for the
    anchors dimension.

    Args:
      image_features: A float tensor of shape [batch_size, height, width,
        channels] containing features for a batch of images.
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
      If predict_masks is True the dictionary also contains:
        instance_masks: A float tensor of shape
          [batch_size, 1, num_classes, image_height, image_width]
      If predict_keypoints is True the dictionary also contains:
        keypoints: [batch_size, 1, num_keypoints, 2]

    Raises:
      ValueError: if num_predictions_per_location is not 1.
    """
    if num_predictions_per_location != 1:
      raise ValueError('Currently FullyConnectedBoxPredictor only supports '
                       'predicting a single box per class per location.')
    spatial_averaged_image_features = tf.reduce_mean(image_features, [1, 2],
                                                     keep_dims=True,
                                                     name='AvgPool')
    flattened_image_features = slim.flatten(spatial_averaged_image_features)
    if self._use_dropout:
      flattened_image_features = slim.dropout(flattened_image_features,
                                              keep_prob=self._dropout_keep_prob,
                                              is_training=self._is_training)
    with slim.arg_scope(self._fc_hyperparams):
      if self._agnostic_box_regressor:
        num_boxes = 1
      else:
        num_boxes = self._num_classes
      box_encodings = slim.fully_connected(
          flattened_image_features,
          num_boxes * self._box_code_size,
          activation_fn=None,
          scope='BoxEncodingPredictor')
      if self._agnostic_box_regressor:
        shape = shape_utils.combined_static_and_dynamic_shape(box_encodings)
        box_encodings = tf.tile(box_encodings,
                [1]*len(shape[:-1]) + [self._num_classes])

      class_predictions_with_background = slim.fully_connected(
          flattened_image_features,
          self._num_classes + 1,
          activation_fn=None,
          scope='ClassPredictor')
    box_encodings = tf.reshape(
        box_encodings, [-1, 1, self._num_classes, self._box_code_size])
    class_predictions_with_background = tf.reshape(
        class_predictions_with_background, [-1, 1, self._num_classes + 1])

    predictions_dict = {
        BOX_ENCODINGS: box_encodings,
        CLASS_PREDICTIONS_WITH_BACKGROUND: class_predictions_with_background
    }

    if self._predict_instance_masks:
      with slim.arg_scope(self._conv_hyperparams):
        upsampled_features = slim.conv2d_transpose(
            image_features,
            num_outputs=self._mask_prediction_conv_depth,
            kernel_size=[2, 2],
            stride=2)
        mask_predictions = slim.conv2d(upsampled_features,
                                       num_outputs=self.num_classes,
                                       activation_fn=None,
                                       kernel_size=[1, 1])
        instance_masks = tf.expand_dims(tf.transpose(mask_predictions,
                                                     perm=[0, 3, 1, 2]),
                                        axis=1,
                                        name='MaskPredictor')
      predictions_dict[MASK_PREDICTIONS] = instance_masks
    return predictions_dict
