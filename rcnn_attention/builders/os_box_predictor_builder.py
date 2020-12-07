"""Function to build one-shot box predictor from configuration."""

from rcnn_attention.coloc import os_box_predictor
from object_detection.protos import os_box_predictor_pb2


def build(argscope_fn, dynamic_argscope_fn, net_builder_fn,
          os_box_predictor_config, is_training, num_classes):
  """Builds box predictor based on the configuration.
  Builds box predictor based on the configuration. See box_predictor.proto for
  configurable options. Also, see box_predictor.py for more details.
  Args:
    argscope_fn: A function that takes the following inputs:
        * hyperparams_pb2.Hyperparams proto
        * a boolean indicating if the model is in training mode.
      and returns a tf slim argscope for Conv and FC hyperparameters.
    dynamic_argscope_fn:
    net_builder_fn:
    os_box_predictor_config: os_box_predictor_pb2.OSBoxPredictor proto containing
      configuration.
    is_training: Whether the models is in training mode.
    num_classes: Number of classes to predict.
  Returns:
    box_predictor: box_predictor.BoxPredictor object.
  Raises:
    ValueError: On unknown box predictor.
  """
  if not isinstance(os_box_predictor_config, os_box_predictor_pb2.OSBoxPredictor):
    raise ValueError('os_box_predictor_config not of type '
                     'os_box_predictor_pb2.OSBoxPredictor.')

  box_predictor_oneof = os_box_predictor_config.WhichOneof('os_box_predictor_oneof')

  if box_predictor_oneof == 'os_convolutional_box_predictor':
    os_conv_box_predictor = os_box_predictor_config.os_convolutional_box_predictor

    conv_hyperparams = None
    if os_conv_box_predictor.HasField('conv_hyperparameters'):
      conv_hyperparams = argscope_fn(os_conv_box_predictor.conv_hyperparameters,
                                 is_training)
    dynamic_conv_hyperparams = None
    if os_conv_box_predictor.HasField('dynamic_conv_hyperparameters'):
      dynamic_conv_hyperparams = dynamic_argscope_fn(
        os_conv_box_predictor.dynamic_conv_hyperparameters, is_training)

    net_before_prediction_list = []
    for net_before_prediction in os_conv_box_predictor.net_before_prediction:
      net_before_prediction_list.append(net_builder_fn(argscope_fn,
                                           dynamic_argscope_fn,
                                           net_before_prediction,
                                           is_training))

    return os_box_predictor.OSConvolutionalBoxPredictor(
        is_training=is_training,
        num_classes=num_classes,
        conv_hyperparameters=conv_hyperparams,
        use_dropout=os_conv_box_predictor.use_dropout,
        dropout_keep_prob=os_conv_box_predictor.dropout_keep_probability,
        kernel_size=os_conv_box_predictor.kernel_size,
        net_before_prediction_list=net_before_prediction_list,
        apply_sigmoid_to_scores=os_conv_box_predictor.apply_sigmoid_to_scores,
        dynamic_conv_hyperparameters=dynamic_conv_hyperparams,
        use_dynamic_box_predictor=os_conv_box_predictor.use_dynamic_box_predictor,
        use_dynamic_class_predictor=os_conv_box_predictor.use_dynamic_class_predictor)

  if box_predictor_oneof == 'os_mask_rcnn_box_predictor':
    os_mask_rcnn_box_predictor = os_box_predictor_config.os_mask_rcnn_box_predictor
    fc_hyperparams = None
    if os_mask_rcnn_box_predictor.HasField('fc_hyperparameters'):
      fc_hyperparams = argscope_fn(os_mask_rcnn_box_predictor.fc_hyperparameters,
                                   is_training)

    dynamic_fc_hyperparameters = None
    if os_mask_rcnn_box_predictor.HasField('dynamic_fc_hyperparameters'):
      dynamic_fc_hyperparameters = dynamic_argscope_fn(
          os_mask_rcnn_box_predictor.dynamic_fc_hyperparameters, is_training)

    return os_box_predictor.OSMaskRCNNBoxPredictor(
        is_training=is_training,
        num_classes=num_classes,
        fc_hyperparameters=fc_hyperparams,
        use_dropout=os_mask_rcnn_box_predictor.use_dropout,
        dropout_keep_prob=os_mask_rcnn_box_predictor.dropout_keep_probability,
        dynamic_fc_hyperparameters=dynamic_fc_hyperparameters,
        use_dynamic_box_predictor=os_mask_rcnn_box_predictor.use_dynamic_box_predictor,
        use_dynamic_class_predictor=os_mask_rcnn_box_predictor.use_dynamic_class_predictor)

  raise ValueError('Unknown box predictor: {}'.format(box_predictor_oneof))
