from os_box_predictor import *
import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

def get_random_dynamic_param(dynamic_parameters_shape_map, batch_size):
  dynamic_parameters_map = dict()
  for name, shape in dynamic_parameters_shape_map.iteritems():
    dynamic_parameters_map[name] = tf.random_normal([batch_size] + shape)
  return dynamic_parameters_map

def os_mask_rcnn_box_predictor_net():
  meta_batch_size = 2
  batch_size = 5
  height = 20
  width = 30
  depth_in = 5

  params = dict(is_training = True,
    num_classes = 10,
    fc_hyperparameters ={},
    use_dropout = True,
    dropout_keep_prob = .5,
    dynamic_fc_hyperparameters = None,
    use_dynamic_box_predictor = False,
    use_dynamic_class_predictor = False)

  inputs = tf.random_normal([meta_batch_size, batch_size, height, width, depth_in])

  predictor = OSMaskRCNNBoxPredictor(**params)
  predictor.init_variables(
          box_code_size=4,
          num_predictions_per_location=1,
          dynamic_parameters_scope = 'OSConvolutionalBoxPredictor')
  dynamic_parameters_shape_map = predictor.dynamic_parameters_shape_map(input_depth=depth_in)
  dynamic_parameters_map = get_random_dynamic_param(dynamic_parameters_shape_map, meta_batch_size)
  net = predictor.build(inputs, dynamic_parameters_map, scope='OSConvolutionalBoxPredictor')
  assert(len(dynamic_parameters_map) == 0)
  return net



def os_convolution_box_predictor_net():
  meta_batch_size = 2
  batch_size = 5
  height = 20
  width = 30
  depth_in = 5

  params = dict(is_training = True,
    num_classes = 10,
    conv_hyperparameters ={},
    use_dropout = True,
    dropout_keep_prob = .5,
    kernel_size = 3,
    net_before_prediction_list = None,
    apply_sigmoid_to_scores = True,
    dynamic_conv_hyperparameters = {},
    use_dynamic_box_predictor = True,
    use_dynamic_class_predictor = False)

  inputs = tf.random_normal([meta_batch_size, batch_size, height, width, depth_in])

  predictor = OSConvolutionalBoxPredictor(**params)
  predictor.init_variables(
          box_code_size=4,
          num_predictions_per_location_list=[9],
          dynamic_parameters_scope = 'OSConvolutionalBoxPredictor')
  dynamic_parameters_shape_map = predictor.dynamic_parameters_shape_map(input_depth=depth_in)
  dynamic_parameters_map = get_random_dynamic_param(dynamic_parameters_shape_map, meta_batch_size)
  net = predictor.build([inputs], dynamic_parameters_map, scope='OSConvolutionalBoxPredictor')
  assert(len(dynamic_parameters_map) == 0)
  return net


if __name__ == '__main__':
  #net = os_mask_rcnn_box_predictor_net()
  net = os_convolution_box_predictor_net()
  init_op = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init_op)
    out = sess.run(net)

