from parameter_learner import *
import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

def get_random_dynamic_param(dynamic_parameters_shape_map, batch_size):
  dynamic_parameters_map = dict()
  for name, shape in dynamic_parameters_shape_map.iteritems():
    dynamic_parameters_map[name] = tf.random_normal([batch_size] + shape)
  return dynamic_parameters_map

def weight_hashing_parameter_learner_net():
  num_classes = 1
  k_shot = 5

  meta_batch_size = 1
  height = 1
  width = 1
  depth_in = 5

  params = dict(is_training = True,
    after_hashing_scale_hyperparameters={},
    parameter_prediction_convline=None,
    fc_hyperparameters ={},
    decompression_factor=num_classes*k_shot)

  inputs = tf.random_normal([
      meta_batch_size, num_classes*k_shot,
      height, width, depth_in])
  dynamic_params_shape_map = dict(w=[3,3,5,50], b=[50])
  param_learner = WeightHashingParameterLearner(**params)
  net = param_learner.learn(inputs, dynamic_params_shape_map)
  return net

from IPython import embed
if __name__ == '__main__':
  net = weight_hashing_parameter_learner_net()
  init_op = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init_op)
    out = sess.run(net)
    embed()

