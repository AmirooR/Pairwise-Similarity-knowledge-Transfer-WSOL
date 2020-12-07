import tensorflow as tf
import numpy as np
from util import *

slim = tf.contrib.slim

def get_random_dynamic_param(dynamic_parameters_shape_map, batch_size):
  dynamic_parameters_map = dict()
  for name, shape in dynamic_parameters_shape_map.iteritems():
    dynamic_parameters_map[name] = tf.random_normal([batch_size] + shape)
  return dynamic_parameters_map

def dynamic_conv2d_net():
  meta_batch_size = 5
  batch_size = 2
  height = 20
  width = 30
  depth_in = 5
  depth_out = 10
  kernel_size = 3
  inputs = tf.random_normal([meta_batch_size, batch_size, height, width, depth_in])
  kernels = tf.random_normal([meta_batch_size, kernel_size, kernel_size, depth_in, depth_out])
  biases = tf.random_normal([meta_batch_size, depth_out])

  with slim.arg_scope([dynamic_conv2d], rate=1,
        activation_fn=tf.nn.relu):
    dyn_net = dynamic_conv2d(inputs, kernels, biases=biases)
  return dyn_net


def convline_net():
  meta_batch_size = 5
  batch_size = 2
  height = 20
  width = 30
  depth_in = 5
  depth_out = 10
  kernel_size = 3
  inputs = tf.random_normal([meta_batch_size, batch_size, height, width, depth_in])
  filter_num = [5, 10, 15]
  dynamic_filter_num = [3,0,3]

  conv_hyperparams = dict()
  dynamic_conv_hyperparams = dict()

  convline = ConvLine(kernel_size=kernel_size,
                      filter_num=filter_num,
                      conv_hyperparameters=conv_hyperparams,
                      dynamic_kernel_size=kernel_size,
                      dynamic_filter_num=dynamic_filter_num,
                      dynamic_conv_hyperparameters=dynamic_conv_hyperparams)

  convline.dynamic_parameters_scope = 'Scope'
  shape_map = convline.dynamic_parameters_shape_map(input_depth=depth_in)
  param_map = get_random_dynamic_param(shape_map, meta_batch_size)
  return convline.build(inputs, param_map)

def weight_hashing_net():
  batch_size = 10
  inputs = tf.random_normal([batch_size, 1, 1, 10])
  output_shape = [2,2,1,3]
  return weight_hashing(inputs, output_shape)

def parameter_hashing_net():
  batch_size = 10
  inputs = tf.random_normal([batch_size, 1, 1, 10]) # 10 elems
  dynamic_params_shape_map = dict(w0=[1,1,2,2], b0=[2], #6 elems +
                               w1=[3,3,1,1], #9 elems +
                               w2=[2,2,1,1], b2=[1]) #5 elems = 20 elems 
  return hash_dynamic_params(inputs, dynamic_params_shape_map)

def conv_parameter_hashing_net():
  meta_batch_size = 5
  batch_size = 2
  inputs = tf.random_normal([meta_batch_size, batch_size, 1, 1, 20]) # 20 elems
  dynamic_params_shape_map = dict(w0=[1,1,2,4], b0=[4], #12 elems +
                               w1=[3,3,1,2], #18 elems +
                               w2=[2,2,1,2], b2=[2]) #10 elems = 40 elems 
  return hash_dynamic_conv_params(inputs, dynamic_params_shape_map)

from IPython import embed
if __name__ == '__main__':
  net = convline_net()
  #net = dynamic_conv2d_net()
  #net = weight_hashing_net()
  #net = parameter_hashing_net()
  #net = conv_parameter_hashing_net()
  init_op = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init_op)
    out = sess.run(net)
    embed()
