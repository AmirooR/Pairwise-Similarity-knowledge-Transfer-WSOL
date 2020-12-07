import functools
import tensorflow as tf
import tensorflow.contrib.layers as tcl

# -----------------------------------------------------------------------------
def conv_layer(inputs, training, filters=64,name='conv', stride=2, max_pool=False):
  activation=tf.nn.relu
  conv_initialization=tf.contrib.layers.xavier_initializer_conv2d(tf.float32)
  with tf.variable_scope(name):
    out = tcl.conv2d(inputs, num_outputs=filters, kernel_size=3, stride=stride, activation_fn=None, normalizer_fn=None, weights_initializer=conv_initialization)
    out = tcl.batch_norm(out, is_training=training, activation_fn=activation)
    if max_pool:
      out = tcl.max_pool2d(out,2)
    return out

def fc(inputs, training, out_dim):
  net = tcl.fully_connected(inputs, out_dim, activation_fn=None)
  net = tcl.batch_norm(net, is_training=training, activation_fn=tf.nn.relu)
  return net

def omniglot_model(
    inputs,
    out_dim=64,
    training=False,
):
  activations = {}
  net = inputs
  for i in range(4):
    net = conv_layer(net, training, name='conv_{}'.format(i+1))
    activations['conv_{}'.format(i+1)] = net
  net = tcl.flatten(net)
  activations['flattened'] = net
  net = fc(net, training, out_dim)
  activations['fc'] = net
  return net, activations

