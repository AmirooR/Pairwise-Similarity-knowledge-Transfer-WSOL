import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
from abc import abstractmethod
import util

slim = tf.contrib.slim

class ParameterLearner(object):
    def __init__(self, is_training):
      self._is_training = is_training

    def learn(self, inputs, dynamic_parameters_shape_map, query_features=None, scope=None, **params):
      '''
        inputs: [meta_batch_size, batch_size, height0, width0, channels0]
        query_features: [meta_batch_size, height1, widht1, channels0]
        batch_size is usually k_shot*num_classes but since we don't use
        this information.
      '''

      with tf.variable_scope(scope, 'ParameterLearner', [inputs, query_features, dynamic_parameters_shape_map]) as scope:
        return self._learn(inputs, dynamic_parameters_shape_map, query_features, scope, **params)

    @abstractmethod
    def _learn(self, inputs, dynamic_parameters_shape_map, query_features, scope=None, **params):
      pass

class WeightHashingParameterLearner(ParameterLearner):
  def __init__(self, is_training,
               output_scale=None,
               add_bias=False,
               one2one=False,
               tanh_activation=False,
               parameter_prediction_convline=None,
               decompression_factor=None,
               fc_hyperparameters=None):
    super(WeightHashingParameterLearner, self).__init__(is_training)
    self._parameter_prediction_convline = parameter_prediction_convline
    self._decompression_factor = decompression_factor
    self._fc_hyperparameters = fc_hyperparameters
    self._output_scale = output_scale
    self._add_bias = add_bias
    self._one2one = one2one
    self._tanh_activation = tanh_activation

    if parameter_prediction_convline is not None:
      shape_map = parameter_prediction_convline.dynamic_parameters_shape_map()
      assert(len(shape_map) == 0)

  def _reduce_khw(self, inputs):
    if self._parameter_prediction_convline is not None:
      # We dont change number of channels in the convline:
      # TODO: Fix this.
      self._parameter_prediction_convline._filter_num = (
          [inputs.shape[-1]]* len(self._parameter_prediction_convline._filter_num))
      inputs = self._parameter_prediction_convline.build(inputs, scope='ConvLine')

    # Average accross k, w, and h
    return tf.reduce_mean(inputs, [1,2,3])

  def _learn(self, inputs, dynamic_parameters_shape_map, query_features, scope=None, **params):
    assert(len(inputs.shape) == 5), 'Input should be 5 dimensional (meta_batch_size, k, h, w, c)'
    if len(dynamic_parameters_shape_map) == 0:
      return dynamic_parameters_shape_map
    # Apply prediction net and reduce khw dimensions
    inputs = self._reduce_khw(inputs)

    # Adds a fully connected layer with an appropariate 
    # output size
    if self._decompression_factor:
      assert(self._decompression_factor >= 1.0)
      size = int(
              util.dynamic_parameters_size(dynamic_parameters_shape_map) /
              self._decompression_factor)
      assert(size > 0)

      if self._add_bias:
        kwargs = dict(biases_initializer=None)
      else:
        kwargs = dict()

      if self._tanh_activation:
        kwargs['activation_fn'] = tf.nn.tanh
      else:
        kwargs['activation_fn'] = None

      fc_hyperparameters = util.overwrite_arg_scope(
            self._fc_hyperparameters,
            [slim.fully_connected],
            **kwargs)

      with slim.arg_scope(fc_hyperparameters):
        inputs = slim.fully_connected(inputs, size, scope='FC')

    # Scales the ouput 
    if self._output_scale != 1.0:
      inputs *= self._output_scale

    # Decompress the dynamic parameters if necessary 
    if not self._decompression_factor or self._decompression_factor > 1.0:
      inputs = util.hash_dynamic_params(inputs,
                                 dynamic_parameters_shape_map,
                                 one2one=self._one2one, scope='Hash')
    else:
      inputs = util.split_dynamic_parameters(inputs,
                                 dynamic_parameters_shape_map,
                                 scope='Split')
    # Adds biases
    if self._add_bias:
      for i, (key, val) in enumerate(inputs.items()):
        inputs[key] += tf.get_variable('Bias_{}'.format(i),
            shape=[1]+val.shape.as_list()[1:],
            dtype=val.dtype,
            initializer=tf.constant_initializer())
    return inputs
