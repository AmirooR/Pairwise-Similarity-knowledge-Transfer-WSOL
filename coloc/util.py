import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
import numpy as np
import copy
from dynamic_function import DynamicFunction
slim = tf.contrib.slim

# dynamic_parameters_map: dictionary that maps dynamic  parameters to their tensor
# dynamic_parameters_shape_map: dictionary that maps dynamic paramters to their shape

############################### Weight hashing helper functions #####################################
def overwrite_arg_scope(arg_scope, ops_list, **kwargs):
  with slim.arg_scope(arg_scope):
    with slim.arg_scope(ops_list, **kwargs) as sc:
        return sc

def dict_union(dict_a, dict_b, allow_similar_keys=False, over_write_a=True):
    if not allow_similar_keys:
        intersect = set(dict_a.keys()) & set(dict_b.keys())
        assert(len(intersect) == 0), 'Dublicated keys %s' % str(intersect)
    if not over_write_a:
        dict_a = dict(dict_a)

    dict_a.update(dict_b)
    return dict_a

def dynamic_parameters_size(dynamic_parameters_shape_map):
  return sum([arr_size(shape) for shape in dynamic_parameters_shape_map.values()])

def arr_size(shape):
  return reduce(lambda x, y: x*y, shape)

def weight_hashing(inputs, output_shape, output_scale_factor=1.0, one2one=False, scope=None):
  ''' Weight hashing function implementation with fully connected layer.
        inputs: input tensor with shape [batch_size, ...]
        output_shape: list of scalers repesenting the shape of the output. output_shape
          would be [batch_size] + output_shape
        output_scale_factor: optionaly can scale the output value
  '''
  def weight_hashing_initializer(scale_factor=1.0, dtype=tf.float32):
    def _initializer(shape, dtype=dtype, partition_info=None):
      if shape is None or len(shape) != 2:
        raise ValueError('Only supports 2d shaped parameters.')
      fan_in = shape[-2]
      fan_out = shape[-1]
      if fan_in > fan_out:
        raise ValueError('Fan in should be less than or equal to fan out.')
      nparr = np.zeros(shape, dtype=dtype.as_numpy_dtype())
      if one2one:
        rows = np.choice(np.arange(fan_in),
                  size=fan_in, replace=False)
        cols = np.choice(np.arange(fan_out),
                  size=fan_in, replace=False)
      else:
        rows = np.random.randint(fan_in, size=fan_out)
        cols = np.arange(fan_out)

      nparr[rows, cols] = (2 * np.random.randint(2, size=len(rows)) - 1) * scale_factor
      return tf.constant(nparr, dtype=dtype)
    return _initializer

  numel = arr_size(output_shape)
  with tf.variable_scope(scope, 'WeightHash') as scope:
    before_reshape = slim.fully_connected(inputs,
                                          numel,
                                          activation_fn=None,
                                          normalizer_fn=None,
                                          normalizer_params=None,
                                          weights_initializer=weight_hashing_initializer(scale_factor=output_scale_factor),
                                          weights_regularizer=None,
                                          biases_initializer=None,
                                          biases_regularizer=None,
                                          reuse=False,
                                          trainable=False,
                                          scope=scope)

    return tf.reshape(before_reshape, shape=[-1] + output_shape)

def split_dynamic_parameters(inputs, dynamic_parameters_shape_map, scope=None):
  with tf.variable_scope(scope, 'SplitDynamicParams') as scope:
    # Compute the total size of the output params
    total_size = dynamic_parameters_size(dynamic_parameters_shape_map)

    # Compute the input size
    inputs = slim.flatten(inputs)
    inputs_size = inputs.shape.as_list()[1]
    if inputs_size is None:
      raise ValueError('For parameter hashing inputs shape must be defined.')

    assert(total_size, inputs_size), 'For spliting input size should matches the output size'
    start_offset = 0
    dynamic_parameters_map = dict()
    for param_name, param_shape in dynamic_parameters_shape_map.iteritems():
      param_size = arr_size(param_shape)
      assert(param_size > 0), 'parameter {} size is 0'.format(param_name)
      end_offset = start_offset + param_size
      inputs_part = inputs[:, start_offset:end_offset]
      dynamic_parameters_map[param_name] = tf.reshape(inputs_part,
          inputs.shape.as_list()[0:1] + param_shape)
      start_offset = end_offset
  return dynamic_parameters_map

def hash_dynamic_params(inputs, dynamic_parameters_shape_map, one2one=False, scope=None):
  ''' Use weight_hashing to map inputs to the tensors with shapes represented in dynamic_parameters_shape_map
        inputs: input tensor [batch_size, ...]
        dynamic_parameters_shape_map: dictionary that maps dynamic paramters to their shape
      Returns:
       dynamic_parameters_map: list of dynamic parameters. Each element is a dictionary in which keys are the
         names of the dynamic parameters and values are their tensors.
  '''

  with tf.variable_scope(scope, 'HashDynamicParams') as scope:
    # Compute the total size of the output params
    total_size = dynamic_parameters_size(dynamic_parameters_shape_map)

    # Compute the input size
    inputs = slim.flatten(inputs)
    inputs_size = inputs.shape.as_list()[1]
    if inputs_size is None:
      raise ValueError('For parameter hashing inputs shape must be defined.')

    decompression_factor = total_size/float(inputs_size)
    start_offset = 0
    dynamic_parameters_map = dict()
    for param_name, param_shape in dynamic_parameters_shape_map.iteritems():
      end_offset = start_offset + max(int(arr_size(param_shape)/decompression_factor), 1)
      inputs_part = inputs[:, start_offset:end_offset]
      param_tensor = weight_hashing(inputs_part, param_shape, one2one=one2one)
      dynamic_parameters_map[param_name] = param_tensor
      start_offset = end_offset
  return dynamic_parameters_map

def hash_dynamic_conv_params(inputs,
                             dynamic_parameters_shape_map,
                             one2one=False,
                             scope=None):
  '''
    This function works specifically for hashing dynamic convolution parameters to be used in convline. Assume we have
    batch_size independent examples in each support set. The input shape is then [meta_batch_size, batch_size, ....] in which
    inputs[i] represents the batch_size independent features which are extracted from the i^th support set.
    The convolutional parameters for each supportset are then concatinated to each other to form paramters
    with shape [..., k*batch_size]. So the last axis of the shapes in
    dynamic_parameters_shape_map should be divisible by batch_size.
  '''

  # Divide each the last dimension of each parameter shape in dynamic_parameters_shape_map by batch_size
  batch_size = inputs.shape[1].value
  meta_batch_size = inputs.shape[0].value
  assert(batch_size and meta_batch_size), 'meta_batch_size and batch_size should be defined'

  dynamic_parameters_shape_map = copy.deepcopy(dynamic_parameters_shape_map)
  for param_shape in dynamic_parameters_shape_map.values():
    if param_shape[-1] % batch_size != 0:
      raise ValueError('channel_out in dyanmic_params_shapes_list %d should be divisible by batch_size %d' % (param_shape[-1], batch_size))
    param_shape[-1] /= batch_size

  dynamic_params_map = hash_dynamic_params(tf.reshape(inputs, (meta_batch_size*batch_size, -1)),
                                           dynamic_parameters_shape_map,
                                           one2one=one2one, scope=scope)

  # Merge dynamic params from independent examples in the supportset.
  merged_dynamic_params_map = dict()
  for param_name, param_tensor in dynamic_params_map.iteritems():
      # Param_tensor has shape [meta_batch_size*batch_size, ...., out_channels]

      # Reshape it to [meta_batch_size, batch_size, ....]
      nshape = tf.shape(param_tensor)
      nshape = tf.concat([[nshape[0]/batch_size, batch_size], nshape[1:]], axis=0)
      param_tensor = tf.reshape(param_tensor, nshape)

      # Bring batch_size dimension to the end
      static_rank = len(param_tensor.shape)
      perm = [0] + range(2, static_rank) + [1]
      param_tensor = tf.transpose(param_tensor, perm)

      # Reshape to [meta_batch_size, ..., out_channels*batch_size]
      nshape = tf.shape(param_tensor)
      nshape = tf.concat([nshape[:-2], [-1]], axis=0)
      merged_dynamic_params_map[param_name] = tf.reshape(param_tensor, nshape)

  return merged_dynamic_params_map

######################################## Dynamic convolution helper functions #########################
def tfmap(fn, inputs, params, name=None):
  return tf.map_fn(lambda x: fn(*x), (inputs, params), dtype=inputs.dtype, name=name)

@add_arg_scope
def dynamic_conv2d(
        inputs,
        kernels,
        biases=None,
        rate=1,
        padding='SAME',
        data_format=None,
        activation_fn=tf.nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        scope=None):

  ''''
    dynamic 2d convolution implementation
    inputs shape [meta_batch_size, batch_size, h, w, out]
    kernels shape [meta_batch_size, kh, kw, out, in]
    biases shape [meta_batch_size, out]
  '''
  assert(kernels.shape[1:].is_fully_defined()), 'Kernel shape should be fully defined'
  assert(len(kernels.shape) == 5), 'kernel should be 5 dimensional (meta_batch_size, kh, kw, out, in)'
  assert(len(inputs.shape) == 5), 'inputs should be 5 dimensional (meta_batch_size, batch_size, h, w, in)'

  if kernels.shape[0].value and inputs.shape[0].value:
      assert(kernels.shape[0] == inputs.shape[0]), 'Kernel and input meta batch size mismatch'
  else:
      tf.assert_equal(tf.shape(kernels)[0],
          tf.shape(inputs)[0], message='Kernel and input meta batch size mismatch')

  if data_format not in [None, 'NHWC']:
    raise ValueError('Invalid data_format: %r' % (data_format,))

  with tf.name_scope(scope, 'dynamic_conv2d', [inputs, kernels, biases]) as scope:
    inputs = tf.convert_to_tensor(inputs)
    kernels = tf.convert_to_tensor(kernels)
    if rate > 1:
      conv2d = lambda inputs, kernel: tf.nn.atrous_conv2d(inputs, kernel, rate, padding=padding)
    else:
      strides = [1]*4
      conv2d = lambda inputs, kernel: tf.nn.conv2d(inputs, kernel, strides, padding=padding)
    outputs = tfmap(conv2d, inputs, kernels, name='conv2d')

    if biases is not None and normalizer_fn is None:
      biases = tf.convert_to_tensor(biases)
      add_bias = lambda inputs, biases: tf.nn.bias_add(inputs, biases)
      outputs = tfmap(add_bias, outputs, biases, name='biases')

    if normalizer_fn is not None:
      normalizer_params = normalizer_params or {}
      outputs = normalizer_fn(outputs, **normalizer_params)

    if activation_fn is not None:
        outputs = activation_fn(outputs)
    return outputs


class ConvLine(DynamicFunction):
    ''' Series of convolutional layers
        each can be hybrid combination of
        static and dynamic convolution
    '''
    def __init__(self,
                 kernel_size,
                 filter_num,
                 conv_hyperparameters,
                 dynamic_kernel_size=None,
                 dynamic_filter_num=None,
                 dynamic_conv_hyperparameters=None):
      super(ConvLine, self).__init__()
      filter_num = filter_num or []
      dynamic_filter_num = dynamic_filter_num or []

      self._nlayers = len(filter_num)
      if len(dynamic_filter_num) > 0:
        assert(self._nlayers == 0 or
                self._nlayers == len(dynamic_filter_num)), 'Lenght of dynamic_filter_num and filter_num does not match.'
        self._nlayers = len(dynamic_filter_num)
      assert(self._nlayers > 0), 'Can not create empty convline.'

      self._kernel_size = kernel_size
      self._filter_num = filter_num
      self._conv_hyperparameters = conv_hyperparameters

      self._dynamic_kernel_size = dynamic_kernel_size
      self._dynamic_filter_num = dynamic_filter_num
      self._dynamic_conv_hyperparameters = dynamic_conv_hyperparameters

    def depth_out(self, ind=-1):
        dout = 0
        if len(self._filter_num) > 0:
            dout += self._filter_num[ind]
        if len(self._dynamic_filter_num) > 0:
            dout += self._dynamic_filter_num[ind]
        return dout

    def _dynamic_var_name(self, ind, name):
      assert(name in ['weights', 'biases'])
      var_name = 'DynamicConv_%d/%s:0' % (ind, name)
      return var_name

    def _dynamic_parameters_shape_map(self, input_depth=None):
      depth_in = input_depth
      shape_map = dict()

      for i, depth in enumerate(self._dynamic_filter_num):
        if depth > 0:
          w = self._dynamic_var_name(i, 'weights')
          b = self._dynamic_var_name(i, 'biases')
          shape_map[w] = [self._dynamic_kernel_size,
                          self._dynamic_kernel_size,
                          depth_in,
                          depth]
          shape_map[b] = [depth]
        depth_in = self.depth_out(i)
      return shape_map

    def _build(self, inputs, dynamic_parameters_map, scope):
      assert(len(inputs.shape) == 5), 'Convline input should be 5 dimensional (meta_batch_size, batch_size, h, w, in)'
      for i in range(self._nlayers):
        output_list = []
        if len(self._filter_num) > 0 and self._filter_num[i] > 0:
          with slim.arg_scope(self._conv_hyperparameters):
            # Reshape (meta_batch_size, batch_size, h, w, in)
            # ==> (meta_batch_size*batch_size, h, w, in)
            in_ = tf.reshape(inputs, tf.concat([[-1], tf.shape(inputs)[2:]], 0))
            static_conv = slim.conv2d(in_, self._filter_num[i],
                    [self._kernel_size, self._kernel_size])
            # Reshape (meta_batch_size*batch_size, h, w, out)
            # ==> (meta_batch_size, batch_size, h, w, out)
            static_conv = tf.reshape(static_conv, tf.concat([tf.shape(inputs)[:2], tf.shape(static_conv)[1:]], 0))
            output_list.append(static_conv)

        if len(self._dynamic_filter_num) > 0 and self._dynamic_filter_num[i] > 0:
          w = self._dynamic_var_name(i, 'weights')
          w_tensor = dynamic_parameters_map.pop(w)

          b = self._dynamic_var_name(i, 'biases')
          if dynamic_parameters_map.has_key(b):
            b_tensor = dynamic_parameters_map.pop(b)
          else:
            b_tensor = None
          with slim.arg_scope(self._dynamic_conv_hyperparameters):
            dynamic_conv = dynamic_conv2d(inputs, w_tensor, b_tensor)
          output_list.append(dynamic_conv)

        if len(output_list) == 1:
          inputs = output_list[0]
        elif len(output_list) > 1:
          inputs = tf.concat(output_list, axis=4)
        else:
          raise Exception
      return inputs

def custom_convline(kernel_size, num_filters, conv_hyperparameters,
          dynamic_conv_hyperparameters, use_dyanamic_conv, dynamic_parameters_scope=None, num_conv=1):
    if use_dyanamic_conv:
      filter_num = None
      dynamic_filter_num = [num_filters] * num_conv
    else:
      filter_num = [num_filters] * num_conv
      dynamic_filter_num = None
    convline = ConvLine(kernel_size, filter_num, conv_hyperparameters,
            dynamic_kernel_size=kernel_size,
            dynamic_filter_num=dynamic_filter_num,
            dynamic_conv_hyperparameters=dynamic_conv_hyperparameters)
    convline.dynamic_parameters_scope = dynamic_parameters_scope
    return convline


debug_collection = []
def collect_debug(name, val):
    debug_collection.append((name, val))

def get_debug_collection():
    return debug_collection
