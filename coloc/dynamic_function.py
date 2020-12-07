import tensorflow as tf
from abc import abstractmethod

class DynamicFunction(object):
  def __init__(self, dynamic_parameters_scope=None):
    self._dynamic_parameters_scope = dynamic_parameters_scope
    self._shape_map = None

  @property
  def dynamic_parameters_scope(self):
    return self._dynamic_parameters_scope

  @dynamic_parameters_scope.setter
  def dynamic_parameters_scope(self, value):
    if value is not None and not value.endswith('/'):
      value = value + '/'
    self._dynamic_parameters_scope = value

  def get_var_global_name(self, var_name):
    if self._dynamic_parameters_scope is not None:
      var_name = self._dynamic_parameters_scope + var_name
    return var_name

  def _get_var_local_name(self, var_name):
    if self._dynamic_parameters_scope is not None:
      assert(var_name.startswith(self._dynamic_parameters_scope)), 'var_name %s should starts with %s' % (var_name, self._dynamic_parameters_scope)
      var_name = var_name[len(self._dynamic_parameters_scope):]
    return var_name

  def build(self, inputs, dynamic_parameters_map={}, scope=None, **param):
      ''' Add the dynamic function operations to the graph.
          Args:
            inputs: the input tensor with shape [batch_size, height, width, input_depth]
            dynamic_parameters_map: dictionary of dynamic paramters
            remove_assigned_dynamic_parameters: if True, removes used dynamic dynamic parameters
              from dynamic_parameters_map
      '''
      if dynamic_parameters_map:
        assert(self._shape_map is not None), 'Call dynamic_parameters_shape_map first.'
        dynamic_parameters_local_map = dict([(self._get_var_local_name(name),
                                              dynamic_parameters_map.pop(name))
                                              for name, value in self._shape_map.iteritems()])
      else:
        dynamic_parameters_local_map = {}

      if scope is not None:
        with tf.variable_scope(scope, values=[inputs, dynamic_parameters_map]):
          net = self._build(inputs, dynamic_parameters_local_map, scope, **param)
      else:
        net = self._build(inputs, dynamic_parameters_local_map, scope, **param)
      assert(len(dynamic_parameters_local_map) == 0), 'unused dynamic parameters'
      return net

  @abstractmethod
  def _build(self, inputs, dynamic_parameters_map, scope, **param):
    raise

  def dynamic_parameters_shape_map(self, **param):
    if self._shape_map is not None:
      raise Exception('dynamic_parameters_shape_map can not be called twice!')
    shape_map = self._dynamic_parameters_shape_map(**param)
    items = [(self.get_var_global_name(name), shape) for name, shape in shape_map.iteritems()]
    self._shape_map = dict(items)
    return self._shape_map

  @abstractmethod
  def _dynamic_parameters_shape_map(self, **param):
    raise
