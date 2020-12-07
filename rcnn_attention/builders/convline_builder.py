
from object_detection.protos import convline_pb2
from rcnn_attention.coloc.util import ConvLine
from collections import namedtuple

def build(argscope_fn, dynamic_argscope_fn, convline_config, is_training):
  if not isinstance(convline_config, convline_pb2.ConvLine):
    raise ValueError('convline_config not of type '
      'convline_pb2.ConvLine.')

  if len(convline_config.filter_num_list) == 0:
    def _func(fea, **kwargs):
      return fea
    EmptyConvline = namedtuple('EmptyConvline', ['build'])
    return EmptyConvline(_func)

  conv_hyperparameters = None
  if convline_config.conv_hyperparameters:
    conv_hyperparameters = argscope_fn(convline_config.conv_hyperparameters,
                                     is_training)

  dynamic_conv_hyperparameters = None
  if convline_config.dynamic_conv_hyperparameters and dynamic_argscope_fn:
    dynamic_conv_hyperparameters = dynamic_argscope_fn(
                                    convline_config.dynamic_conv_hyperparameters,
                                    is_training)

  return ConvLine(convline_config.kernel_size,
                  convline_config.filter_num_list,
                  conv_hyperparameters,
                  dynamic_kernel_size=convline_config.dynamic_kernel_size,
                  dynamic_filter_num=convline_config.dynamic_filter_num_list,
                  dynamic_conv_hyperparameters=dynamic_conv_hyperparameters)
