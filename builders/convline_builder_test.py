
from google.protobuf import text_format
from rcnn_attention.builders import convline_builder, dynamic_hyperparams_builder
from object_detection.protos import convline_pb2
from object_detection.builders import hyperparams_builder

def build_convline(text_proto):
  convline_config = convline_pb2.ConvLine()
  text_format.Merge(text_proto, convline_config)
  argscope_fn = hyperparams_builder.build
  dynamic_argscope_fn = dynamic_hyperparams_builder.build
  return convline_builder.build(argscope_fn, dynamic_argscope_fn,
                                convline_config, True)


if __name__ == '__main__':
  text_proto = '''
    kernel_size: 3
    filter_num_list: [3, 5, 10]
    conv_hyperparameters {
      regularizer {
        l1_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
    }
    dynamic_kernel_size: 3
    dynamic_filter_num_list: [0, 2, 0]
  '''
  convline = build_convline(text_proto)
