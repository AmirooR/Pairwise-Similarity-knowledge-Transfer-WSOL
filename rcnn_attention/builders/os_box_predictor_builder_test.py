
from google.protobuf import text_format
from rcnn_attention.builders import convline_builder, dynamic_hyperparams_builder, os_box_predictor_builder
from object_detection.protos import os_box_predictor_pb2
from object_detection.builders import hyperparams_builder


def build_os_box_predictor(text_proto):
  os_box_predictor_config = os_box_predictor_pb2.OSBoxPredictor()
  text_format.Merge(text_proto, os_box_predictor_config)
  argscope_fn = hyperparams_builder.build
  dynamic_argscope_fn = dynamic_hyperparams_builder.build
  net_builder_fn = convline_builder.build

  return os_box_predictor_builder.build(
    argscope_fn, dynamic_argscope_fn,
    net_builder_fn,
    os_box_predictor_config,
    True, 10)


if __name__ == '__main__':
  osconv_proto = '''
    os_convolutional_box_predictor {
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
      net_before_prediction {
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
      }
      use_dropout: true
      dropout_keep_probability: .5
      kernel_size: 5
      apply_sigmoid_to_scores: true
      use_dynamic_box_predictor: true
      use_dynamic_class_predictor: true
    }
  '''

  osmaskrcnn_proto = '''
    os_mask_rcnn_box_predictor {
      fc_hyperparameters {
        regularizer {
          l1_regularizer {
          }
        }
        initializer {
          truncated_normal_initializer {
          }
        }
      }
      use_dropout: true
      dropout_keep_probability: .5
      use_dynamic_box_predictor: true
      use_dynamic_class_predictor: true
    }
  '''

  os_box_predictor = build_os_box_predictor(osconv_proto)
