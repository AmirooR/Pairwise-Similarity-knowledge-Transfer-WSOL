
from google.protobuf import text_format
from coloc_ssd.builders import convline_builder, parameter_learner_builder, dynamic_hyperparams_builder
from object_detection.builders import hyperparams_builder
from object_detection.protos import parameter_learner_pb2


def build_param_learner(text_proto):
  parameter_learner_config = parameter_learner_pb2.ParameterLearner()
  text_format.Merge(text_proto, parameter_learner_config)
  argscope_fn = hyperparams_builder.build
  dynamic_argscope_fn = dynamic_hyperparams_builder.build
  net_builder_fn = convline_builder.build

  return parameter_learner_builder.build(
    argscope_fn,
    dynamic_argscope_fn,
    net_builder_fn,
    parameter_learner_config,
    True)


if __name__ == '__main__':
  param_learner_proto = '''
    weight_hashing_parameter_learner {
      output_scale: 0.01
      parameter_predictor_net {
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
      }
      decompression_factor: 50
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
    }
  '''
  parameter_learner = build_param_learner(param_learner_proto)
