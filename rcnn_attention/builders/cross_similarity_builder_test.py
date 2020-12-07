
from google.protobuf import text_format
from rcnn_attention.builders import convline_builder
from rcnn_attention.builders import cross_similarity_builder
from object_detection.builders import hyperparams_builder
from object_detection.protos import cross_similarity_pb2

def build_cross_similarity(text_proto):
  cross_similarity_config = cross_similarity_pb2.CrossSimilarity()
  text_format.Merge(text_proto, cross_similarity_config)
  argscope_fn = hyperparams_builder.build

  return cross_similarity_builder.build(
    argscope_fn,
    cross_similarity_config,
    True)

if __name__ == '__main__':
  linear_cross_similarity_proto = '''
    linear_cross_similarity {
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
  cosine_cross_similarity_proto = '''
    cosine_cross_similarity {
    }
  '''

  cross_similarity = build_cross_similarity(linear_cross_similarity_proto)
