from google.protobuf import text_format
from object_detection.builders import hyperparams_builder
from object_detection.protos import attention_tree_pb2
from rcnn_attention.builders import attention_tree_builder
import tensorflow as tf
import numpy as np
from IPython import embed

def build_attention_tree(text_proto, k_shot, num_classes, is_training):
  attention_tree_config = attention_tree_pb2.AttentionTree()
  text_format.Merge(text_proto, attention_tree_config)
  argscope_fn = hyperparams_builder.build

  return attention_tree_builder.build(
    argscope_fn,
    attention_tree_config,
    k_shot,
    num_classes,
    is_training)

def test_attention_tree(attention_tree_proto):
  meta_batch_size = 3
  k_shot = 8
  is_training = True
  nproposals = 10
  num_classes = 5

  attention_tree = build_attention_tree(attention_tree_proto, k_shot, num_classes, is_training)

  features = tf.random_normal((meta_batch_size*k_shot, nproposals, 1, 1, 8))
  scores = tf.random_uniform((meta_batch_size*k_shot, nproposals))
  proposals = tf.random_uniform((meta_batch_size*k_shot, nproposals, 4))

  if is_training:
    match = tf.constant(np.random.uniform(
      size=(meta_batch_size*k_shot, nproposals)) < .5, dtype=tf.bool)
    match = tf.one_hot(tf.to_int32(match), num_classes+1)
  else:
    match = None

  res = attention_tree.build(features, match, 4)

  with tf.Session() as sess:
    embed()



if __name__ == '__main__':
  attention_tree_proto = '''
    unit {
      cross_similarity {
        deep_cross_similarity {
          fc_hyperparameters {
            op: FC
            regularizer {
              l2_regularizer {
                weight: 0.0
              }
            }
            initializer {
              variance_scaling_initializer {
                factor: 1.0
                uniform: true
                mode: FAN_AVG
              }
            }
          }
        }
        loss: "softmax_cross_entropy"
      }
      post_convline {
        kernel_size: 1
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
        filter_num_list: [3, 3, 3]
      }
      ncobj_proposals: 8
      training_ncobj_proposals: 4
      orig_fea_in_post_convline: true
      use_tanh_sigmoid_in_post_convline: false
    }
    unit {
      cross_similarity {
        deep_cross_similarity {
          fc_hyperparameters {
            op: FC
            regularizer {
              l2_regularizer {
                weight: 0.0
              }
            }
            initializer {
              variance_scaling_initializer {
                factor: 1.0
                uniform: true
                mode: FAN_AVG
              }
            }
          }
        }
        loss: "softmax_cross_entropy"
      }
      post_convline {
        kernel_size: 1
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
        filter_num_list: [3, 3, 3]
      }
      ncobj_proposals: 8
      training_ncobj_proposals: 4
      orig_fea_in_post_convline: true
      use_tanh_sigmoid_in_post_convline: false
    }
  '''
  test_attention_tree(attention_tree_proto)
