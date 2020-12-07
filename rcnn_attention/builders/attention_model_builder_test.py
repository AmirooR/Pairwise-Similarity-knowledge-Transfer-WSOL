# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for object_detection.models.model_builder."""

import tensorflow as tf

from google.protobuf import text_format
from rcnn_attention.builders import attention_model_builder
from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.models import faster_rcnn_inception_resnet_v2_feature_extractor as frcnn_inc_res
from object_detection.models import faster_rcnn_resnet_v1_feature_extractor as frcnn_resnet_v1
from object_detection.protos import attention_model_pb2
from rcnn_attention.attention import rcnn_attention_meta_arch

FEATURE_EXTRACTOR_MAPS = {
    'faster_rcnn_resnet50':
    frcnn_resnet_v1.FasterRCNNResnet50FeatureExtractor,
    'faster_rcnn_resnet101':
    frcnn_resnet_v1.FasterRCNNResnet101FeatureExtractor,
    'faster_rcnn_resnet152':
    frcnn_resnet_v1.FasterRCNNResnet152FeatureExtractor
}


class ModelBuilderTest(tf.test.TestCase):

  def create_model(self, model_config):
    """Builds a AttentionModel based on the model config.

    Args:
      model_config: A attention_model.proto object containing the config for the desired
        AttentionModel.

    Returns:
      AttentionModel based on the config.
    """
    return attention_model_builder.build(model_config, is_training=True)

  def test_create_rcnn_attention_resnet_v1_models_from_config(self):
    model_text_proto = """
      rcnn_attention {
        k_shot: 4
        image_resizer {
          keep_aspect_ratio_resizer {
            min_dimension: 600
            max_dimension: 1024
          }
        }
        feature_extractor {
          type: 'faster_rcnn_resnet101'
        }
        first_stage_anchor_generator {
          grid_anchor_generator {
            scales: [0.25, 0.5, 1.0, 2.0]
            aspect_ratios: [0.5, 1.0, 2.0]
            height_stride: 16
            width_stride: 16
          }
        }
        first_stage_box_predictor_conv_hyperparams {
          regularizer {
            l2_regularizer {
            }
          }
          initializer {
            truncated_normal_initializer {
            }
          }
        }
        initial_crop_size: 14
        maxpool_kernel_size: 2
        maxpool_stride: 2
        second_stage_box_predictor {
          mask_rcnn_box_predictor {
            fc_hyperparams {
              op: FC
              regularizer {
                l2_regularizer {
                }
              }
              initializer {
                truncated_normal_initializer {
                }
              }
            }
          }
        }
        second_stage_post_processing {
          batch_non_max_suppression {
            score_threshold: 0.01
            iou_threshold: 0.6
            max_detections_per_class: 100
            max_total_detections: 300
          }
          score_converter: SOFTMAX
        }
        attention_tree {
          cross_similarity {
            cosine_cross_similarity {
            }
          }

          pre_convline {
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
            filter_num_list: [256]
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
            filter_num_list: [256]
          }
          ncobj_proposals: [32, 32]
          orig_fea_in_post_convline: true
          hard_negative_mining: false
          hard_positive_mining: false
        }
      }"""
    attention_model_proto = attention_model_pb2.AttentionModel()
    text_format.Merge(model_text_proto, attention_model_proto)
    for extractor_type, extractor_class in FEATURE_EXTRACTOR_MAPS.items():
      attention_model_proto.rcnn_attention.feature_extractor.type = extractor_type
      model = attention_model_builder.build(attention_model_proto, is_training=True)
      self.assertIsInstance(model, rcnn_attention_meta_arch.RCNNAttentionMetaArch)
      self.assertIsInstance(model._feature_extractor, extractor_class)

    # build the model
    batch_size = 4
    image_size = 100
    image_shape = (batch_size, image_size, image_size, 3)
    preprocessed_inputs = tf.zeros(image_shape, dtype=tf.float32)
    groundtruth_boxes_list = [
        tf.constant([[0, 0, .5, .5], [.5, .5, 1, 1]], dtype=tf.float32),
        tf.constant([[0, 0, .25, .25], [.25, .25, 1, 1]], dtype=tf.float32),
        tf.constant([[0, .25, .25, 1], [.25, 0, 1, .25]], dtype=tf.float32),
        tf.constant([[0, .5, .5, 1], [.5, 0, 1, .5]], dtype=tf.float32)]
    groundtruth_classes_list = [
        tf.constant([[1], [1]], dtype=tf.float32),
        tf.constant([[1], [1]], dtype=tf.float32),
        tf.constant([[1], [1]], dtype=tf.float32),
        tf.constant([[1], [1]], dtype=tf.float32)]

    model.provide_groundtruth(groundtruth_boxes_list,
                              groundtruth_classes_list)

    result_tensor_dict = model.predict(preprocessed_inputs)

if __name__ == '__main__':
  tf.test.main()
