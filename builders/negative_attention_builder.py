from rcnn_attention.builders import convline_builder
from object_detection.protos import negative_attention_pb2
from functools import partial
import tensorflow as tf
from object_detection.utils import shape_utils
from rcnn_attention.attention.negative_attention import NegativeAttention

slim = tf.contrib.slim

def build(argscope_fn, negative_attention_config, is_training):
  convline = None
  if negative_attention_config.HasField('convline'):
    convline = convline_builder.build(argscope_fn, None,
                                      negative_attention_config.convline,
                                      is_training)

  return NegativeAttention(convline=convline,
      concat_type=negative_attention_config.ConcatType.Name(
        negative_attention_config.concat_type
      ),
      similarity_type=negative_attention_config.SimilarityType.Name(
        negative_attention_config.similarity_type
      ),
      use_gt_labels=negative_attention_config.use_gt_labels,
      add_loss=negative_attention_config.add_loss)
