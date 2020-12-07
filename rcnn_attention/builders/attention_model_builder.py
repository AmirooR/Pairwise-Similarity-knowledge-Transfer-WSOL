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

"""A function to build a AttentionModel from configuration."""
from object_detection.builders import anchor_generator_builder
from object_detection.builders import box_coder_builder
#from object_detection.builders import box_predictor_builder
from object_detection.builders import hyperparams_builder
from object_detection.builders import image_resizer_builder
from object_detection.builders import losses_builder
from object_detection.builders import matcher_builder
from object_detection.builders import post_processing_builder
from object_detection.builders import region_similarity_calculator_builder as sim_calc
from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.core import box_predictor
from rcnn_attention.attention import rcnn_attention_meta_arch
from object_detection.models import faster_rcnn_inception_resnet_v2_feature_extractor as frcnn_inc_res
from object_detection.models import faster_rcnn_resnet_v1_feature_extractor as frcnn_resnet_v1
from object_detection.protos import attention_model_pb2
from object_detection.builders.model_builder import _build_faster_rcnn_feature_extractor
from rcnn_attention.builders import attention_tree_builder
from rcnn_attention.builders import convline_builder
from rcnn_attention.wrn import wrn_meta_arch
from rcnn_attention.attention import rcnn_attention_box_predictor
# A map of names to Faster R-CNN feature extractors.
FASTER_RCNN_FEATURE_EXTRACTOR_CLASS_MAP = {
    'faster_rcnn_resnet50':
    frcnn_resnet_v1.FasterRCNNResnet50FeatureExtractor,
    'faster_rcnn_resnet101':
    frcnn_resnet_v1.FasterRCNNResnet101FeatureExtractor,
    'faster_rcnn_resnet152':
    frcnn_resnet_v1.FasterRCNNResnet152FeatureExtractor,
    'faster_rcnn_inception_resnet_v2':
    frcnn_inc_res.FasterRCNNInceptionResnetV2FeatureExtractor
}

def build_box_predictor(argscope_fn, box_predictor_config, is_training, num_classes):
  box_predictor_oneof = box_predictor_config.WhichOneof('box_predictor_oneof')
  if box_predictor_oneof == 'mask_rcnn_box_predictor':
    mask_rcnn_box_predictor = box_predictor_config.mask_rcnn_box_predictor
    fc_hyperparams = argscope_fn(mask_rcnn_box_predictor.fc_hyperparams,
                                 is_training)
    conv_hyperparams = None
    if mask_rcnn_box_predictor.HasField('conv_hyperparams'):
      conv_hyperparams = argscope_fn(mask_rcnn_box_predictor.conv_hyperparams,
                                     is_training)
    box_predictor_object = rcnn_attention_box_predictor.RCNNAttentionBoxPredictor(
        is_training=is_training,
        num_classes=num_classes,
        fc_hyperparams=fc_hyperparams,
        use_dropout=mask_rcnn_box_predictor.use_dropout,
        dropout_keep_prob=mask_rcnn_box_predictor.dropout_keep_probability,
        box_code_size=mask_rcnn_box_predictor.box_code_size,
        conv_hyperparams=conv_hyperparams,
        predict_instance_masks=mask_rcnn_box_predictor.predict_instance_masks,
        mask_prediction_conv_depth=(mask_rcnn_box_predictor.
                                    mask_prediction_conv_depth),
        predict_keypoints=mask_rcnn_box_predictor.predict_keypoints)
    return box_predictor_object

  raise ValueError('Unknown box predictor: {}'.format(box_predictor_oneof))

def build(attention_model_config, k_shot, is_training, is_calibration=False):
  """Builds a AttentionModel based on the model config.

  Args:
    attention_model_config: A model.proto object containing the config for the desired
      AttentionModel.
    is_training: True if this model is being built for training purposes.

  Returns:
    AttentionModel based on the config.

  Raises:
    ValueError: On invalid meta architecture or model.
  """
  if not isinstance(attention_model_config, attention_model_pb2.AttentionModel):
    raise ValueError('attention_model_config not of type attention_model_pb2.AttentionModel.')
  meta_architecture = attention_model_config.WhichOneof('model')
  if meta_architecture == 'rcnn_attention':
    return _build_rcnn_attention_model(attention_model_config.rcnn_attention, is_training, is_calibration)
  if meta_architecture == 'wrn_attention':
    return _build_wrn_attention_model(attention_model_config.wrn_attention, k_shot, is_training, is_calibration)
  raise ValueError('Unknown meta architecture: {}'.format(meta_architecture))

def _build_wrn_attention_model(wrn_config, k_shot, is_training, is_calibration):
  """Builds a WRN attention model based on the model config"""

  attention_tree = None
  if wrn_config.HasField('attention_tree'):
    attention_tree = attention_tree_builder.build(
          hyperparams_builder.build,
          wrn_config.attention_tree,
          k_shot,
          wrn_config.num_classes,
          wrn_config.num_negative_bags,
          is_training,
          is_calibration)

  nms_fn = None
  if wrn_config.HasField('batch_non_max_suppression'):
    nms_fn = post_processing_builder._build_non_max_suppressor(
                          wrn_config.batch_non_max_suppression)
  return wrn_meta_arch.WRNMetaArch(is_training,
                                   k_shot            = k_shot,
                                   bag_size          = wrn_config.bag_size,
                                   num_negative_bags = wrn_config.num_negative_bags,
                                   num_classes       = wrn_config.num_classes,
                                   wrn_depth         = wrn_config.wrn_depth,
                                   wrn_width         = wrn_config.wrn_width,
                                   wrn_dropout_rate  = wrn_config.wrn_dropout_rate,
                                   wrn_data_format   = wrn_config.wrn_data_format,
                                   weight_decay      = wrn_config.weight_decay,
                                   attention_tree    = attention_tree,
                                   use_features      = wrn_config.use_features,
                                   model_type        = wrn_config.ModelType.Name(wrn_config.model_type),
                                   nms_fn            = nms_fn)

def _build_rcnn_attention_model(rcnna_config, is_training, is_calibration):
  """Builds a R-CNN attention model based on the model config.

  Args:
    rcnna_config: A rcnn_attention.proto object containing the config for the
    desired RCNNAttention model.
    is_training: True if this model is being built for training purposes.

  Returns:
    RCNNAttentionMetaArch based on the config.
  Raises:
    ValueError: If rcnna_config.type is not recognized (i.e. not registered in
      model_class_map).
  """
  num_classes = rcnna_config.num_classes
  k_shot = rcnna_config.k_shot
  image_resizer_fn = image_resizer_builder.build(rcnna_config.image_resizer)

  feature_extractor = _build_faster_rcnn_feature_extractor(
      rcnna_config.feature_extractor, is_training)

  first_stage_only = rcnna_config.first_stage_only
  first_stage_anchor_generator = anchor_generator_builder.build(
      rcnna_config.first_stage_anchor_generator)

  first_stage_atrous_rate = rcnna_config.first_stage_atrous_rate
  first_stage_box_predictor_arg_scope = hyperparams_builder.build(
      rcnna_config.first_stage_box_predictor_conv_hyperparams, is_training)
  first_stage_box_predictor_kernel_size = (
      rcnna_config.first_stage_box_predictor_kernel_size)
  first_stage_box_predictor_depth = rcnna_config.first_stage_box_predictor_depth
  first_stage_minibatch_size = rcnna_config.first_stage_minibatch_size
  first_stage_positive_balance_fraction = (
      rcnna_config.first_stage_positive_balance_fraction)
  first_stage_nms_score_threshold = rcnna_config.first_stage_nms_score_threshold
  first_stage_nms_iou_threshold = rcnna_config.first_stage_nms_iou_threshold
  first_stage_max_proposals = rcnna_config.first_stage_max_proposals
  first_stage_loc_loss_weight = (
      rcnna_config.first_stage_localization_loss_weight)
  first_stage_obj_loss_weight = rcnna_config.first_stage_objectness_loss_weight

  initial_crop_size = rcnna_config.initial_crop_size
  maxpool_kernel_size = rcnna_config.maxpool_kernel_size
  maxpool_stride = rcnna_config.maxpool_stride

  second_stage_box_predictor = build_box_predictor(
      hyperparams_builder.build,
      rcnna_config.second_stage_box_predictor,
      is_training=is_training,
      num_classes=num_classes)
  second_stage_batch_size = rcnna_config.second_stage_batch_size
  second_stage_balance_fraction = rcnna_config.second_stage_balance_fraction
  (second_stage_non_max_suppression_fn, second_stage_score_conversion_fn
  ) = post_processing_builder.build(rcnna_config.second_stage_post_processing)
  second_stage_localization_loss_weight = (
      rcnna_config.second_stage_localization_loss_weight)
  second_stage_classification_loss_weight = (
      rcnna_config.second_stage_classification_loss_weight)

  hard_example_miner = None
  if rcnna_config.HasField('hard_example_miner'):
    hard_example_miner = losses_builder.build_hard_example_miner(
        rcnna_config.hard_example_miner,
        second_stage_classification_loss_weight,
        second_stage_localization_loss_weight)

  attention_tree = None
  if rcnna_config.HasField('attention_tree'):
    attention_tree = attention_tree_builder.build(
          hyperparams_builder.build,
          rcnna_config.attention_tree,
          rcnna_config.k_shot,
          num_classes,
          rcnna_config.num_negative_bags,
          is_training,
          is_calibration)

  second_stage_convline = None
  if rcnna_config.HasField('second_stage_convline'):
    second_stage_convline = convline_builder.build(hyperparams_builder.build,
                        None, rcnna_config.second_stage_convline, is_training)

  common_kwargs = {
      'is_training': is_training,
      'image_resizer_fn': image_resizer_fn,
      'feature_extractor': feature_extractor,
      'first_stage_only': first_stage_only,
      'first_stage_anchor_generator': first_stage_anchor_generator,
      'first_stage_atrous_rate': first_stage_atrous_rate,
      'first_stage_box_predictor_arg_scope':
      first_stage_box_predictor_arg_scope,
      'first_stage_box_predictor_kernel_size':
      first_stage_box_predictor_kernel_size,
      'first_stage_box_predictor_depth': first_stage_box_predictor_depth,
      'first_stage_minibatch_size': first_stage_minibatch_size,
      'first_stage_positive_balance_fraction':
      first_stage_positive_balance_fraction,
      'first_stage_nms_score_threshold': first_stage_nms_score_threshold,
      'first_stage_nms_iou_threshold': first_stage_nms_iou_threshold,
      'first_stage_max_proposals': first_stage_max_proposals,
      'first_stage_localization_loss_weight': first_stage_loc_loss_weight,
      'first_stage_objectness_loss_weight': first_stage_obj_loss_weight,
      'second_stage_batch_size': second_stage_batch_size,
      'second_stage_balance_fraction': second_stage_balance_fraction,
      'second_stage_non_max_suppression_fn':
      second_stage_non_max_suppression_fn,
      'second_stage_score_conversion_fn': second_stage_score_conversion_fn,
      'second_stage_localization_loss_weight':
      second_stage_localization_loss_weight,
      'second_stage_classification_loss_weight':
      second_stage_classification_loss_weight,
      'hard_example_miner': hard_example_miner,
      'initial_crop_size': initial_crop_size,
      'maxpool_kernel_size': maxpool_kernel_size,
      'maxpool_stride': maxpool_stride,
      'second_stage_mask_rcnn_box_predictor':second_stage_box_predictor,
      'num_classes': num_classes}


  if isinstance(second_stage_box_predictor, box_predictor.RfcnBoxPredictor):
    raise ValueError('RFCNBoxPredictor is not supported.')
  elif rcnna_config.build_faster_rcnn_arch:
    model = faster_rcnn_meta_arch.FasterRCNNMetaArch(
                **common_kwargs)
    model._k_shot = k_shot
    model._tree_debug_tensors = lambda: {}
    return model
  else:
    return rcnn_attention_meta_arch.RCNNAttentionMetaArch(
        k_shot=k_shot,
        attention_tree=attention_tree,
        second_stage_convline=second_stage_convline,
        attention_tree_only=rcnna_config.attention_tree_only,
        add_gt_boxes_to_rpn=rcnna_config.add_gt_boxes_to_rpn,
        **common_kwargs)
