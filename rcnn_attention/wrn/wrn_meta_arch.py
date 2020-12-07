"""WRN meta-architecture definition.
    wide residual network architecture
"""
from abc import abstractmethod
from functools import partial
import tensorflow as tf

from object_detection.anchor_generators import grid_anchor_generator
from object_detection.core import balanced_positive_negative_sampler as sampler
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import box_predictor
from object_detection.core import losses
from object_detection.core import model
from object_detection.core import post_processing
#import post_processing as post_processing_with_resampling
from object_detection.core import standard_fields as fields
from object_detection.core import target_assigner
from object_detection.utils import ops
from object_detection.utils import shape_utils
import rcnn_attention.attention.util as util
#import coloc_target_assigner

from rcnn_attention.wrn.model.wide_resnet import wide_resnet_model
from rcnn_attention.wrn.model.omniglot_model_def import omniglot_model

slim = tf.contrib.slim

class WRNMetaArch(model.DetectionModel):
  """WRN Meta-architecture definition."""

  def __init__(self,
               is_training,
               k_shot,
               bag_size,
               num_negative_bags,
               num_classes,
               wrn_depth,
               wrn_width,
               wrn_dropout_rate,
               wrn_data_format,
               weight_decay,
               attention_tree,
               use_features,
               model_type,
               nms_fn,
               parallel_iterations=16):
    """WRNMetaArch Constructor.

    Args:
      is_training: A boolean indicating whether the training version of the
        computation graph should be constructed.
      num_classes: Number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      parallel_iterations: (Optional) The number of iterations allowed to run
        in parallel for calls to tf.map_fn.
      k_shot:
      bag_size:
    """
    super(WRNMetaArch, self).__init__(num_classes=num_classes)

    self._k_shot = k_shot
    self._is_training = is_training
    self._bag_size = bag_size
    self._num_classes = num_classes
    self._weight_decay = weight_decay
    self._use_features = use_features
    self._net_model = None
    self._model_type = model_type
    self._num_negative_bags = num_negative_bags
    if not use_features:
      if model_type == 'WRN':
        self._net_model = partial(
          wide_resnet_model,
          depth=wrn_depth,
          width_factor=wrn_width,
          training=is_training,
          dropout_rate=wrn_dropout_rate,
          data_format=wrn_data_format
        )
        self._regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
      elif model_type == 'OMNIGLOT':
        self._net_model = partial(
            omniglot_model,
            training=is_training)
        self._regularizer = None
      else:
        raise ValueError('model_type {} is not defined'.format(model_type))

    self._attention_tree = attention_tree
    self._parallel_iterations = parallel_iterations
    self._nms_fn = nms_fn

    self._summary_tensors = self._is_training
    if self._summary_tensors:
      # The average of tensor values for each key would
      # be summerized
      from collections import defaultdict
      self._debug_summary_collection = defaultdict(list)

  @property
  def feature_extractor_scope(self):
    return 'FeatureExtractor'

  @property
  def classifier_scope(self):
    return 'Classifier'

  def preprocess(self, inputs):
    """Feature-extractor specific preprocessing.

    See base class.

    Args:
      inputs: a [batch, height_in, width_in, channels] float tensor representing
        a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: a [batch, height_out, width_out, channels] float
        tensor representing a batch of images.
    Raises:
      ValueError: if inputs tensor does not have type tf.float32
    """
    if inputs.dtype is not tf.float32:
      raise ValueError('`preprocess` expects a tf.float32 tensor')
    return inputs

  def predict(self, preprocessed_inputs):
    """Predicts unpostprocessed tensors from input tensor.

    This function takes an input batch of images and runs it through the
    forward pass of the network to yield "raw" un-postprocessed predictions.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      prediction_dict: a dictionary holding "raw" prediction tensors:
        1) predictor_features: A 4-D float32 tensor with shape
          [batch_size, height, width, depth] to be used for classification
        2) class_predictions: a 3-D tensor with shape
          [batch_size, num_classes] containing class
          predictions (logits) for each of the images.
    """
    image_shape = tf.shape(preprocessed_inputs)
    if self._use_features:
      box_features = preprocessed_inputs
    else:
      # batch*[k_shot+num_negative_bags]*bag_size, 1,1, 640/256/64
      box_features, _ = self._extract_features(preprocessed_inputs)

    box_features_shape = shape_utils.combined_static_and_dynamic_shape(box_features)

    shape = [-1, self._bag_size] + box_features_shape[1:]
    # batch*[k_shot+num_negative_bags], bag_size, 1,1,640
    box_classifier_features = tf.reshape(box_features, shape)

    proposal_matches = None
    proposal_boxes = None
    if fields.BoxListFields.classes in self._groundtruth_lists:
      (groundtruth_boxlists,
        groundtruth_classes_with_background_list
        ) = self._format_groundtruth_data(image_shape,
                                          to_absolute=False)
      # batch*[k_shot+num_negative_bags]*bag_size, 
      proposal_matches = tf.concat(groundtruth_classes_with_background_list, 0)
      _shape = shape_utils.combined_static_and_dynamic_shape(proposal_matches)
      shape  = [-1, self._bag_size] + _shape[1:]
      proposal_matches = tf.reshape(proposal_matches, shape)

    if fields.BoxListFields.boxes in self._groundtruth_lists:
      # batch*[k_shot+num_negative_bags]*bag_size, 4
      proposal_boxes = groundtruth_boxlists #tf.concat(groundtruth_boxlists, 0)
      _shape = shape_utils.combined_static_and_dynamic_shape(proposal_boxes)
      shape = [-1, self._bag_size] + _shape[1:]
      # batch*[k_shot+num_negative_bags], bag_size, 4
      proposal_boxes = tf.reshape(proposal_boxes, shape)

    positive_bags = box_classifier_features
    positive_proposal_matches = proposal_matches
    positive_proposal_boxes   = proposal_boxes
    negative_bags = None
    negative_proposal_matches = None
    negative_proposal_boxes = None
    objectness_score = self._groundtruth_lists.get('objectness_score', None)

    total_bags = self._k_shot + self._num_negative_bags
    if self._num_negative_bags > 0:
      pos_inds = [i for i in range(
                  box_classifier_features.shape[0]
                  ) if i % total_bags < self._k_shot]
      neg_inds = [i for i in range(
                  box_classifier_features.shape[0]
                  ) if i % total_bags >= self._k_shot]
      positive_bags = tf.gather(box_classifier_features, pos_inds)
      positive_proposal_matches = tf.gather(proposal_matches, pos_inds)
      positive_proposal_boxes = tf.gather(proposal_boxes, pos_inds)
      if objectness_score is not None:
        objectness_score = tf.gather(objectness_score, pos_inds)
      negative_bags = tf.gather(box_classifier_features, neg_inds)
      negative_proposal_matches = tf.gather(proposal_matches, neg_inds)
      negative_proposal_boxes = tf.gather(proposal_boxes, neg_inds)
      util.add_extra_tensor('neg_cls_matches', negative_proposal_matches)
      util.add_extra_tensor('pos_cls_matches', positive_proposal_matches)
      util.add_extra_tensor('proposal_boxes', proposal_boxes)
      util.add_extra_tensor('positive_proposal_boxes', positive_proposal_boxes)

    target_class = self._groundtruth_lists.get('target_class', None)

    #objectness_score = tf.zeros(positive_bags.shape.as_list()[:2])
    #from IPython import embed;embed()
    tree_output = self._attention_tree.build(
      positive_bags, matched_class=positive_proposal_matches,
      parallel_iterations=self._parallel_iterations,
      neg_fea=negative_bags, neg_matched_class=negative_proposal_matches,
      fea_boxes=positive_proposal_boxes,
      problem_target_class=target_class,
      objectness_score=objectness_score)

    (cobjs_fea, fg_scores, cobj_cls_scores,
     matched_class, box_classifier_features,
     proposal_boxes_normalized) = tree_output.format_output(
                    box_classifier_features, positive_proposal_boxes)

    self._tree_debug_tensors = partial(self._attention_tree.debug_tensors,
                                       boxes=positive_proposal_boxes,
                                       nms_fn=self._nms_fn)

    prediction_dict = {
        'predictor_features': box_classifier_features,
        'fg_predictions': fg_scores,
        'matched_class': matched_class,
        'image_shape': image_shape,
        'tree_output': tree_output
    }


    if self._summary_tensors:
      self._debug_summary_collection['top_attention_images'] = preprocessed_inputs

    return prediction_dict

  def _extract_features(self, preprocessed_inputs):
    with tf.variable_scope(self.feature_extractor_scope,
                           regularizer=self._regularizer) as var_scope:
      features, activations = self._net_model(preprocessed_inputs)
      if self._model_type == 'WRN':
        features = tf.reduce_mean(activations['post_relu'],
                                  [1,2], keep_dims=True, name='AvgPool')
      elif self._model_type == 'OMNIGLOT':
        features = activations['flattened']
        features = features[:,None,None,:]

    return features, activations

  def _format_groundtruth_data(self, image_shape, to_absolute=True):
    """Helper function for preparing groundtruth data for target assignment.

    In order to be consistent with the model.DetectionModel interface,
    groundtruth boxes are specified in normalized coordinates and classes are
    specified as label indices with no assumed background category.  To prepare
    for target assignment, we:
    1) convert boxes to absolute coordinates,
    2) add a background class at class index 0

    Args:
      image_shape: A 1-D int32 tensor of shape [4] representing the shape of the
        input image batch.

    Returns:
      groundtruth_boxlists: A list of BoxLists containing (absolute) coordinates
        of the groundtruth boxes.
      groundtruth_classes_with_background_list: A list of 2-D one-hot
        (or k-hot) tensors of shape [num_boxes, num_classes+1] containing the
        class targets with the 0th index assumed to map to the background class.
    """
    if to_absolute:
      groundtruth_boxlists = [
        box_list_ops.to_absolute_coordinates(
            box_list.BoxList(boxes), image_shape[1], image_shape[2])
        for boxes in self.groundtruth_lists(fields.BoxListFields.boxes)]
    else:
      groundtruth_boxlists = tf.concat(self.groundtruth_lists(fields.BoxListFields.boxes),0)
    groundtruth_classes_with_background_list = [
        tf.to_float(one_hot_encoding)
        #tf.to_float(
        #    tf.pad(one_hot_encoding, [[0, 0], [1, 0]], mode='CONSTANT'))
        for one_hot_encoding in self.groundtruth_lists(
            fields.BoxListFields.classes)]
    return groundtruth_boxlists, groundtruth_classes_with_background_list

  def loss(self, prediction_dict, scope=None):
    """Compute scalar loss tensors given prediction tensors.

    If first_stage_only=True, only RPN related losses are computed (i.e.,
    `rpn_localization_loss` and `rpn_objectness_loss`).  Otherwise all
    losses are computed.

    Args:
      prediction_dict: a dictionary holding prediction tensors (see the
        documentation for the predict method.  If first_stage_only=True, we
        expect prediction_dict to contain `rpn_box_encodings`,
        `rpn_objectness_predictions_with_background`, `rpn_features_to_crop`,
        `image_shape`, and `anchors` fields.  Otherwise we expect
        prediction_dict to additionally contain `refined_box_encodings`,
        `class_predictions_with_background`, `num_proposals`, and
        `proposal_boxes` fields.
      scope: Optional scope name.

    Returns:
      a dictionary mapping loss keys (`first_stage_localization_loss`,
        `first_stage_objectness_loss`, 'second_stage_localization_loss',
        'second_stage_classification_loss') to scalar tensors representing
        corresponding loss values.
    """
    with tf.name_scope(scope, 'Loss', prediction_dict.values()):
      loss_dict = self._attention_tree.loss()

      # REMOVE MOVING MEAN,etc from REG LOSSES
      rv = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=".*moving.*")
      [tf.get_default_graph()._collections[tf.GraphKeys.REGULARIZATION_LOSSES].remove(_e) for _e in rv]

    return loss_dict

  def postprocess(self, prediction_dict):
    return prediction_dict

  def restore_map(self, from_detection_checkpoint=True,
                  load_all_detection_checkpoint_vars=False):
    """Returns a map of variables to load from a foreign checkpoint.

    See parent class for details.

    Args:
      from_detection_checkpoint: whether to restore from a full detection
        checkpoint (with compatible variable names) or to restore from a
        classification checkpoint for initialization prior to training.

    Returns:
      A dict mapping variable names (to load from a checkpoint) to variables in
      the model graph.
    """
    variables_to_restore = tf.global_variables()
    global_step = slim.get_or_create_global_step()
    #variables_to_restore.append(global_step)
    # Only load feature extractor variables to be consistent with loading from
    # a classification checkpoint.
    include_patterns = None
    if not load_all_detection_checkpoint_vars:
      include_patterns = [
          self.feature_extractor_scope
      ]
    variables_to_restore = tf.contrib.framework.filter_variables(
        variables_to_restore,
        include_patterns=include_patterns,
        exclude_patterns=[global_step.op.name])
    return {var.op.name: var for var in variables_to_restore}
