"""Contains classes specifying naming conventions used for object detection.


Specifies:
  CoLocInputDataFields: standard fields used by reader/preprocessor/batcher.
"""


class CoLocInputDataFields(object):
  """Names for the input tensors.

  Holds the standard data field names to use for identifying input tensors. This
  should be used by the decoder to identify keys for the returned tensor_dict
  containing input tensors. And it should be used by the model to identify the
  tensors it needs.

  Attributes:
    query: query image.
    supportset: support set images.
    query_original_image: image in the original input size.
    key: unique key corresponding to image.
    source_id: source of the original image.
    filename: original filename of the dataset (without common path).
    groundtruth_image_classes: image-level class labels.
    groundtruth_boxes: coordinates of the ground truth boxes in the image.
    groundtruth_classes: box-level class labels.
    groundtruth_label_types: box-level label types (e.g. explicit negative).
    groundtruth_is_crowd: is the groundtruth a single object or a crowd.
    groundtruth_area: area of a groundtruth segment.
    groundtruth_difficult: is a `difficult` object
    proposal_boxes: coordinates of object proposal boxes.
    proposal_objectness: objectness score of each proposal.
    groundtruth_instance_masks: ground truth instance masks.
    groundtruth_instance_classes: instance mask-level class labels.
    groundtruth_keypoints: ground truth keypoints.
    groundtruth_keypoint_visibilities: ground truth keypoint visibilities.
    groundtruth_label_scores: groundtruth label scores.
  """
  query = 'query'
  supportset = 'supportset'
  original_query = 'original_query'
  key = 'key'
  source_id = 'source_id'
  filename = 'filename'
  groundtruth_image_classes = 'groundtruth_image_classes'
  groundtruth_boxes = 'groundtruth_boxes'
  groundtruth_classes = 'groundtruth_classes'
  groundtruth_target_class = 'groundtruth_target_class'
  original_groundtruth_classes = 'original_groundtruth_classes'
  groundtruth_label_types = 'groundtruth_label_types'
  groundtruth_is_crowd = 'groundtruth_is_crowd'
  groundtruth_area = 'groundtruth_area'
  groundtruth_difficult = 'groundtruth_difficult'
  proposal_boxes = 'proposal_boxes'
  proposal_objectness = 'proposal_objectness'
  groundtruth_instance_masks = 'groundtruth_instance_masks'
  groundtruth_instance_classes = 'groundtruth_instance_classes'
  groundtruth_keypoints = 'groundtruth_keypoints'
  groundtruth_keypoint_visibilities = 'groundtruth_keypoint_visibilities'
  groundtruth_label_scores = 'groundtruth_label_scores'
  original_images = 'original_images'
  #groundtruth_image_classes = 'groundtruth_image_classes'
  groundtruth_image_boxes = 'groundtruth_image_boxes'
