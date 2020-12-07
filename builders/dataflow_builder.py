from rcnn_attention.dataflow.tensorpack_utils import TensorpackReader, augment_supportset
from rcnn_attention.wrn.dataflow.tensorpack_reader import TensorpackReaderWRN
from rcnn_attention.wrn.dataflow.tensorpack_reader_det import TensorpackReaderDet
from rcnn_attention.coloc import standard_fields as fields
from object_detection.protos import coloc_input_reader_pb2
from tensorpack import *
import tensorflow as tf

def build(dataflow_config, k_shot, is_training, queue_size,
    bag_size=1,
    use_features=False,
    num_negative_bags=0,
    mode=None):
  if not isinstance(dataflow_config, coloc_input_reader_pb2.CoLocInputReader):
    raise ValueError('dataflow_config not of type '
      'coloc_input_reader_pb2.CoLocInputReader.')
  if dataflow_config.WhichOneof('coloc_input_reader_oneof') == 'detection_input_reader':
    return _build_det_reader(dataflow_config.detection_input_reader, k_shot, is_training, queue_size)
  if dataflow_config.WhichOneof('coloc_input_reader_oneof') == 'mil_input_reader':
    return _build_mil_reader(dataflow_config.mil_input_reader, k_shot, is_training, queue_size, bag_size, use_features, num_negative_bags)
  if dataflow_config.WhichOneof('coloc_input_reader_oneof') == 'mil_det_fea_input_reader':
    return _build_mil_det_fea_reader(dataflow_config.mil_det_fea_input_reader, k_shot, is_training, num_negative_bags, queue_size, mode, bag_size)
  raise ValueError('dataflow_config.coloc_input_reader_oneof is unknown.')

def _create_det_tensors(ds, bag_size, fea_dim, im_shape, total_bags, queue_size):
  fea_shape = (bag_size,) + (1,1,fea_dim)
  placeholders = [
      tf.placeholder(tf.float32, shape=(total_bags,)+fea_shape), #feas
      tf.placeholder(tf.float32, shape=(total_bags, fea_shape[0], 4)), #fea_boxes (k,300,4)
      tf.placeholder(tf.float32, shape=(total_bags, fea_shape[0])),  #fea_target_classes (k,300)
      tf.placeholder(tf.float32, shape=(total_bags, fea_shape[0])),  #fea_classes (original)
      tf.placeholder(tf.float32, shape=[total_bags] + im_shape), #imgs
      tf.placeholder(tf.float32, shape=[total_bags, fea_shape[0]]), #objectness
      tf.placeholder(tf.float32, shape=()), #target class
      tf.placeholder(tf.string, shape=(total_bags,))] #img_fns
  placeholders += [tf.placeholder(tf.float32, shape=(None,4)) for _ in range(total_bags)] #gt boxes
  placeholders += [tf.placeholder(tf.float32, shape=(None,)) for _ in range(total_bags)] # gt classes
  queue = tf.FIFOQueue(queue_size, [x.dtype for x in placeholders])
  thread = graph_builder.EnqueueThread(queue, ds, placeholders)
  tensors = queue.dequeue()
  for tensor, placeholder in zip(tensors, placeholders):
    tensor.set_shape(placeholder.shape)

  tensors_dict = {fields.CoLocInputDataFields.supportset: tensors[0],
        fields.CoLocInputDataFields.groundtruth_boxes: tensors[1],
        fields.CoLocInputDataFields.groundtruth_classes: tensors[2],
        fields.CoLocInputDataFields.original_groundtruth_classes: tensors[3],
        fields.CoLocInputDataFields.original_images: tensors[4],
        fields.CoLocInputDataFields.proposal_objectness: tensors[5],
        fields.CoLocInputDataFields.groundtruth_target_class: tensors[6],
        fields.CoLocInputDataFields.filename: tensors[7]}
  s = 8
  for k in range(total_bags):
    tensors_dict[fields.CoLocInputDataFields.groundtruth_image_boxes+'_{}'.format(k)] = tensors[k+s] #gt box k
    tensors_dict[fields.CoLocInputDataFields.groundtruth_image_classes+'_{}'.format(k)] = tensors[k+s+total_bags] #gt class k

  return tensors_dict, thread

def _build_mil_reader(dataflow_config, k_shot, is_training, queue_size, bag_size, use_features, num_negative_bags):
  dataset_name = dataflow_config.dataset_name
  total_bags = k_shot + num_negative_bags
  ds = TensorpackReaderWRN(dataflow_config.is_training,
                           k_shot=k_shot,
                           bag_size=bag_size,
                           data_format=dataflow_config.data_format,
                           has_bag_image_iterator=dataflow_config.has_bag_iterator,
                           shuffle=dataflow_config.shuffle,
                           dataset_name=dataset_name,
                           split=dataflow_config.split,
                           train_num=dataflow_config.train_num,
                           use_features=use_features,
                           has_single_target=dataflow_config.has_single_target,
                           num_negative_bags=num_negative_bags,
                           num_sample_classes=dataflow_config.num_sample_classes,
                           omniglot_prefix=dataflow_config.omniglot_prefix,
                           num_sample_classes_min=dataflow_config.num_sample_classes_min,
                           one_example_per_class=dataflow_config.one_example_per_class,
                           add_images=True) #TODO add to proto (or always true?)

  #ds = augment_wrn(ds, is_training, total_bags, bag_size, dataflow_config.do_pp_mean, dataset_name=dataset_name, use_features=use_features)

  ds.reset_state()
  #d = next(ds.get_data())
  #from IPython import embed;embed()
  if is_training:
    ds = PrefetchDataZMQ(ds, dataflow_config.num_readers)
  else:
    ds = PrefetchDataZMQ(ds,1)
  if dataset_name == 'miniimagenet':
    im_shape = [84*bag_size,84,3]
    fea_dim = 640
  else:
    raise ValueError('dataset {} is not implemented yet for det'.format(dataset_name))

  raise NotImplementedError('I have added objectness to mil_det ds but not TensorpackReaderWRN')
  return _create_det_tensors(ds, bag_size, fea_dim, im_shape, total_bags, queue_size)

#  if dataset_name == 'cifar10':
#    img_shape = (32,32,3) if dataflow_config.data_format == 'channels_last' else (3,32,32)
#  elif dataset_name == 'miniimagenet':
#    if use_features:
#      img_shape = (1,1,640)
#    else:
#      img_shape = (84,84,3)
#  elif dataset_name == 'omniglot' or dataset_name == 'mnist':
#    if use_features:
#      img_shape = (1,1,256)
#    else:
#      img_shape = (28,28,1)
#  else:
#    raise ValueError('dataset_name {} is not valid'.format(dataset_name))

def _build_det_reader(dataflow_config,  k_shot,
    is_training, queue_size):
  im_size = [336, 336]
  if len(dataflow_config.image_size) > 0:
    im_size[0] = int(dataflow_config.image_size[0])
    if len(dataflow_config.image_size) > 1:
      im_size[1] = int(dataflow_config.image_size[1])
  ds = TensorpackReader(dataflow_config.support_db,
      split=dataflow_config.split, k_shot=k_shot,
      shuffle=dataflow_config.shuffle,
      noise_rate=dataflow_config.noise_rate,
      mask_dir=dataflow_config.mask_dir,
      class_agnostic=False) #not is_training)

  ds = augment_supportset(ds, is_training, k_shot, im_size)
  ds.reset_state()

  if is_training:
    ds = PrefetchDataZMQ(ds, dataflow_config.num_readers)
  else:
    ds = PrefetchDataZMQ(ds, 1)

  # ds: a list with 2*k_shot+2 elemets
  placeholders = [tf.placeholder(tf.float32, shape=(k_shot, im_size[0], im_size[1], 3))] # Images
  placeholders += [tf.placeholder(tf.float32, shape=())] # Target Class
  placeholders += [tf.placeholder(tf.float32, shape=(None, 4)) for _ in range(k_shot)] # Boxes
  placeholders += [tf.placeholder(tf.float32, shape=(None,)) for _ in range(k_shot)] # Classes

  queue = tf.FIFOQueue(queue_size, [x.dtype for x in placeholders])
  thread = graph_builder.EnqueueThread(queue, ds, placeholders)
  tensors = queue.dequeue()
  for tensor, placeholder in zip(tensors, placeholders):
    tensor.set_shape(placeholder.shape)

  tensors_dict = {fields.CoLocInputDataFields.supportset: tensors[0],
                  fields.CoLocInputDataFields.groundtruth_target_class: tensors[1]}
  for k in range(k_shot):
    tensors_dict[fields.CoLocInputDataFields.groundtruth_boxes+'_{}'.format(k)] = tensors[k+2]
    tensors_dict[fields.CoLocInputDataFields.groundtruth_classes+'_{}'.format(k)] = tensors[k+2+k_shot]
  return tensors_dict, thread

def _build_mil_det_fea_reader(dataflow_config, k_shot, is_training,
                              num_negative_bags, queue_size, mode, bag_size):
  im_size = [336, 336, 3]
  if len(dataflow_config.image_size) > 0:
    im_size[0] = int(dataflow_config.image_size[0])
    if len(dataflow_config.image_size) > 1:
      im_size[1] = int(dataflow_config.image_size[1])
  ds = TensorpackReaderDet(dataflow_config.support_db,
      split=dataflow_config.split, k_shot=k_shot,
      shuffle=dataflow_config.shuffle,
      num_negative_bags=num_negative_bags,
      noise_rate=dataflow_config.noise_rate,
      class_agnostic=False,
      min_nclasses_in_positive_images=dataflow_config.min_nclasses_in_positive_images,
      smart_neg_bag_sampler=dataflow_config.smart_neg_bag_sampler,
      pos_threshold=dataflow_config.pos_threshold,
      output_im_shape=im_size[:2],
      objectness_key=dataflow_config.objectness_key,
      mode=mode,
      folds=dataflow_config.folds,
      feature_normalization=dataflow_config.feature_normalization,
      feas_key=dataflow_config.feas_key,
      bag_size=bag_size)


  ds.reset_state()

  if is_training:
    ds = PrefetchDataZMQ(ds, dataflow_config.num_readers)
  #else:
  #  ds = PrefetchDataZMQ(ds, 1)

  total_bags = k_shot + num_negative_bags
  return _create_det_tensors(ds, bag_size, dataflow_config.fea_dim, im_size, total_bags, queue_size)


