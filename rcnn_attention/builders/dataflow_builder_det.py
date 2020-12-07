from rcnn_attention.dataflow.tensorpack_utils import TensorpackReader, augment_supportset
from rcnn_attention.wrn.dataflow.tensorpack_reader import TensorpackReaderWRN, augment_wrn
from rcnn_attention.wrn.dataflow.tensorpack_reader_det import TensorpackReaderDet
from rcnn_attention.coloc import standard_fields as fields
from object_detection.protos import coloc_input_reader_pb2
from tensorpack import *
import tensorflow as tf

def build(dataflow_config, k_shot, is_training, queue_size,
    bag_size=1,
    use_features=False,
    num_negative_bags=0):
  if not isinstance(dataflow_config, coloc_input_reader_pb2.CoLocInputReader):
    raise ValueError('dataflow_config not of type '
      'coloc_input_reader_pb2.CoLocInputReader.')
  if dataflow_config.WhichOneof('coloc_input_reader_oneof') == 'detection_input_reader':
    return _build_det_reader(dataflow_config.detection_input_reader, k_shot, is_training, queue_size)
  if dataflow_config.WhichOneof('coloc_input_reader_oneof') == 'mil_input_reader':
    return _build_mil_reader(dataflow_config.mil_input_reader, k_shot, is_training, queue_size, bag_size, use_features, num_negative_bags)
  if dataflow_config.WhichOneof('coloc_input_reader_oneof') == 'mil_det_fea_input_reader':
    return _build_mil_det_fea_reader(dataflow_config.mil_det_fea_input_reader, k_shot, is_trainging, num_negative_bags, queue_size)
  raise ValueError('dataflow_config.coloc_input_reader_oneof is unknown.')

def _build_mil_reader(dataflow_config, k_shot, is_training, queue_size, bag_size, use_features, num_negative_bags):
  dataset_name = dataflow_config.dataset_name
  k_shot = k_shot + num_negative_bags
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
                               one_example_per_class=dataflow_config.one_example_per_class)
  ds = augment_wrn(ds, is_training, k_shot, bag_size, dataflow_config.do_pp_mean, dataflow_config.add_background, dataset_name=dataset_name, use_features=use_features)
  ds.reset_state()
  if is_training:
    ds = PrefetchDataZMQ(ds, dataflow_config.num_readers)
  else:
    ds = PrefetchDataZMQ(ds,1)
  if dataset_name == 'cifar10':
    img_shape = (32,32,3) if dataflow_config.data_format == 'channels_last' else (3,32,32)
  elif dataset_name == 'miniimagenet':
    if use_features:
      img_shape = (1,1,640)
    else:
      img_shape = (84,84,3)
  elif dataset_name == 'omniglot' or dataset_name == 'mnist':
    if use_features:
      img_shape = (1,1,256)
    else:
      img_shape = (28,28,1)
  else:
    raise ValueError('dataset_name {} is not valid'.format(dataset_name))
  placeholders = [tf.placeholder(tf.float32, shape=(k_shot*bag_size,)+img_shape)]
  placeholders += [tf.placeholder(tf.float32, shape=(1,4)) for _ in range(k_shot*bag_size)]
  placeholders += [tf.placeholder(tf.float32, shape=(1,)) for _ in range(k_shot*bag_size)]
  placeholders += [tf.placeholder(tf.float32, shape=(1,)) for _ in range(k_shot*bag_size)]

  queue = tf.FIFOQueue(queue_size, [x.dtype for x in placeholders])
  thread = graph_builder.EnqueueThread(queue, ds, placeholders)
  tensors = queue.dequeue()
  for tensor, placeholder in zip(tensors, placeholders):
    tensor.set_shape(placeholder.shape)

  tensors_dict = {fields.CoLocInputDataFields.supportset: tensors[0]}
  for k in range(k_shot*bag_size):
    bind = k+1
    cind = bind+k_shot*bag_size
    oind = bind+2*k_shot*bag_size
    tensors_dict[fields.CoLocInputDataFields.groundtruth_boxes+'_{}'.format(k)] = tensors[bind]
    tensors_dict[fields.CoLocInputDataFields.groundtruth_classes+'_{}'.format(k)] = tensors[cind]
    tensors_dict[fields.CoLocInputDataFields.original_groundtruth_classes+'_{}'.format(k)] = tensors[oind]
  return tensors_dict, thread


def _build_det_reader(dataflow_config,  k_shot,
    is_training, queue_size):
  ds = TensorpackReader(dataflow_config.support_db,
      split=dataflow_config.split, k_shot=k_shot,
      shuffle=dataflow_config.shuffle,
      noise_rate=dataflow_config.noise_rate,
      mask_dir=dataflow_config.mask_dir,
      class_agnostic=False) #not is_training)

  im_size = (336,336)
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

def _build_mil_det_fea_reader(dataflow_config, k_shot, is_trainging, num_negative_bags, queue_size):
  ds = TensorpackReaderDet(dataflow_config.support_db,
      split=dataflow_config.split, k_shot=k_shot,
      shuffle=dataflow_config.shuffle,
      noise_rate=dataflow_config.noise_rate,
      mask_dir=dataflow_config.mask_dir,
      class_agnostic=False,
      min_nclasses_in_positive_images=dataflow_config.min_nclasses_in_positive_images)

  fea_shape = (300,1,1,640)
  im_shape = (336,336,3)
  bag_size = fea_shape[0]
  #ds = augment_supportset(ds, is_training, k_shot, im_size)
  ds.reset_state()

  if is_training:
    ds = PrefetchDataZMQ(ds, dataflow_config.num_readers)
  else:
    ds = PrefetchDataZMQ(ds, 1)

  total_bags = k_shot + num_negative_bags

  placeholders = [tf.placeholder(tf.float32, shape=(total_bags*bag_size,)+fea_shape[1:])] #feas
  placeholders += [tf.placeholder(tf.float32, shape=(1,4)) for _ in range(total_bags*bag_size)] #fea_boxes
  placeholders += [tf.placeholder(tf.float32, shape=(1,)) for _ in range(total_bags*bag_size)]  #fea_target_classes
  placeholders += [tf.placeholder(tf.float32, shape=(1,)) for _ in range(total_bags*bag_size)]  #fea_classes (original)
  placeholders += [tf.placeholder(tf.float32, shape=(total_bags,) + im_shape)] #imgs
  placeholders += [tf.placeholder(tf.float32, shape=())] #taget class
  placeholders += [tf.placeholder(tf.float32, shape=(None,4)) for _ in range(total_bags)] #gt boxes
  placeholders += [tf.placeholder(tf.float32, shape=(None,) for _ in range(total_bags)] # gt classes

  queue = tf.FIFOQueue(queue_size, [x.dtype for x in placeholders])
  thread = graph_builder.EnqueueThread(queue, ds, placeholders)
  tensors = queue.dequeue()
  for tensor, placeholder in zip(tensors, placeholders):
    tensor.set_shape(placeholder.shape)

  tensors_dict = {fields.CoLocInputDataFields.supportset: tensors[0]}
  for k in range(total_bags*bag_size):
    bind = k+1
    cind = bind+total_bags*bag_size
    oind = bind+2*k_shot*bag_size
    tensors_dict[fields.CoLocInputDataFields.groundtruth_boxes+'_{}'.format(k)] = tensors[bind]
    tensors_dict[fields.CoLocInputDataFields.groundtruth_classes+'_{}'.format(k)] = tensors[cind]
    tensors_dict[fields.CoLocInputDataFields.original_groundtruth_classes+'_{}'.format(k)] = tensors[oind]

  s = 3*total_bags*bag_size + 1
  tensor_dict[fields.CoLocInputDataFields.original_images] = tensors[s] #imgs
  tensor_dict[fields.CoLocInputDataFields.groundtruth_target_class] = tensors[s+1] #target class
  for k in range(k_shot):
    tensor_dict[fields.CoLocInputDataFields.groundtruth_image_boxes+'_{}'.format(k)] = tensors[k+s+2] #gt box k
    tensor_dict[fields.CoLocInputDataFields.groundtruth_image_classes+'_{}'.format(k)] = tensors[k+s+2+k_shot] #gt class k

  return tensors_dict, thread
