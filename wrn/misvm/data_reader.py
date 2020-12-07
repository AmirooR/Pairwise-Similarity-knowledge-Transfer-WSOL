from rcnn_attention.wrn.dataflow.tensorpack_reader import TensorpackReaderWRN
from rcnn_attention.wrn.dataflow.tensorpack_reader_det import TensorpackReaderDet


def get_data_reader(params):
  reader_names = ['miniimagenet', 'omniglot']
  det_reader_names = ['coco']
  if params.dataset in reader_names:
    # reader.get_data yeilds [feas, fea_boxes, fea_target_classes,
    #                           fea_classes, imgs, target_class]+boxes+classes
    num_sample_classes = 20 if params.b == 10 else 15
    omniglot_prefix = '_no_rot'
    num_sample_classes_min = params.b
    reader = TensorpackReaderWRN(False,
                              shuffle=False,
                              k_shot=params.k,
                              bag_size=params.b,
                              dataset_name=params.dataset,
                              split=params.split,
                              train_num=0,
                              use_features=True,
                              has_single_target=False,
                              num_negative_bags=params.neg,
                              num_sample_classes=num_sample_classes,
                              omniglot_prefix=omniglot_prefix,
                              num_sample_classes_min=num_sample_classes_min,
                              one_example_per_class=False,
                              add_images=True)
    return reader
  elif params.dataset in det_reader_names:
    #reader.get_data returns [feas, fea_boxes, fea_target_classes, 
    #                           fea_classes, imgs, target_class]+boxes+classes
    reader = TensorpackReaderDet(root=params.data_root,
                                 split=params.split,
                                 k_shot=params.k,
                                 shuffle=False,
                                 noise_rate=0.0,
                                 class_agnostic=False,
                                 num_negative_bags=params.neg)
    return reader
  else:
    raise ValueError('Dataset {} is unkown'.format(params.dataset))

