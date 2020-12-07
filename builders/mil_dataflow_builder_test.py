from rcnn_attention.coloc import standard_fields as fields
from google.protobuf import text_format
from rcnn_attention.builders import dataflow_builder
from object_detection.protos import coloc_input_reader_pb2
import os
import os.path as osp
import shutil
import cv2
import tensorflow as tf
import numpy as np

def build_dataflow(text_proto, k_shot, bag_size):
  dataflow_config = coloc_input_reader_pb2.CoLocInputReader()
  text_format.Merge(text_proto, dataflow_config)
  return dataflow_builder.build(
    dataflow_config,
    k_shot, is_training=False,
    queue_size=50,
    bag_size=bag_size)

def draw_anns(img, boxes, tags, image_tag=None):
  ps = 14
  h, w = img.shape[:2]
  img = np.pad(img, ((ps,ps),(ps,ps),(0,0)), 'constant')
  for box, tag in zip(boxes, tags):
    color = (0, 0, 0)
    y1 = int(box[0]*h + ps)
    x1 = int(box[1]*w + ps)
    y2 = int(box[2]*h + ps)
    x2 = int(box[3]*w + ps)
    cv2.putText(img, str(tag), (x1,y1),
                cv2.FONT_HERSHEY_SIMPLEX, .5, color)

    cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
  if image_tag is not None:
    color = (0,255,0)
    cv2.putText(img, image_tag, (10,10),
        cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)
  return img

def viz(path, supportset, boxes, classes, k_shot, bag_size):
  for k in range(k_shot*bag_size):
    support_img = supportset[k][..., ::-1]
    support_img = 255*(support_img - support_img.min())/(support_img.max()-support_img.min())
    img_boxes = boxes[k]
    img_classes = [int(x) for x in classes[k]]
    support_img = draw_anns(support_img, img_boxes, img_classes)
    cv2.imwrite(osp.join(path, 'image_' + str(k) +'.jpg'), support_img)

if __name__ == '__main__':
  dataflow_proto = '''
  mil_input_reader {
    shuffle: true
    num_readers: 1
  }
  '''
  #"/media/amir/9376dfff-cfcf-48e6-8928-9ffb94c56098/detection-data/det/ILSVRC2014_processed"
  #support_db: "/home/amir/detection-data/det/ILSVRC2014_censored_processed"
  #mask_dir: "train_censor_masks"


  k_shot = 4
  bag_size = 3
  tensors_dict, thread = build_dataflow(dataflow_proto, k_shot, bag_size)

  viz_path = './mil_dataflow_builder_test'
  if osp.exists(viz_path):
    shutil.rmtree(viz_path)

  os.mkdir(viz_path)
  iters = 300

  with tf.Session() as sess:
    with sess.as_default():
      thread.start()
    for i in range(iters):
      print('Step #%d/%d' % (i, iters))
      arrs_dict = sess.run(tensors_dict)
      supportset = arrs_dict[fields.CoLocInputDataFields.supportset]
      boxes = arrs_dict[fields.CoLocInputDataFields.groundtruth_boxes]
      classes = arrs_dict[fields.CoLocInputDataFields.groundtruth_classes]
      supportset = supportset.reshape((-1,)+supportset.shape[2:])
      boxes = boxes.reshape((-1,) + boxes.shape[2:])
      classes = classes.reshape((-1,) + classes.shape[2:])
      boxes = np.split(boxes, boxes.shape[0])
      classes = np.split(classes, classes.shape[0])
      if i % 10 == 0:
        npath = osp.join(viz_path, 'index_' + str(i))
        os.mkdir(npath)
        viz(npath, supportset,
            boxes, classes, k_shot, bag_size)
