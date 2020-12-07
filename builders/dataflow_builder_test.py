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

def build_dataflow(text_proto, k_shot):
  dataflow_config = coloc_input_reader_pb2.CoLocInputReader()
  text_format.Merge(text_proto, dataflow_config)
  return dataflow_builder.build(
    dataflow_config,
    k_shot, is_training=False,
    queue_size=50)

def draw_anns(img, boxes, tags, image_tag=None):
  ps = 50
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

def viz(path, supportset, boxes, classes, k_shot):
  for k in range(k_shot):
    support_img = supportset[k][..., ::-1]
    img_boxes = boxes[k]
    img_classes = [int(x) for x in classes[k]]
    support_img = draw_anns(support_img, img_boxes, img_classes)
    cv2.imwrite(osp.join(path, 'image_' + str(k) +'.jpg'), support_img)

if __name__ == '__main__':
  dataflow_proto = '''
  detection_input_reader {
    shuffle: false
    num_readers: 1
    split: "iccv"
    support_db: "../data/detection-data/det/ILSVRC2014_processed/"
    support_db_name: "iccv"
    label_map_path: "./data/detection-data/det/ILSVRC2014_processed/ImageSet/imagenet_labelmap.pbtxt"
  }
  '''

  k_shot = 1
  tensors_dict, thread = build_dataflow(dataflow_proto, k_shot)

  viz_path = './dataflow_builder_test'
  if osp.exists(viz_path):
    shutil.rmtree(viz_path)

  os.mkdir(viz_path)
  iters = 3000

  with tf.Session() as sess:
    with sess.as_default():
      thread.start()
    for i in range(iters):
      print('Step #%d/%d' % (i, iters))
      arrs_dict = sess.run(tensors_dict)
      #from IPython import embed;embed()
      supportset = arrs_dict[fields.CoLocInputDataFields.supportset]
      boxes = []
      classes = []
      for k in range(k_shot):
        n = fields.CoLocInputDataFields.groundtruth_classes + '_{}'
        m = fields.CoLocInputDataFields.groundtruth_boxes + '_{}'
        classes.append(arrs_dict[n.format(k)])
        boxes.append(arrs_dict[m.format(k)])
      if i % 10 == 0:
        npath = osp.join(viz_path, 'index_' + str(i))
        os.mkdir(npath)
        viz(npath, supportset,
            boxes, classes, k_shot)
