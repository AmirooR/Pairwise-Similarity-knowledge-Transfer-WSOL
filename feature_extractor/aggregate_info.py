#aggragates and creates pkl files
import os,sys
import cPickle as pickle
import numpy as np
import glob
from tqdm import tqdm

dataset = 'imagenet' #'coco' , or 'imagenet' or 'INiter2049'
#logs/imagenet/resnet_source_ex512_box_predictor/extract300_152k/
root_dir = 'logs/{}/resnet338k/extract300_338k/'.format(dataset)
original_ImageSet_dir = {
    'coco':'data/detection-data/det/coco2017/ImageSet/',
    'imagenet':'data/detection-data/det/ILSVRC2013_alex/ImageSet/',
    #'imagenet':'data/detection-data/det/ILSVRC2014_processed/ImageSet/',
    'INiter2049':'data/detection-data/det/ILSVRC2014_processed/ImageSet/'}

set_names = os.listdir(os.path.join(root_dir, 'Feas'))

for set_name in set_names:
  print(set_name)
  with open(os.path.join(original_ImageSet_dir[dataset],
                         set_name + '.pkl'), 'rb') as f:
    original_info = pickle.load(f)
  info_dir = os.path.join(root_dir, 'Feas', set_name)
  info_files = glob.glob(info_dir+'/*.pkl')
  synsets = original_info['synsets']
  images = []
  rois = []
  feas = []
  for info_file in tqdm(info_files):
    with open(info_file, 'rb') as f:
      img_info = pickle.load(f)
    images.append(set_name+'/'+img_info['name']+'.jpg')
    feas.append(set_name+'/'+img_info['name']+'.npy')
    roi = {'classes': img_info['gt_classes'].astype(np.uint32),
           'boxes': img_info['gt_boxes'],
           'fea_boxes': img_info['fea_boxes'],
           'fea_classes': img_info['fea_classes'].astype(np.uint32)}
    rois.append(roi)
  set_info = dict(rois=rois, images=images, feas=feas, synsets=synsets)
  with open(os.path.join(root_dir, 'ImageSet', set_name + '.pkl'), 'wb') as f:
    pickle.dump(set_info, f)

