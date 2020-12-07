#aggragates and creates pkl files
import os,sys
import cPickle as pickle
import numpy as np
import glob
from tqdm import tqdm

root_dir = 'logs/coco/k1/extract300/'
original_ImageSet_dir = 'data/detection-data/det/coco2017/ImageSet/'

input_set_names = ["train_unseen", "val_unseen"]
output_set_name = "unseen"

images = []
rois = []
feas = []
synsets_list = []
for set_name in input_set_names:
  print(set_name)
  with open(os.path.join(original_ImageSet_dir, set_name + '.pkl'), 'rb') as f:
    original_info = pickle.load(f)
  info_dir = os.path.join(root_dir, 'Feas', set_name)
  info_files = glob.glob(info_dir+'/*.pkl')
  synsets = original_info['synsets']
  synsets_list.append(synsets)
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

joined_synsets = synsets_list[0] #they have the same classes
set_info = dict(rois=rois, images=images, feas=feas, synsets=joined_synsets)
with open(os.path.join(root_dir, 'ImageSet', output_set_name + '.pkl'), 'wb') as f:
  pickle.dump(set_info, f)

