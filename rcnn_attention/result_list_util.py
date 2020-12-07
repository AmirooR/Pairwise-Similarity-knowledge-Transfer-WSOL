from object_detection.utils.np_box_list_ops import (non_max_suppression, concatenate,
                                                    filter_scores_greater_than,
                                                    sort_by_field, gather)
from object_detection.utils.np_box_list import BoxList
import numpy as np
import copy
from object_detection.utils.np_box_ops import iou as np_iou
from multiprocessing import Pool
from functools import partial

def multi_class_non_max_suppression(boxlist, score_thresh, iou_thresh,
                                    max_output_size):
  if not 0 <= iou_thresh <= 1.0:
    raise ValueError('thresh must be between 0 and 1')
  scores = boxlist.get_field('scores')
  classes = boxlist.get_field('classes')
  num_boxes = boxlist.num_boxes()

  selected_boxes_list = []
  for cls in np.unique(classes):
    inds = np.where(cls == classes)[0]
    subboxlist = gather(boxlist, inds)
    boxlist_filt = filter_scores_greater_than(subboxlist, score_thresh)
    nms_result = non_max_suppression(boxlist_filt,
                                     max_output_size=max_output_size,
                                     iou_threshold=iou_thresh,
                                     score_threshold=score_thresh)
    selected_boxes_list.append(nms_result)
  selected_boxes = concatenate(selected_boxes_list)
  sorted_boxes = sort_by_field(selected_boxes, 'scores')
  return sorted_boxes

def get_key(result_lists_or_dict):
  k = get_k_shot(result_lists_or_dict)
  for method in result_lists_or_dict.keys():
    if method.startswith('Tree_K{}'.format(k)) or method.startswith('mrf_K{}'.format(k)):
      return method

def get_k_shot(result_lists_or_dict):
  #find k_shot in a stupid way
  k_shot = 0
  for method in result_lists_or_dict.keys():
    if method.startswith('Tree_K') or method.startswith('mrf_K'):
      k = int(method.split('_')[1][1:])
      if k > k_shot:
        k_shot = k
  return k_shot


def _to_boxlist(res):
  boxlist = BoxList(res['boxes'])
  for field in ['classes', 'scores', 'proposal_inds']:
    if field in res:
      boxlist.add_field(field, res[field])
  return boxlist

def _merge_results(res0, res1=None, nms=False, max_proposal=50):
  res0 = dict(res0)

  had_scores = 'scores' in res0

  if not had_scores:
    res0['scores'] = np.zeros_like(res0['boxes'][..., 0])

  boxlist = _to_boxlist(res0)

  # Do the same thing to res1
  if res1 is not None:
    res1 = dict(res1)
    if not had_scores:
      res1['scores'] = np.zeros_like(res1['boxes'][..., 0])
    boxlist1 = _to_boxlist(res1)
    boxlist = concatenate([boxlist, boxlist1])

  if nms:
    boxlist = multi_class_non_max_suppression(boxlist, score_thresh=-float('inf'),
                                              iou_thresh=0.99, max_output_size=max_proposal)

  res = boxlist.data

  if 'class_agnostic' in res0:
    res['class_agnostic'] = res0['class_agnostic']

  if 'groundtruth' in res0:
    if res1 is not None:
      assert(np.all([np.all(res0['groundtruth'][key] == res1['groundtruth'][key]
                                ) for key in res0['groundtruth'].keys()]))
    res['groundtruth'] = res0['groundtruth']

  if not had_scores:
    res.pop('scores')
  return res

def merge_elems(elem0, elem1=None, nms=False, max_proposal=50):
  elem = dict()
  for key in elem0:
    if key == 'meta_info':
      ## merge meta_info
      elem[key] = elem0[key]
    else: # Tree_K*, upper_bound, mrf
      elem[key] = _merge_results(elem0[key],
                                 None if elem1 is None else elem1[key],
                                 nms, max_proposal=max_proposal)
  return elem

class ResultList(object):
  def __init__(self, remove_duplicates=False, override=False, parallel_iters=12):
    self._result_lists = []
    self._remove_duplicates = remove_duplicates
    self._merged_inds = []
    self._pool = Pool(parallel_iters)
    self._override = override #NOT USED
    if remove_duplicates:
      self._name_to_idx = dict()

  def add_results(self, result_dict):
    elems = self._ravel_results_dict(result_dict)
    if self._remove_duplicates:
      merged_inds = []
      for elem in elems:
        fn = elem['meta_info']['positive_images_name']
        #import copy
        #self._name_to_idx[fn] = len(self._result_lists)
        #self._result_lists.append(copy.deepcopy(elem))
        if fn in self._name_to_idx:
          ind = self._name_to_idx[fn]
          elem_orig = self._result_lists[ind]
          elem_new = merge_elems(elem_orig, elem)
          self._result_lists[ind] = elem_new
          #print('Duplicate found {}'.format(len(self._merged_inds)))
          self._merged_inds.append(ind)
          #if len(np.unique(elem_new['groundtruth']['classes'])) > 1:
          #  from IPython import embed;embed()
        else:
          self._name_to_idx[fn] = len(self._result_lists)
          self._result_lists.append(elem)
      ## Apply nms on the merged elems if necessary
      if len(self._merged_inds) > 5000:
        self.apply_nms()
    else:
      self._result_lists.extend(elems)

  def apply_nms(self, max_proposal=50):
    print('Applying NMS')
    inds = np.unique(self._merged_inds)
    reduce_fn = partial(merge_elems, elem1=None, nms=True, max_proposal=max_proposal)
    nmsed_elems = self._pool.map(reduce_fn, [self._result_lists[i] for i in inds])
    for i, nmsed_elem in zip(inds, nmsed_elems):
      self._result_lists[i] = nmsed_elem

    self._merged_inds = []
    print('Done.')

  def _ravel_results_dict(self, result_dict, length=None):
    if length is None:
      length = get_k_shot(result_dict)
    raveled = None
    for key, val in result_dict.items():
      if isinstance(val , dict):
        vals = self._ravel_results_dict(val, length)
      else:
        vals = list(val)
      assert(len(vals) == 1 or len(vals) == length
                        ), '{}, {}, {}'.format(key, len(vals), length)
      if len(vals) == 1:
        vals = vals*length
      if not raveled:
        raveled = [dict() for _ in range(length)]

      for val, r in zip(vals, raveled):
        r[key] = val

    return raveled

  def get_upack_results(self):
    return self._result_lists

  def get_packed_results(self, result_lists=None):
    result_lists = result_lists or self._result_lists
    result_dict = dict()
    elem0 = result_lists[0]
    keys = elem0.keys()
    for key in keys:
      nlist = [r[key] for r in result_lists]
      if isinstance(elem0[key], dict):
        unrvaled = self.get_packed_results(nlist)
      else:
        unrvaled = nlist
      result_dict[key] = unrvaled
    return result_dict

  def stats(self, key):
    num_images = len(self._result_lists)
    num_partially_labeled = 0
    for i in range(num_images):
      method = self._result_lists[i][key]
      gt_classes = set(np.unique(method['groundtruth']['classes']))
      pred_classes = set(np.unique(method['classes']))
      assert(gt_classes.issuperset(pred_classes))
      if len(gt_classes) != len(pred_classes):
        num_partially_labeled += 1

    return {'#Images:':num_images,  '#Partially Labeled':num_partially_labeled}


  def update_pseudo_dataset(self, update_ds):
    ''' Updates a dataset using self._result_lists
        Args:
          update_ds: dataset with feas and fea_boxes to be update with current results.
        Notes:
          - It assumes boxes, and classes for each result are comming after a multiclass NMS.
          - For each image and class, it assigns the highest scoring box. One box can come in
              the results multiple times with different classes.
    '''
    key = get_key(self._result_lists[0])

    for result in self._result_lists:
      imgname = result['meta_info']['positive_images_name']
      ind = update_ds['images'].index(imgname)
      roi = {'boxes': [], 'classes': []}
      tree_result = result[key]
      classes = np.unique(tree_result['classes'])

      for cls in classes:
        roi['classes'].append(cls)
        cls_inds = tree_result['classes'] == cls
        cls_boxes = tree_result['boxes'][cls_inds]
        cls_scores = tree_result['scores'][cls_inds]
        top_box = cls_boxes[np.argmax(cls_scores)]
        roi['boxes'].append(top_box)

      roi['boxes'] = np.array(roi['boxes']).astype(np.float32)
      roi['classes'] = np.array(roi['classes']).astype(np.uint32)
      update_ds['rois'][ind]['classes'] = roi['classes']
      update_ds['rois'][ind]['boxes'] = roi['boxes']
    return update_ds, self.stats(key)

  def create_pseudo_dataset(self, gt_ds):
    #TODO: this does not have objectness....
    ''' Creates a dataset using self._result_lists
        Args:
          gt_ds: groundtruth dataset with feas and fea_boxes.
        Notes:
          - It assumes boxes, and classes for each result are comming after a multiclass NMS.
          - For each image and class, it assigns the highest scoring box. One box can come in
              the results multiple times with different classes.
    '''
    images = []
    rois = []
    feas = []
    key = get_key(self._result_lists[0])
    all_classes = set()
    for result in self._result_lists:
      imgname = result['meta_info']['positive_images_name']
      images.append(imgname)
      ind = gt_ds['images'].index(imgname)
      roi = {'boxes': [], 'classes': []}
      roi['fea_boxes'] = gt_ds['rois'][ind]['fea_boxes']
      feas.append(gt_ds['feas'][ind])
      tree_result = result[key]
      classes = np.unique(tree_result['classes'])

      for cls in classes:
        all_classes.add(cls)
        roi['classes'].append(cls)
        cls_inds = tree_result['classes'] == cls
        cls_boxes = tree_result['boxes'][cls_inds]
        cls_scores = tree_result['scores'][cls_inds]
        top_box = cls_boxes[np.argmax(cls_scores)]
        roi['boxes'].append(top_box)

      roi['boxes'] = np.array(roi['boxes']).astype(np.float32)
      roi['classes'] = np.array(roi['classes']).astype(np.uint32)
      rois.append(roi)
    ds = {'rois': rois, 'images':images, 'feas': feas, 'synsets': gt_ds['synsets'], 'meta': "y0x0y1x1,relative" }
    return ds, self.stats(key)

  def evaluate_created_dataset(self, ds, gt_ds, evaluator_fn, add_synsets_from_gt=True, partial=False):
    ''' Computes PASCAL_VOC metric of a created dataset ds given the groundtruth dataset.
        Args:
          ds: A dictionary representing the created dataset. It must have rois and images
              keys each of them is a list. Each roi is a dictionary with boxes, and classes.
              Each image is a (relative) path from dataset root to the images.
          gt_ds: groundtruth dataset.
          add_synsets_from_gt: add groundtruth dataset synsets to created dataset ds.
          evaluator_fn: evaluator function
          partial: if True only evaluates the images in the result_list else evaluates all
                   images in ds
    '''
    result_lists = {'boxes':[], 'classes':[], 'scores':[]}
    groundtruth = {'boxes':[], 'classes':[]}
    def _add_result(i, imgname):
      roi = ds['rois'][i]
      result_lists['boxes'].append(roi['boxes'].astype(np.float32))
      result_lists['classes'].append(roi['classes'].astype(np.float32))
      result_lists['scores'].append(np.ones(roi['classes'].shape, dtype=np.float32))
      ind = gt_ds['images'].index(imgname)
      groundtruth['boxes'].append(gt_ds['rois'][ind]['boxes'])
      groundtruth['classes'].append(gt_ds['rois'][ind]['classes'])

    if partial:
      for result in self._result_lists:
        imgname = result['meta_info']['positive_images_name']
        i = ds['images'].index(imgname)
        _add_result(i, imgname)
    else:
      for i, imgname in enumerate(ds['images']):
        _add_result(i, imgname)
    if add_synsets_from_gt:
      ds['synsets'] = gt_ds['synsets']
    categories = [{'id': syn[0], 'name': syn[1]} for syn in gt_ds['synsets']]
    result_lists['groundtruth'] = groundtruth
    return evaluator_fn(result_lists, categories)

