import numpy as np
import itertools
import tensorflow as tf
from tensorflow.python.platform import flags
from sklearn.metrics import average_precision_score
import time
import cPickle as pickle
import gzip
import os
import logging

FLAGS = flags.FLAGS

def get_extra_dense_indices(k_shot):
  a = np.arange(k_shot)
  x = []
  for c,b in itertools.product(a,a):
    if c < b and not (c==b-1 and c % 2 == 0):
      x.append((c,b))
  return x

class GM(object):
  def __init__(self, num_nodes):
    self.num_nodes = num_nodes

  def get_indices(self):
    pass

  def get_num_edges(self):
    pass

  def infer(self, pairwise_scores, unary_scores, k, mrf_type,
            filenames=None, target_class=None, bcd_scores=None):
    '''
    Returns:
      - argmin: a list of size k where each element is the label selected for the
        corresponding node.
      - t: runing time for inference
    '''
    import opengm
    num_nodes = self.num_nodes
    if mrf_type.endswith('energy'):
      energy = -pairwise_scores.sum()
      argmin = np.zeros((num_nodes,), dtype=np.int32)
      t = 1
      return argmin, t, energy
    num_edges = self.get_num_edges()
    num_labels = int(np.sqrt(pairwise_scores.shape[1]))
    indices = self.get_indices()
    gm = opengm.gm([num_labels]*num_nodes, operator='adder')
    if unary_scores is not None:
      assert unary_scores.shape[0] == k
      assert unary_scores.shape[1] == num_labels
      for i in range(num_nodes):
        gm.addFactor(gm.addFunction(-unary_scores[i]), [i])
    for i,e in enumerate(indices):
      gm.addFactor(gm.addFunction(-pairwise_scores[i].reshape(
                                      (num_labels, num_labels))), [e[0],e[1]])

    if mrf_type.endswith('astar'):
      inf = opengm.inference.AStar(gm, accumulator='minimizer')
    elif mrf_type.endswith('trws'):
      inf = opengm.inference.TrwsExternal(gm, accumulator='minimizer')
    elif mrf_type.endswith('trbp'):
      inf = opengm.inference.TreeReweightedBp(gm, accumulator='minimizer')
    elif mrf_type.endswith('map'):
      inf = opengm.inference.BeliefPropagation(gm, accumulator='minimizer')
    else:
      raise ValueError('Inference for mrf type {} is not implemented'.format(mrf_type))
    #br_inf = opengm.inference.Bruteforce(gm, accumulator='minimizer')
    #br_inf.infer()
    t0 = time.time()
    inf.infer()
    argmin = inf.arg()
    t1 = time.time() - t0
    energy = inf.value()
    return argmin, t1, energy

class DenseGM(GM):
  def __init__(self, num_nodes):
    super(DenseGM, self).__init__(num_nodes)

  def get_indices(self):
    indices = [(i,i+1) for i in range(0, self.num_nodes,2)]
    extra_indices = get_extra_dense_indices(self.num_nodes)
    indices.extend(extra_indices)
    return indices

  def get_num_edges(self):
    return self.num_nodes * (self.num_nodes - 1) / 2

class LoopGM(GM):
  def __init__(self, num_nodes):
    super(LoopGM, self).__init__(num_nodes)

  def get_indices(self):
    indices = [(i,i+1) for i in range(0, self.num_nodes,2)]
    indices.extend([(i,i+1) for i in range(1,self.num_nodes-1,2)])
    indices.extend([(0,self.num_nodes-1)])
    return indices

  def get_num_edges(self):
    return self.num_nodes

class ChainGM(GM):
  def __init__(self, num_nodes):
    super(ChainGM, self).__init__(num_nodes)

  def get_indices(self):
    indices = [(i,i+1) for i in range(0, self.num_nodes,2)]
    indices.extend([(i,i+1) for i in range(1,self.num_nodes-1,2)])
    return indices

  def get_num_edges(self):
    return self.num_nodes-1

class GMAccumulator(object):
  """NOTE: assumes proposals are not reordered!
  """
  def __init__(self, save_path):
    localtime = time.localtime()
    start_time = "{}_{}_{}_{}_{}_{}".format(localtime.tm_year,
                                            localtime.tm_mon,
                                            localtime.tm_mday,
                                            localtime.tm_hour,
                                            localtime.tm_min,
                                            localtime.tm_sec)
    self.save_path = os.path.join(save_path, start_time)
    self.current_class = -1
    self.db = {'unaries': {}, 'pairwises': {}}
    if not os.path.exists(self.save_path):
      os.makedirs(self.save_path)

  def infer(self, pairwise_scores, unary_scores, k, mrf_type, filenames, target_class, bcd_scores=None):
    if target_class != self.current_class and self.current_class >= 0:
      # save current db

      # Adding time to db. A class will be read multiple times.
      # We can also update a single pkl every time but it might not be efficient
      localtime = time.localtime()
      logging.info("Saving db for class {}".format(self.current_class))
      #with gzip.GzipFile
      with open(os.path.join(self.save_path,
                         'mrf_class_{}_{}_{}_{}_{}.pkl'.format(
                             self.current_class,
                             localtime.tm_mday,
                             localtime.tm_hour,
                             localtime.tm_min,
                             localtime.tm_sec)), 'wb') as f:
        pickle.dump(self.db, f)
      # TODO: ? create graph and run inference if needed

      self.db = {'unaries': {}, 'pairwises': {}}
    self.current_class = target_class
    if 'dense' in mrf_type:
      model = DenseGM(k)
    elif 'loop' in mrf_type:
      model = LoopGM(k)
    elif 'chain' in mrf_type:
      model = ChainGM(k)
    else:
      raise ValueError('Mrf type {} not known'.format(mrf_type))
    indices = model.get_indices()
    if unary_scores is not None:
      for i in range(k):
        self.db['unaries'][filenames[i]] = -unary_scores[i].copy()
    num_labels = int(np.sqrt(pairwise_scores.shape[1]))
    for t, (i,j) in enumerate(indices):
      (p,q) = (min(filenames[i],filenames[j]), max(filenames[i],filenames[j]))
      self.db['pairwises'][(p,q)] = -pairwise_scores[t].copy().reshape((num_labels, num_labels))

    #fake results
    argmin = np.zeros((k,), dtype=np.uint64)
    t = 1
    energy = 0
    return argmin, t, energy

gm_accumulator = None #GMAccumulator("gm_db") #FLAGS.accumulator_save_path

class BCDDenseGM(GM):
  def __init__(self, num_nodes):
    super(BCDDenseGM, self).__init__(num_nodes)

  def get_indices(self):
    indices = [(i,i+1) for i in range(0, self.num_nodes,2)]
    extra_indices = get_extra_dense_indices(self.num_nodes)
    indices.extend(extra_indices)
    return indices

  def get_num_edges(self):
    return self.num_nodes * (self.num_nodes - 1) / 2

  def infer(self, pairwise_scores, unary_scores, k, mrf_type, filenames, target_class, bcd_scores):
    '''
    Returns:
      - argmin: a list of size k where each element is the label selected for the
        corresponding node.
      - t: runing time for inference
    '''
    assert(bcd_scores is not None)
    import opengm
    num_internal_nodes = self.num_nodes
    num_external_nodes = bcd_scores.shape[0]
    num_labels = int(np.sqrt(pairwise_scores.shape[1]))
    indices = self.get_indices()
    gm = opengm.gm([num_labels]*num_internal_nodes + [1]*num_external_nodes, operator='adder')
    if unary_scores is not None:
      assert unary_scores.shape[0] == k
      assert unary_scores.shape[1] == num_labels
      for i in range(num_internal_nodes):
        gm.addFactor(gm.addFunction(-unary_scores[i]), [i])
    for i,e in enumerate(indices):
      gm.addFactor(gm.addFunction(-pairwise_scores[i].reshape(
                                      (num_labels, num_labels))), [e[0],e[1]])

    for i,j in itertools.product(range(num_external_nodes), range(num_internal_nodes)):
      gm.addFactor(gm.addFunction(-bcd_scores[i,j][...,None]),
                                    [j, i+num_internal_nodes])

    if mrf_type.endswith('astar'):
      inf = opengm.inference.AStar(gm, accumulator='minimizer')
    elif mrf_type.endswith('trws'):
      inf = opengm.inference.TrwsExternal(gm, accumulator='minimizer')
    elif mrf_type.endswith('trbp'):
      inf = opengm.inference.TreeReweightedBp(gm, accumulator='minimizer')
    elif mrf_type.endswith('map'):
      inf = opengm.inference.BeliefPropagation(gm, accumulator='minimizer')
    else:
      raise ValueError('Inference for mrf type {} is not implemented'.format(mrf_type))
    #br_inf = opengm.inference.Bruteforce(gm, accumulator='minimizer')
    #br_inf.infer()
    t0 = time.time()
    inf.infer()
    argmin = inf.arg()
    t1 = time.time() - t0
    energy = inf.value()
    self._debug = dict(gm=gm, pairwise_scores=pairwise_scores,
                       unary_scores=unary_scores,
                       k=k, mrf_type=mrf_type,
                       filenames=filenames, target_class=target_class,
                       bcd_scores=bcd_scores,
                       argmin=argmin,
                       min_energy=energy)
    return argmin[:k], t1, energy



def _fill_to_size(_lists, size):
  for _list in _lists:
    _list.extend(_list[:size])

def _extend_sub_list(_lists, start, end):
  for _list in _lists:
    _list.extend(_list[start:end])

def _extend_for_chain_mrf(lists, k_shot, bag_size, num_negative_bags):
  for k in range(1,k_shot - 1):
    _k = (k//2)*num_negative_bags + k # //2 for negative bags
    _extend_sub_list(lists, start=_k*bag_size, end=(_k+1)*bag_size)
    if k % 2 == 0 and num_negative_bags > 0:
      #add negatives after 2 positives k=1,2, Ns, 3, 4, Ns
      _extend_sub_list(lists, start=2*bag_size, end=(2+num_negative_bags)*bag_size)

  #make it loop
  _extend_sub_list(lists, start=0, end=bag_size) #first bag
  _k = ((k_shot - 1) // 2)*num_negative_bags + k_shot - 1 # last positive bag start
  _extend_sub_list(lists, start=_k*bag_size, end=(_k+1)*bag_size)
  if num_negative_bags > 0:
    _extend_sub_list(lists, start=2*bag_size, end=(2+num_negative_bags)*bag_size)

def _extend_for_dense_mrf(lists, bcd_bags, k_shot, bag_size, num_negative_bags):
  '''
  each element in lists has the form [pos_bag_1, pos_bag_2, neg1, ..., negN,
                                      pos_bag_3, pos_bag_4, neg1, ..., negN,
                                      ...
                                      pos_bag_k-1, pos_bag_k, neg1, ..., negN]
  this will extend it for dense mrf by using all pairs from 0...k
  Note:
    - assumes negative bags all have bag_size elements
  '''
  extra_indices = get_extra_dense_indices(k_shot)

  for i,j in extra_indices:
    _i = (i // 2)*num_negative_bags + i # //2 for negative bags after reorder
    _j = (j // 2)*num_negative_bags + j
    _extend_sub_list(lists, start=_i*bag_size,end=(_i+1)*bag_size)
    _extend_sub_list(lists, start=_j*bag_size,end=(_j+1)*bag_size)
    # add negatives after each 2 positives
    if num_negative_bags > 0:
      _extend_sub_list(lists, start=2*bag_size, end=(2+num_negative_bags)*bag_size)

  if k_shot % 2 == 1:
    assert(bag_size == 1 and bcd_bags is None),"Odd K-shot is not supported for bag size > 1"
    for l in lists:
      l.pop(k_shot-1)

  if bcd_bags is not None:
    num_bcd_bags = len(bcd_bags)/bag_size
    images_list = lists[0]
    list1 = lists[1:]
    # Add all the pairs between the original k_shot subproblem and bcd_bags
    for j,i in itertools.product(range(num_bcd_bags), range(k_shot)):
      images_list.extend(bcd_bags[j*bag_size:(j+1)*bag_size])
      images_list.extend(images_list[i*bag_size:(i+1)*bag_size])
      # Just pad everything else. We do not care about the actual values
      _extend_sub_list(list1, start=0, end=2*bag_size)

def extend_for_mrf(lists, bcd_bags, k_shot, bag_size, num_negative_bags):
  if 'chain' in FLAGS.mrf_type or 'loop' in FLAGS.mrf_type:
    _extend_for_chain_mrf(lists, k_shot, bag_size, num_negative_bags)
  elif 'dense' in FLAGS.mrf_type:
    _extend_for_dense_mrf(lists, bcd_bags, k_shot, bag_size, num_negative_bags)
  else:
    raise ValueError('mrf type: {} is not implemented'.format(FLAGS.mrf_type))

def argmin_to_result_dict(result_dict, argmin):
  k1_boxes = result_dict['Tree_K1']['boxes']
  k1_classes = result_dict['Tree_K1']['classes']
  mrf_boxes = k1_boxes[:len(argmin),:,:].copy()
  mrf_classes = k1_classes[:len(argmin),:].copy()
  mrf_scores = np.zeros_like(mrf_classes)
  mrf_scores[np.arange(len(argmin)), argmin] = 1
  return {'boxes': mrf_boxes,
          'classes': mrf_classes,
          'scores':mrf_scores,
          'class_agnostic': True}


def add_mrf_scores(result_dict, meta_info,
                    function, scale, input_k_shot,
                    filenames,
                    return_model=False):
  orig_scores = meta_info['k2_cross_similarity_scores']
  bcd_scores = None
  target_class = 0
  if 'target_class' in result_dict:
    target_class = int(result_dict['target_class'])
  if orig_scores.shape[-1] > 1:
    #target_class = int(result_dict['groundtruth']['target_class'])
    orig_scores = orig_scores[..., target_class - 1]#np.max(orig_scores, axis=-1)
  else:
    orig_scores = orig_scores[..., 0]
  scores, name = function(orig_scores)
  if 'bcd_pairwises' in meta_info:
    #[N,k,bag_size]
    bcd_scores, _ = function(meta_info['bcd_pairwises'])
  unary_scores = None
  if FLAGS.add_unaries:
    #TODO
    #raise NotImplementedError('Multi-class version that uses target_class is not implemented.')
    #reads k1 scores for first input_k_shot positive bags
    unary_scores,_ = function(result_dict['Tree_K1']['scores'][:input_k_shot,:])
    unary_scores = unary_scores*scale
  supported_dense_methods = ['dense_trws', 'dense_astar', 'dense_trbp', 'dense_map']
  supported_loop_methods = ['loop_trws', 'loop_astar', 'loop_trbp', 'loop_map']
  if FLAGS.mrf_type.startswith('dense'):
    model = DenseGM(input_k_shot)
  elif FLAGS.mrf_type.startswith('loop'):
    model = LoopGM(input_k_shot)
  elif FLAGS.mrf_type.startswith('chain'):
    model = ChainGM(input_k_shot)
  elif FLAGS.mrf_type.startswith('accumulator'):
    model = gm_accumulator
  elif FLAGS.mrf_type.startswith('bcd'):
    assert(bcd_scores is not None)
    model = BCDDenseGM(input_k_shot)
  else:
    raise ValueError('mrf type {} not supported yet'.format(FLAGS.mrf_type))
  argmin, t, energy = model.infer(scores, unary_scores, input_k_shot, FLAGS.mrf_type,
                                  filenames, target_class, bcd_scores)
  d = argmin_to_result_dict(result_dict, argmin)
  mrf_name = 'mrf_K{}_{}_f_{}_s{}'.format(input_k_shot, FLAGS.mrf_type, name, scale)
  ret = (d, t, energy, mrf_name)
  if return_model:
    ret += (model,)
  return ret

def add_mrf_scores_dense_greedy(result_dict, scores, inds, gt, name, scale, unary_scores, unary_inds):
  from greedy import GreedyCoObj
  num_nodes = gt.shape[0]
  num_edges = num_nodes * (num_nodes - 1) / 2
  num_labels = gt.shape[1]
  indices = [(i,i+1) for i in range(0, gt.shape[0],2)]
  extra_indices = get_extra_dense_indices(num_nodes)
  indices.extend(extra_indices)
  factors = {}
  if unary_scores is not None and unary_inds is not None:
    for i in range(num_nodes):
      potential = np.zeros((num_labels,))
      for j in range(num_labels):
        potential[unary_inds[i,j]] = unary_scores[i,j]
      factors[i] = potential
  for i,e in enumerate(indices):
    potential = np.zeros((num_labels, num_labels))
    for j in range(inds.shape[1]):
      index = i*2
      potential[inds[index,j], inds[index+1,j]] = scores[index,j]
    factors[e] = potential

  #first layer co-objects
  t0 = time.time()
  num_to_keep = 30
  use_scores = True
  cobjs = [GreedyCoObj(vars=[i], values=[[x] for x in range(num_labels)]) for i in range(num_nodes)]
  while len(cobjs) > 1:
    joined_cobjs = [cobjs[i].join(cobjs[i+1]) for i in range(0,len(cobjs),2)]
    evalutions = [co.evaluate(factors, num_to_keep, use_scores=use_scores) for co in joined_cobjs]
    cobjs = joined_cobjs
  t1 = time.time() - t0
  result_dict['Tree_K2']['time_dense_greedy_inference'] = [t1]

  mrf_map = np.zeros_like(gt)
  for var, val in zip(cobjs[0].vars, cobjs[0].values[0]):
    mrf_map[var, val] = 1

  max_marginals = mrf_map
  result_dict['Tree_K2']['mrf_dense_max_marginal_greedy_top_acc_{}_scale_{}'.format(name,scale)] = [np.all(gt[max_marginals==1] == 1)]
  result_dict['Tree_K2']['mrf_dense_max_marginal_greedy_top_percent_cor_{}_scale_{}'.format(name,scale)] = [np.mean(gt[max_marginals==1])]


