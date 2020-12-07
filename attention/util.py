import tensorflow as tf
from object_detection.core import balanced_positive_negative_sampler as sampler
import object_detection.utils.shape_utils as shape_utils
import cross_similarity
slim = tf.contrib.slim
import numpy as np

def reset_static_values():
    cross_similarity.PairwiseCrossSimilarity.k2_scope = None
    cross_similarity.PairwiseCrossSimilarity.n_unique_scores = 0

def get_temperature_variable(var_name, initializer=1.0):
  assert(initializer > 0)
  initializer = np.float32(np.log(np.exp(initializer) - 1))
  temperature = slim.variable(var_name, initializer=initializer,
                              regularizer=slim.l2_regularizer(0.0))
  temperature = tf.nn.softplus(temperature)
  tf.summary.scalar(var_name, temperature)
  return temperature

def maybe_call(fn):
  if callable(fn):
    return fn()
  else:
    return fn

def tile_and_reshape_cobj_prop(prop, k):
  # Since we have one feature vector for each co-object
  # (each co-object is for k images) we need to repeat each
  # co-object feature vector k times.
  shape = shape_utils.combined_static_and_dynamic_shape(
                                          prop)
  prop = tf.tile(prop[:, tf.newaxis],
                 [1, k] + [1]*(len(shape)-1))
  shape = [-1] + shape[1:]
  return tf.reshape(prop, shape)


def convert_proposal_inds(proposal_inds):
    # [N, M, J]
    # proposal_inds.shape = [meta_batch_size, self.ncobj_proposals, k_shot]
    # ==> [meta_batch_size, k_shot, self.ncobj_proposals]
    proposal_inds = tf.transpose(proposal_inds, perm=[0, 2, 1])
    ncobj_proposals = shape_utils.combined_static_and_dynamic_shape(
        proposal_inds)[2]
    # ==> [meta_batch_size*k_shot, self.ncobj_proposals]
    return tf.reshape(proposal_inds,
                     [-1, ncobj_proposals])

def topk_inds(indicators, scores, k, ntrues=None):
  assert(k > 0)
  if ntrues is None:
    ntrues = tf.count_nonzero(indicators, dtype=tf.int32)
  assertion = tf.Assert(tf.greater_equal(ntrues, k), [k, ntrues])
  with tf.control_dependencies([assertion]):
    valid_inds = tf.where(indicators)[:,0]
    if scores is None:
      new_inds = tf.random_shuffle(valid_inds)[:k]
    else:
      valid_scores = tf.gather(scores, valid_inds)
      _, indices = tf.nn.top_k(valid_scores, k)
      new_inds = tf.gather(valid_inds, indices)
    new_inds.set_shape(k)
    return new_inds

def pad_inds_with_resampling(indicators, k, ntrues=None):
  if ntrues is None:
    ntrues = tf.count_nonzero(indicators, dtype=tf.int32)
  nsamples = k - ntrues
  valid_inds = tf.where(indicators)[:,0]

  assertion0 = tf.Assert(tf.greater(nsamples, 0), [nsamples, k, ntrues])
  assertion1 = tf.Assert(tf.greater(ntrues, 0), [nsamples, k, ntrues])

  with tf.control_dependencies([assertion0, assertion1]):
    # Resampling
    idx = tf.random_uniform((nsamples,), maxval=ntrues, dtype=tf.int32)
    resample_inds = tf.gather(valid_inds, idx)
    new_inds = tf.concat([valid_inds, resample_inds], 0)
    new_inds.set_shape(k)
    return new_inds

def topk_or_pad_inds_with_resampling(indicators, scores, k):
  '''
    Args:
      indicators: boolean tensor of shape [J] whose True
                  entries can be sampled
      scores: shape [J]
      k: an integer
    Return: [k]
  '''
  assert(k > 0)

  # For cases that indicators are all false
  ntrues = tf.count_nonzero(indicators, dtype=tf.int32)
  indicators = tf.logical_or(indicators, tf.equal(ntrues, 0))
  ntrues = tf.count_nonzero(indicators, dtype=tf.int32)
  return tf.cond(ntrues < k,
                 lambda: pad_inds_with_resampling(indicators, k, ntrues),
                 lambda: topk_inds(indicators, scores, k, ntrues))

def batched_topk_or_pad_inds_with_resampling(indicators, scores, k):
  '''
    Args:
      indicators: [N, J]
      scores: [N, J]
      k: an integer

    Return: [N, k]
  '''
  ### <DEBUG>
  #print('\t\t*** check if scores is None ***');from IPython import embed;embed()
  #scores = tf.Print(scores, [tf.reduce_mean(scores)], 'Scores mean: ')
  ### </DEBUG>
  if scores is None:
    fn = lambda x: topk_or_pad_inds_with_resampling(x, None, k)
    elems = indicators
  else:
    fn = lambda x: topk_or_pad_inds_with_resampling(x[0], x[1], k)
    elems = (indicators, scores)

  return tf.map_fn(fn, elems, dtype=tf.int64,
        parallel_iterations=10,
        back_prop=True,
        name='batched_topk_or_pad_with_resampling')

def subsample(indicators, k):
  return balanced_subsample(indicators, tf.ones_like(indicators), k, 1.0)

def balanced_subsample(indicators, labels, k, balance_fraction):
  '''
    If sum(indicators) < k uses resampleing to return k indices. Otherwise uses
    object_detection.core.blaance_positive_negative_sampler.BalancedPositiveNegativeSampler
    to sample k indices.
    Args:
      indicators: boolean tensor of shape [N] whose True entries can be sampled.
      k: desired batch size.
      labels: boolean tensor of shape [N] denoting positive(=True) and negative
          (=False) examples.
    Returns:
      sampled_indices: tensor of shape [k], which has the indicies
      for entries which are sampled.
  '''
  assert(k > 0)

  # For cases that indicators are all false
  ntrues = tf.count_nonzero(indicators, dtype=tf.int32)
  indicators = tf.logical_or(indicators, tf.equal(ntrues, 0))
  ntrues = tf.count_nonzero(indicators, dtype=tf.int32)
  def _subsample():
    balance_sampler = sampler.BalancedPositiveNegativeSampler(
        positive_fraction=balance_fraction)
    is_sampled = balance_sampler.subsample(indicators, k, labels)
    n_sampled = tf.count_nonzero(is_sampled, dtype=tf.int32)

    def get_inds():
      inds = tf.where(is_sampled)[:,0]
      inds.set_shape(k)
      return inds

    def resample():
      inds = pad_inds_with_resampling(is_sampled, k, n_sampled)
      return tf.Print(inds, ['Warning: balance_sampler result padded',
                             k-n_sampled])

    return tf.cond(tf.equal(n_sampled, k),
                   get_inds,  # Return indices
                   resample)  # Add k - n_sampled indices by resampling

  return tf.cond(ntrues < k,
                 lambda: pad_inds_with_resampling(indicators, k, ntrues),
                 lambda: _subsample())


def subsample_enforce_balance_fraction(indicators, labels, scores, k, balance_fraction):
  '''
    Args:
      indicators: boolean tensor of shape [N] whose True entries can be sampled.
      k: desired batch size.
      labels: boolean tensor of shape [N] denoting positive(=True) and negative
          (=False) examples.
    Returns:
      sampled_indices: tensor of shape [k], which has the indicies
      for entries which are sampled.
  '''
  npos = int(k*balance_fraction)
  nneg = k - npos
  pos_scores = None if scores is None else 1-scores
  neg_scores = scores
  pos_indicators = tf.logical_and(indicators, labels)
  neg_indicators = tf.logical_and(indicators, tf.logical_not(labels))

  sampled_inds_list = []
  if npos > 0:
    sampled_inds_list.append(
        topk_or_pad_inds_with_resampling(pos_indicators, pos_scores, npos))

  if nneg > 0:
    sampled_inds_list.append(
        topk_or_pad_inds_with_resampling(neg_indicators, neg_scores, nneg))

  assert(sampled_inds_list)
  if len(sampled_inds_list) == 1:
    return sampled_inds_list[0]
  else:
    return tf.concat(sampled_inds_list, 0)


def unique_with_inverse(x):
  y, idx = tf.unique(x)
  num_segments = tf.shape(y)[0]
  num_elems = tf.shape(x)[0]
  return (y, idx, tf.unsorted_segment_max(
                        tf.range(num_elems),
                        idx,
                        num_segments))

def batched_gather(indices, *tensor_list):
  ''' Batched gather operation. For every tensor in the tensor_list
      output[i,j, ...] = tensor[i, indices[i, j], ...]
      Args:
        indices: array of indices to select with size [N, M]
        tensor_list: list of tensors with size [N, L, ...]
  '''

  assert(len(indices.shape) == 2)

  # create a 3D tensor with size [N, M, 2]
  # in which array[i, j] = (i, indices[i,j])
  n_ids = tf.range(indices.shape[0], dtype=indices.dtype)
  n_ids = tf.tile(n_ids[:, tf.newaxis], [1, indices.shape[1]])
  indices = tf.stack([n_ids, indices], axis=2)
  ret = []
  for tensor in tensor_list:
    out = None if tensor is None else tf.gather_nd(tensor, indices)
    ret.append(out)
  if len(ret) == 1:
    return ret[0]
  return ret

def add_extra_tensor(key, value):
  #if True:
  #  return
  #from IPython import embed;embed()
  extra_dict = get_extra_tensors_dict()
  if key in extra_dict:
    key += '*'
    print(key)
    add_extra_tensor(key, value)
    return
  tf.add_to_collection('extra_tensors', (key, value))

def get_extra_tensors_dict():
  return dict(tf.get_collection('extra_tensors'))
