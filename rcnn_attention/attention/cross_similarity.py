import tensorflow as tf
import object_detection.utils.shape_utils as shape_utils
from abc import abstractmethod
import util
import cross_similarity

slim = tf.contrib.slim

class CrossSimilarity(object):
  def __init__(self):
    self._joined_fea = None
    self._joining_convline = None

  @property
  def support_pre_joining(self):
    return False

  def set_joining_convline(self, joining_convline):
    if not self.support_pre_joining:
      raise ValueError('{} does not support pre-joining'.format(self))
    self._joining_convline = joining_convline

  def build(self, fea0, fea1,
            ind0, ind1,
            score_size, neg_fea=None,
            matched_class0=None,
            neg_matched_class=None,
            reuse_vars=False,
            scope=None,
            target_score_inds=None):
    '''
      Args:
        fea0, fea1: feature tensors with size [N, M, h, w, d]
      Returns:
        scores: upper triangular part of the matching scores
                with size [N, K]. K = M*(M-1)/2
        pairs: A tensor which keeps pair indices for each
               matching score with size [N, K, 2]
    '''
    with tf.variable_scope(scope, 'cross_similarity',
        reuse=reuse_vars, values=[fea0, fea1]):

      # Rehape tensor: [N, M, M, ...] ==> [N, M*M, ...]
      def _flat(tensor):
        shape = shape_utils.combined_static_and_dynamic_shape(tensor)
        return tf.reshape(tensor, [shape[0], shape[1]*shape[2]] + shape[3:])

      # Reshape fea: [N, M, h, w, d] ==> [N, M, h*w*d]
      fea0_shape = shape_utils.combined_static_and_dynamic_shape(fea0)
      fea0 = tf.reshape(fea0, fea0_shape[:2] + [-1])

      if fea1 is None:
        fea1_shape = list(fea0_shape)
        fea1_shape[1] = 1
      else:
        fea1_shape = shape_utils.combined_static_and_dynamic_shape(fea1)
        fea1 = tf.reshape(fea1, fea1_shape[:2] + [-1])

      with tf.variable_scope('similarity_matrix', values=[fea0, fea1]) as sc:
        self._target_score_inds = target_score_inds
        scores, negative_loss = self._build(fea0, fea1, ind0, ind1,
                                            score_size, neg_fea,
                                            matched_class0, neg_matched_class,
                                            reuse_vars, sc)

        scores = _flat(scores)

      # Find indices
      pairs = tf.stack(tf.meshgrid(tf.range(fea0_shape[1]),
                                   tf.range(fea1_shape[1]), indexing='ij'),
                       axis=-1)
      pairs = tf.tile(pairs[tf.newaxis], [fea0_shape[0], 1, 1, 1])
      pairs = _flat(pairs)

      if self._joined_fea is None:
        joined_fea = None
      else:
        joined_fea = _flat(self._joined_fea)

      return (scores, pairs, joined_fea, negative_loss)

  @abstractmethod
  def _build(self, fea0, fea1, ind0, ind1,
             score_size, neg_fea, matched_class0,
             neg_matched_class, reuse_vars, scope):
    '''
      Args:
        fea0, fea1: feature tensors with size [N, M, d]
      Returns:
        Similarity matrix with size [N, M, M]
    '''
    pass

class CosineCrossSimilarity(CrossSimilarity):
  def __init__(self):
    super(CosineCrossSimilarity, self).__init__()

  def _build(self, fea0, fea1, ind0, ind1,
             score_size, neg_fea, matched_class0,
             neg_matched_class, reuse_vars, scope):
    assert(neg_fea is None), 'cosine similarity does not handle negative features'
    assert(score_size == 1), 'cosine similarity only works with class agnostic ON'
    fea0 = tf.nn.l2_normalize(fea0, dim=-1)
    fea1 = tf.nn.l2_normalize(fea1, dim=-1)
    # [N, M, K] * [N, P, K]' = [N, M, P]
    scores = tf.matmul(fea0, fea1, transpose_b=True)
    return scores[..., tf.newaxis], None

class EuclideanCrossSimilarity(CrossSimilarity):
  def __init__(self):
    super(EuclideanCrossSimilarity, self).__init__()

  def _build(self, fea0, fea1, ind0, ind1,
             score_size, neg_fea,
             matched_class0, neg_matched_class,
             reuse_vars, scope):
    assert(neg_fea is None), 'EuclideanCrossSimilarity does not handle negative features'
    assert(score_size == 1), 'EuclideanCrossSimilarity only works with class agnostic ON'

    fea0_shape = shape_utils.combined_static_and_dynamic_shape(fea0)
    fea1_shape = shape_utils.combined_static_and_dynamic_shape(fea1)
    m = fea0_shape[1]
    p = fea1_shape[1]

    # [N, M, K] * [N, P, K]' = [N, M, P]
    dotxy = tf.matmul(fea0, fea1, transpose_b=True)
    #[N, M]
    norm0 = tf.reduce_sum(tf.square(fea0), axis=-1)
    #[N, P]
    norm1 = tf.reduce_sum(tf.square(fea1), axis=-1)

    # shape = [N, M, P]
    norm0 = tf.tile(norm0[:, :, tf.newaxis], [1,1,p])

    # shape = [N, M, P]
    norm1 = tf.tile(norm1[:, tf.newaxis], [1,m,1])

    scores = 2*dotxy - norm0 - norm1
    return scores[..., tf.newaxis], None

class AverageCrossSimilarity(CrossSimilarity):
  def __init__(self):
    super(AverageCrossSimilarity, self).__init__()

  @property
  def support_pre_joining(self):
    return True

  # fea shape [N, M, K] ==> [N, M, M, K]
  def _build(self, fea0, fea1, ind0, ind1,
             score_size, neg_fea, matched_class0,
             neg_matched_class, reuse_vars, scope):
    fea0_shape = shape_utils.combined_static_and_dynamic_shape(fea0)
    m = fea0_shape[1]

    # shape = [N, M, M, K]
    fea0 = tf.tile(fea0[:, :, tf.newaxis], [1,1,m,1])
    # shape = [N, M, M, K]
    fea1 = tf.tile(fea1[:, tf.newaxis], [1,m,1,1])
    # shape = [N, M, M, 2K]
    fea01 = tf.concat((fea0, fea1), axis=-1)

    # shape = [N, 1, M, M, 2K] or [N, 1, M, 1, K]
    fea01 = fea01[:, tf.newaxis]
    assert(self._joining_convline is not None)
    fea01 = self._joining_convline.build(fea01, scope='join')
    self._joined_fea = tf.squeeze(fea01, 1)

    # shape [N, M, M, K]
    return self._joined_fea, None

class PairwiseCrossSimilarity(CrossSimilarity):
  k2_scope = None
  n_unique_scores = 0
  def __init__(self, stop_gradient, cross_similarity,
               k, attention_tree,
               k2_scope_key='pairwise_cross_similarity'):

    super(PairwiseCrossSimilarity, self).__init__()
    self._cross_similarity = cross_similarity
    self._k = k
    self._attention_tree = attention_tree
    self._stop_gradient = stop_gradient
    self._k2_scope_key = k2_scope_key
    self._fea_range = None
    assert(k >= 2)

  def support_pre_joining(self):
    return True

  def set_joining_convline(self, joining_convline):
    self._cross_similarity.set_joining_convline(joining_convline)

  def _build(self, fea0, fea1, ind0, ind1,
             score_size, neg_fea, matched_class0,
             neg_matched_class, reuse_vars, scope):

    if PairwiseCrossSimilarity.k2_scope is None:
      PairwiseCrossSimilarity.k2_scope = dict()

    if self._k2_scope_key in PairwiseCrossSimilarity.k2_scope:
      reuse_vars = True
      scope = PairwiseCrossSimilarity.k2_scope[self._k2_scope_key]
    else:
      PairwiseCrossSimilarity.k2_scope[self._k2_scope_key] = scope

    hk = int(self._k/2)

    # Number of new pairwise potentials
    s = hk**2

    solo_fea = self._attention_tree._solo_cobj.get('fea')
    if self._fea_range is not None:
      solo_fea = solo_fea[..., self._fea_range[0]:self._fea_range[1]]

    solo_fea = tf.squeeze(solo_fea, [2,3])
    solo_fea_shape = shape_utils.combined_static_and_dynamic_shape(solo_fea)
    def create_ids(ind):
      ind_shape = shape_utils.combined_static_and_dynamic_shape(ind)
      bag_size = solo_fea_shape[1]
      ids = ind + bag_size * tf.range(ind_shape[-1], dtype=tf.int32)
      #[ind_shape[0],-1] -> in map_fn: [-1]
      return tf.reshape(ids, [ind_shape[0], -1])

    def unique_with_inverse(x):
      y, idx = tf.unique(x)
      num_segments = tf.shape(y)[0]
      num_elems = tf.shape(x)[0]
      return (y, idx,  tf.unsorted_segment_max(tf.range(num_elems), idx, num_segments))

    ids0 = create_ids(ind0)
    ids1 = create_ids(ind1)
    def reduced_score(inp, reuse_vars=True):
        solo_fea, ids0, ids1, target_score_inds = inp
        u_ids0, idx_map0, inverse0 = unique_with_inverse(ids0)
        u_ids1, idx_map1, inverse1 = unique_with_inverse(ids1)
        reduced_fea0 = tf.gather(solo_fea[0], u_ids0)
        reduced_fea1 = tf.gather(solo_fea[1], u_ids1)

        with tf.variable_scope(scope, reuse=reuse_vars) as sc:
          self._cross_similarity._target_score_inds = target_score_inds
          reduced_scores, loss = self._cross_similarity._build(
              reduced_fea0[tf.newaxis,...],
              reduced_fea1[tf.newaxis,...], None, None,
              score_size, None, None,
              None, reuse_vars=reuse_vars, scope=sc)
          assert(reduced_scores.shape[-1] == 1)
        # [1, m', l', 1]
        reduced_scores_shape = shape_utils.combined_static_and_dynamic_shape(
                                                              reduced_scores)
        nscores = reduced_scores_shape[1]*reduced_scores_shape[2]
        # [m', l', 1]
        reduced_scores = tf.reshape(reduced_scores, reduced_scores_shape[1:])
        # [m, l', 1]
        scores_0 = tf.gather(reduced_scores, idx_map0)
        # [m, l, 1]
        scores = tf.gather(scores_0, idx_map1, axis=1)
        return scores, nscores

    bs = shape_utils.combined_static_and_dynamic_shape(ids0)[0]
    rsolo_fea = tf.reshape(solo_fea, [bs, 2, -1, solo_fea_shape[2]])

    if False:
      unused = reduced_score((rsolo_fea[0], ids0[0],ids1[0]), reuse_vars)
      scores, nscores = tf.map_fn(reduced_score, elems=(rsolo_fea, ids0, ids1),
                                  dtype=(tf.float32, tf.int32))
    else:
      scores, nscores = [], []
      num = rsolo_fea.shape[0]
      assert(num is not None)
      for i in range(num):
        target_score_inds = None
        if self._target_score_inds is not None:
          target_score_inds = self._target_score_inds[i][tf.newaxis]
        ret = reduced_score((rsolo_fea[i], ids0[i], ids1[i], target_score_inds),
                             True if i > 0 else reuse_vars)
        scores.append(ret[0])
        nscores.append(ret[1])
      scores = tf.stack(scores)
      nscores = tf.stack(nscores)
    PairwiseCrossSimilarity.n_unique_scores += tf.reduce_sum(nscores)

    scores= slim.conv2d(scores, 1, kernel_size=hk, stride=hk,
                        padding='VALID', biases_initializer=None,
                        activation_fn=None,
                        weights_initializer=tf.constant_initializer(1.0),
                        trainable=False)
    # pass empty feature to the next level
    fea_shape = shape_utils.combined_static_and_dynamic_shape(scores)
    fea_shape[-1] = 0
    self._joined_fea = tf.constant(0.0, shape=fea_shape)

    return scores, None

class K1CrossSimilarity(CrossSimilarity):
  def __init__(self, cross_similarity, k,
               share_weights_with_pairwise_cs=False,
               mode='MAX',
               topk=None):
    super(K1CrossSimilarity, self).__init__()
    assert(k == 1)
    if mode not in ['MAX', 'MEAN', 'SOFTMAX']:
       raise ValueError('mode {} is not valid.'.format(mode))
    self._cross_similarity = cross_similarity
    self._share_weights_with_pairwise_cs = share_weights_with_pairwise_cs
    self._mode = mode
    self._topk = topk

  def support_pre_joining(self):
    return True

  def set_joining_convline(self, joining_convline):
    self._cross_similarity.set_joining_convline(joining_convline)


  def _sim(self, pos_fea, neg_fea, scope):
    # Reshape neg_fea [MBS, L, 1, 1, d] ==> [MBS, L, d]
    neg_fea = tf.squeeze(neg_fea, [2,3])

    # Reshape pos_fea: [MBS*K, M, d] ==> [MBS, K*M, d]
    neg_shape = shape_utils.combined_static_and_dynamic_shape(neg_fea)
    pos_shape = shape_utils.combined_static_and_dynamic_shape(pos_fea)
    pos_fea = tf.reshape(pos_fea, [neg_shape[0], -1, pos_shape[-1]])

    if self._share_weights_with_pairwise_cs:
      scope = PairwiseCrossSimilarity.k2_scope['pairwise_cross_similarity']

    pos_shape = shape_utils.combined_static_and_dynamic_shape(pos_fea)

    kwargs = {}
    ## Only compute sim to topk nn in the negative bags
    if self._topk and not isinstance(self._cross_similarity,
                                     CosineCrossSimilarity):
        cs = CosineCrossSimilarity()
        fast_sim, _ = cs._build(pos_fea, neg_fea, None, None, 1,
                                None, None, None, False, None)
        fast_sim = tf.stop_gradient(fast_sim[...,0])
        _, inds = tf.nn.top_k(fast_sim, self._topk, sorted=False)
        inds = tf.reshape(inds, [neg_shape[0], -1])
        neg_fea = util.batched_gather(inds, neg_fea)
        neg_fea = tf.reshape(neg_fea, pos_shape[:2] + [
                                      self._topk,
                                      neg_shape[-1]])
        kwargs['tile_fea1'] = False

    with tf.variable_scope(scope, 'k1_cross_similarity') as scope:
        self._cross_similarity._target_score_inds = self._target_score_inds
        sim, _ = self._cross_similarity._build(pos_fea, neg_fea,
                                               None, None, 1, None,
                                               None, None, False, None,
                                               **kwargs)
    if self._share_weights_with_pairwise_cs:
      PairwiseCrossSimilarity.k2_scope['pairwise_cross_similarity'] = scope

    return sim

  def _build_(self, fea0, fea1, ind0, ind1,
             score_size, neg_fea, matched_class0,
             neg_matched_class, reuse_vars, scope):
    print('Warning: Do not use this function (K1CrossSimilarity._build) for training')

    def fn(fea0):
      fea0 = fea0[tf.newaxis]
      scores, loss = self._build_inner(fea0, fea1, ind0, ind1,
                               score_size, neg_fea, matched_class0,
                               neg_matched_class, reuse_vars, scope)
      return scores[0]

    mini_bs = 64

    fea0_shape = shape_utils.combined_static_and_dynamic_shape(fea0)
    rfea0 = tf.reshape(fea0, [-1] + fea0_shape[2:])

    rem = tf.mod(mini_bs - tf.mod(fea0_shape[0]*fea0_shape[1], mini_bs), mini_bs)

    rfea0 = tf.pad(rfea0, [[0, rem], [0,0]])
    rfea0 = tf.reshape(rfea0, [-1, mini_bs, fea0_shape[-1]])

    scores = tf.map_fn(
                      fn,
                      rfea0,
                      dtype=tf.float32,
                      parallel_iterations=1,
                      back_prop=False,
                      swap_memory=True,
                      infer_shape=True,
                      name='memory_efficient_k1')
    scores_shape = shape_utils.combined_static_and_dynamic_shape(scores)
    scores = tf.reshape(scores, [-1]+scores_shape[2:])
    scores_shape = shape_utils.combined_static_and_dynamic_shape(scores)
    scores = scores[:(scores_shape[0]-rem)]
    scores = tf.reshape(scores, fea0_shape[:2] + [1, 1])

    # Ignores postconvline
    self._joined_fea = fea0[:, :, tf.newaxis]

    return scores, None

  def _build(self, fea0, fea1, ind0, ind1,
             score_size, neg_fea, matched_class0,
             neg_matched_class, reuse_vars, scope):
    assert(score_size == 1)
    # Ignores postconvline
    self._joined_fea = fea0[:, :, tf.newaxis]

    fea0_shape = shape_utils.combined_static_and_dynamic_shape(fea0)
    if neg_fea is None:
      scores = tf.zeros(fea0_shape[:-1] + [1, 1], dtype=tf.float32)
      return scores, None

    scores = self._sim(fea0, neg_fea, scope)
    if self._mode == 'MAX':
      scores = tf.reduce_max(scores, axis=-2)
    elif self._mode == 'MEAN':
      scores = tf.reduce_mean(scores, axis=-2)
    elif self._mode == 'SOFTMAX':
      scale = util.get_temperature_variable('negative_softmax_param')
      w = tf.nn.softmax(scale*scores[..., 0])[..., tf.newaxis]
      scores = tf.reduce_sum(w*scores, axis=-2)

    ## -scores since negative instances get higher score
    scores = tf.reshape(-scores, fea0_shape[:-1] + [1, 1])
    return scores, None

class DoubleCrossSimilarity(CrossSimilarity):
  def __init__(self, cs0, cs1, mix_w=0.5, split_ind=None):
    super(DoubleCrossSimilarity, self).__init__()
    assert(cs0 is not None and cs1 is not None and split_ind is not None)
    self._cs0 = cs0
    self._cs1 = cs1
    self._mix_w = mix_w
    self._split_ind = split_ind
    if isinstance(cs0, PairwiseCrossSimilarity):
      cs0._fea_range = [None, split_ind]

    if isinstance(cs1, PairwiseCrossSimilarity):
      cs1._fea_range = [split_ind, None]


  def _split_fea(self, fea):
    if fea is None:
      return [None, None]

    if self._split_ind is None:
      if self._cs0 is None:
        return [None, fea]
      if self._cs1 is None:
        return [fea, None]
      raise Exception('Should not reach here!')

    return fea[..., :self._split_ind], fea[..., self._split_ind:]

  def _build(self, fea0, fea1, ind0, ind1,
             score_size, neg_fea, matched_class0,
             neg_matched_class, reuse_vars, scope,
             **kwargs):
    fea0 = self._split_fea(fea0)
    fea1 = self._split_fea(fea1)
    neg_fea = self._split_fea(neg_fea)
    joined_fea = []
    loss0, loss1, scores0, scores1 = None, None, 0.0, 0.0

    if self._mix_w > 0.0:
      self._cs0._target_score_inds = self._target_score_inds
      self._cs0.set_joining_convline(self._joining_convline)
      with tf.variable_scope('A') as scope:
        scores0, loss0 = self._cs0._build(fea0[0], fea1[0], ind0, ind1,
                                          score_size, neg_fea[0], matched_class0,
                                          neg_matched_class, reuse_vars, scope,
                                          **kwargs)
        joined_fea.append(self._cs0._joined_fea)

    if self._mix_w < 1.0:
      ### CS1 always provides agnostic predictions
      score_size = 1
      self._cs1._target_score_inds = None
      self._cs1.set_joining_convline(self._joining_convline)

      with tf.variable_scope('B') as scope:
        scores1, loss1 = self._cs1._build(fea0[1], fea1[1], ind0, ind1,
                                         score_size, neg_fea[1], matched_class0,
                                         neg_matched_class, reuse_vars, scope,
                                         **kwargs)
        joined_fea.append(self._cs1._joined_fea)


    scores = self._mix_w * scores0 + (1.0 - self._mix_w) * scores1

    if loss0 is not None and loss1 is not None:
      loss = loss0 + loss1
    elif loss0 is not None:
      loss = loss0
    elif loss1 is not None:
      loss = loss1
    else:
      loss = None

    self._joined_fea = tf.concat(joined_fea, axis=-1)
    return scores, loss

  @property
  def support_pre_joining(self):
    return True

class DeepCrossSimilarity(CrossSimilarity):
  def __init__(self, stop_gradient,
                fc_hyperparameters, convline,
                negative_attention,
                sum_output=False):
    super(DeepCrossSimilarity, self).__init__()
    self._fc_hyperparameters = fc_hyperparameters
    self._convline = convline
    self._stop_gradient = stop_gradient
    self._negative_attention = negative_attention
    self._sum_output = False#sum_output

  @property
  def support_pre_joining(self):
    return True

  def _build_(self, fea0, fea1, ind0, ind1,
             score_size, neg_fea, matched_class0,
             neg_matched_class, reuse_vars, scope,
             tile_fea1=True):
    print('Warning: Do not use this function (DeepCrossSimilarity._build) for training')
    if fea1 is None or not tile_fea1:
      a,b = self._build_inner(fea0, fea1, ind0, ind1,
                               score_size, neg_fea, matched_class0,
                               neg_matched_class, reuse_vars, scope,
                               tile_fea1)
      return a,b

    def fn(fea0):
      fea0 = fea0[tf.newaxis]
      scores, loss = self._build_inner(fea0, fea1, ind0, ind1,
                               score_size, neg_fea, matched_class0,
                               neg_matched_class, reuse_vars, scope,
                               tile_fea1)
      return scores[0]

    mini_bs = 64
    fea0 = fea0[0]

    fea0_shape = shape_utils.combined_static_and_dynamic_shape(fea0)
    rem = tf.mod(mini_bs - tf.mod(fea0_shape[0], mini_bs), mini_bs)

    fea0 = tf.pad(fea0, [[0, rem], [0,0]])
    fea0 = tf.reshape(fea0, [-1, mini_bs, fea0_shape[-1]])

    scores = tf.map_fn(
                      fn,
                      fea0,
                      dtype=tf.float32,
                      parallel_iterations=1,
                      back_prop=False,
                      swap_memory=True,
                      infer_shape=True,
                      name='efficient_memory_deep_cs')
    scores_shape = shape_utils.combined_static_and_dynamic_shape(scores)
    scores = tf.reshape(scores, [-1]+scores_shape[2:])
    scores_shape = shape_utils.combined_static_and_dynamic_shape(scores)
    scores = scores[:(scores_shape[0]-rem)]
    return scores[tf.newaxis], None

  def _build_symmetric(self, fea0, fea1, ind0, ind1,
             score_size, neg_fea, matched_class0,
             neg_matched_class, reuse_vars, scope,
             tile_fea1=True):
      ret =  self._build_non_symmetric(fea0, fea1, ind0, ind1,
                                       score_size, neg_fea, matched_class0,
                                       neg_matched_class, reuse_vars, scope,
                                       tile_fea1)

      if tile_fea1 == False or fea1 is None or neg_fea is not None:
        return ret


      with tf.variable_scope(scope, '', reuse=True):
        ret2 =  self._build_non_symmetric(fea1, fea0, ind1, ind0,
                                          score_size, neg_fea, matched_class0,
                                          neg_matched_class, True, scope,
                                          tile_fea1)
      scores = (ret[0] + tf.transpose(ret2[0], (0,2,1,3)))/2

      negative_loss = None
      if ret[1] is not None and ret2[1] is not None:
        negative_loss = (ret[1] + ret2[1])/2

      return scores, negative_loss

  def _build(self, fea0, fea1, ind0, ind1,
             score_size, neg_fea, matched_class0,
             neg_matched_class, reuse_vars, scope,
             tile_fea1=True):
    fea0_shape = shape_utils.combined_static_and_dynamic_shape(fea0)
    assert(len(fea0_shape) == 3)
    m = fea0_shape[1]

    if fea1 is None:
      # shape: [N, M, 1, K]
      fea01 = fea0[:, :, tf.newaxis]
    else:
      fea1_shape = shape_utils.combined_static_and_dynamic_shape(fea1)

      if tile_fea1:
        assert(len(fea1_shape) == 3)
        l = fea1_shape[1]
        # shape = [N, M, L, K]
        fea1 = tf.tile(fea1[:, tf.newaxis], [1,m,1,1])
      else:
        assert(len(fea1_shape) == 4)
        l = fea1_shape[2]

      # shape = [N, M, L, K]
      fea0 = tf.tile(fea0[:, :, tf.newaxis], [1,1,l,1])
      # shape = [N, M, L, 2K]
      fea01 = tf.concat((fea0, fea1), axis=-1)


      ### Rearange the order of the input features
      ### This is required to have a symmetric funciton
      if False: #neg_fea is None:
        fea01_shape = shape_utils.combined_static_and_dynamic_shape(fea01)
        rfea01 = tf.reshape(fea01, fea0_shape[:2] + [l, 2, -1])
        absum  = tf.reduce_sum(tf.abs(rfea01), axis=-1)
        order = tf.argmax(absum, axis=-1)
        rf = tf.reshape(rfea01, [-1, 2,fea0_shape[-1]])
        oo = tf.reshape(order, [-1,1])
        xfea0 = util.batched_gather(oo, rf)
        xfea1 = util.batched_gather(1-oo, rf)
        xfea01 = tf.concat((xfea0,xfea1), axis=1)
        fea01 = tf.reshape(xfea01, fea01_shape)


    # shape = [N, 1, M, L, 2K] or [N, 1, M, 1, K]
    fea01 = fea01[:, tf.newaxis]


    if self._joining_convline is not None:
      fea01 = self._joining_convline.build(fea01, scope='join')
    self._joined_fea = tf.squeeze(fea01, 1)

    if self._stop_gradient:
      fea01 = tf.stop_gradient(fea01)

    #### ADD BACKGROUND FEATURE
    negative_loss = None
    if neg_fea is not None:
      if self._negative_attention is None: #just averaged
        #neg_fea is [MBS, 1, 1, 1, d]
        #fea01 is [MBS*k_shot, 1, M, P, C] where P is 1 or L
        fea01_shape = shape_utils.combined_static_and_dynamic_shape(fea01)
        neg_fea_shape = shape_utils.combined_static_and_dynamic_shape(neg_fea)
        mbs = neg_fea_shape[0]
        k = fea01_shape[0]//mbs
        util.add_extra_tensor('fea01_before', fea01)
        # list of size MBS with [1,1,1,1,d] shape tensors
        util.add_extra_tensor('neg_fea_before', neg_fea)
        neg_fea_mbs = tf.split(neg_fea, mbs, axis=0)
        # list of size MBS with [k, 1, M, P, d] shape tensors
        util.add_extra_tensor('neg_fea_mbs', neg_fea_mbs)
        neg_fea_mbs_tiled = [tf.tile(n, [k, 1, fea01_shape[2],
                             fea01_shape[3], 1]) for n in neg_fea_mbs]
        util.add_extra_tensor('neg_fea_mbs_tiled', neg_fea_mbs_tiled)
        # neg_fea will be [MBS*k, 1, M, P, d] tensor
        neg_fea = tf.concat(neg_fea_mbs_tiled, axis=0)
        util.add_extra_tensor('neg_fea', neg_fea)
        #fea01 becomes [MBS*k_shot, 1, M, P, C+d]
        fea01 = tf.concat((fea01, neg_fea), axis=-1)
        util.add_extra_tensor('fea01_after', fea01)
      else:
        #[MBS*k_shot, 1, M, 1, C]
        #fea01 = tf.Print(fea01, tf.nn.moments(fea01, axes=[0,1,2,3,4]) +
        #                        tf.nn.moments(neg_fea, axes=[0,1,2,3,4]))
        fea01, negative_loss = self._negative_attention.build(
                                fea01, neg_fea, matched_class0,
                                neg_matched_class, reuse_vars)
    #### 

    # shape = [N, 1, M', M, 2K] or [N, 1, M, 1, K]
    if self._convline is not None:
      fea01 = self._convline.build(fea01, scope='convline')

    with slim.arg_scope(self._fc_hyperparameters):
      with slim.arg_scope([slim.fully_connected],
                          activation_fn=None,
                          normalizer_fn=None,
                          normalizer_params=None):
        scores = slim.fully_connected(fea01, score_size, scope='score')
        if self._sum_output:
          with tf.variable_scope('score', reuse=True):
            w = tf.get_variable('weights')
            b = tf.get_variable('biases')
            sw = tf.reduce_sum(w, axis=-1)
            sb = tf.reduce_sum(b)
            scores = (tf.tensordot(fea01, w[:, 0], [[4], [0]])
                      + sb)[..., tf.newaxis]

        scores = tf.squeeze(scores, 1)

        if self._target_score_inds is not None and score_size > 1:
          score_inds = self._target_score_inds[..., tf.newaxis]
          scores_trans = tf.transpose(scores, [0, 3, 1, 2])
          scores_trans = util.batched_gather(score_inds, scores_trans)
          scores = tf.transpose(scores_trans, [0, 2, 3, 1])

        return scores, negative_loss

class LinearCrossSimilarity(CrossSimilarity):
  def __init__(self, fc_hyperparameters):
    super(LinearCrossSimilarity, self).__init__()
    self._fc_hyperparameters = fc_hyperparameters

  def _build(self, fea0, fea1, ind0, ind1,
             score_size, neg_fea, matched_class0,
             neg_matched_class, reuse_vars, scope):
    with slim.arg_scope(self._fc_hyperparameters):
      with slim.arg_scope([slim.fully_connected],
                          activation_fn=None,
                          normalizer_fn=None,
                          normalizer_params=None):
        fea0 = tf.sigmoid(fea0)
        fea1 = tf.sigmoid(fea1)
        diff = tf.abs(fea0[:,:, tf.newaxis] - fea1[:, tf.newaxis])
        scores = slim.fully_connected(diff, score_size)
        return scores, None
