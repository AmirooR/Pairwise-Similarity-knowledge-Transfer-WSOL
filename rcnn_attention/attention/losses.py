import tensorflow as tf
from abc import abstractmethod
import object_detection.utils.shape_utils as shape_utils
from approx_kl_divergence import approx_kl_divergence
from util import add_extra_tensor
import numpy as np
from rcnn_attention.attention import util


def _partitions_dist(k, witness_rate):
    dist = tf.contrib.distributions.Binomial(total_count=float(k),
                                             probs=witness_rate)

    # Number of targets T = [0... k] has bionomial distribution
    binom_pdf = dist.prob(tf.range(float(k)+1))

    # Max number of similar objects S=[1...k] distirbution.
    # For S=2..k is the same as binom_pdf but for S=1 it is the sum
    # of binom_pdf[0] and binom_pdf[1].
    binom_pdf_unstack = tf.unstack(binom_pdf)
    pdf = tf.concat([[binom_pdf_unstack[0]+binom_pdf_unstack[1]] +
                       binom_pdf_unstack[2:]], axis=0)
    return pdf


def _clear_negative_classes(class_hist, neg_class_hist):
    ## "not", "tile", and then "and" with class_hist
    if neg_class_hist is not None:
      # [MBS, num_classes + 1]
      neg_shape = shape_utils.combined_static_and_dynamic_shape(neg_class_hist)
      # [MBS*N, B, num_classes + 1]
      class_hist_shape = shape_utils.combined_static_and_dynamic_shape(class_hist)

      N = class_hist_shape[0]//neg_shape[0]
      B = class_hist_shape[1]
      add_extra_tensor('neg_class_hist_N{}'.format(N), neg_class_hist)
      add_extra_tensor('class_hist_N{}'.format(N), class_hist)
      # [MBS, 1, 1, num_classes + 1]
      neg_class_hist = neg_class_hist[:, tf.newaxis, tf.newaxis]
      # [MBS, N, B, num_classes + 1]
      neg_class_hist = tf.tile(neg_class_hist, [1, N, B, 1])
      neg_class_hist = tf.reshape(neg_class_hist, class_hist_shape)
      not_neg_class_hist = tf.logical_not(neg_class_hist)
      add_extra_tensor('not_neg_class_hist_tiled_N{}'.format(N), not_neg_class_hist)

      not_neg_class_hist = tf.cast(not_neg_class_hist, dtype=class_hist.dtype)
      class_hist = tf.multiply(class_hist, not_neg_class_hist)
      add_extra_tensor('final_class_hist_N{}'.format(N), class_hist)
    ##
    return class_hist

def _soft_labels(class_hist, neg_class_hist, k, label_type='entropy'):
  non_neg_class_hist = _clear_negative_classes(class_hist, neg_class_hist)
  if label_type == 'entropy':
    class_hist = class_hist/k
    if k == 1:
      entropy = 0.0
    else:
      entropy = - class_hist * tf.log(tf.clip_by_value(class_hist,
                                            1e-13, 1.0))/np.log(k)
      entropy = tf.reduce_sum(entropy, axis=-1)
    non_neg_rate = tf.reduce_sum(non_neg_class_hist, axis=-1)/k
    soft_labels = (1.0 - entropy) * non_neg_rate
  elif label_type == 'target_rate':
    soft_labels = tf.reduce_max(non_neg_class_hist, axis=-1)/k
  elif label_type == 'hard_cutoff':
    threshold = 0.98
    target_rate = tf.reduce_max(non_neg_class_hist, axis=-1)/k
    # StepFunction_{threhold}(target_rate)
    soft_labels = tf.maximum(0.0, tf.sign(target_rate - threshold))
  return tf.stop_gradient(soft_labels)

def _pairwise_scores(class_hist0, class_hist1, neg_class_hist,
                     type='sum'):
    pairwise_scores = tf.multiply(class_hist0, class_hist1)
    pairwise_scores = _clear_negative_classes(pairwise_scores,
                                              neg_class_hist)
    if type == 'sum':
        return tf.reduce_sum(pairwise_scores[...,1:], axis=-1)
    elif type == 'max':
      return tf.reduce_max(pairwise_scores[...,1:], axis=-1)
    raise Exception('type {} is not supported'.format(type))

def _onehot_labels(class_hist, min_nmatch, neg_class_hist):
    '''
      If the highest value in the class_hist is more than the required
      threshold the label would be the corresponding class for the highest
      value. Otherwise, it would be background. Length of the one_hot label
      might be num_classes + 1 or 2 depending on the class_agnostic value.
    '''
    # Match happens when at least min_nmatch 
    # objects belongs to the same foreground class.
    labels = tf.less_equal(np.float32(min_nmatch), class_hist)

    labels = tf.to_float(labels)
    labels = _clear_negative_classes(labels, neg_class_hist)
    labels = tf.cast(labels, dtype=tf.bool)[..., 1:]

    num_classes = shape_utils.combined_static_and_dynamic_shape(labels)[-1]

    # Choose at most one positive label
    argmax = tf.argmax(class_hist[..., 1:], axis=-1)
    optim_labels = tf.cast(tf.one_hot(argmax, num_classes), tf.bool)
    labels = tf.logical_and(labels, optim_labels)

    fg = tf.reduce_any(labels, axis=-1,
                        keep_dims=True)

    bg = tf.logical_not(fg)
    return labels, fg, bg

class Loss(object):
  def __init__(self, class_agnostic, num_classes, name, is_training=None):
    self._class_agnostic = class_agnostic
    self._num_classes = num_classes
    self._name = name
    self._is_training = is_training
    self._bag_target_class = None

  @property
  def score_size(self):
    if self._class_agnostic:
      return 2
    else:
      return self._num_classes + 1

  def energy(self, scores):
    '''
      Normalize the scores and returns the normalized scores
      and the energy. Note that the normalization
      function might be different with respect to loss.
    '''
    with tf.variable_scope('energy', values=[scores]):
      return self._energy(scores)

  @abstractmethod
  def _energy(self, scores):
    pass

  def multiclass_energy(self, scores):
    '''
      Return multiclass version of the scores depending on the loss type.
      Currently, softmax score is the only loss which is not returning None.
    '''
    return self._multiclass_energy(scores)

  #@abstractmethod
  def _multiclass_energy(self, scores):
    return None

  def provide_bag_target_class(self, bag_target_class):
    self._bag_target_class = bag_target_class

  def normalize_labels(self, class_hist,
                       class_hist0,
                       class_hist1,
                       neg_class_hist):
    '''
      Given the class histogram of each co-object. Returns
      normalized labels that could be used in the loss function
      and a logical tensor which shows the the foreground co-objects.

      class_hist: class histogram tensor with size [..., num_classes + 1]
    '''

    with tf.variable_scope('normalize_labels', values=[class_hist]):
      return self._normalize_labels(class_hist, class_hist0,
                                    class_hist1, neg_class_hist)

  @abstractmethod
  def _normalize_labels(self, class_hist,
                        class_hist0,
                        class_hist1,
                        neg_class_hist):
    pass

  def loss(self, scores, labels, weights=None):
    '''
      Returns total and instance loss tensors.
        scores: un-normalized scores with shape [..., score_size]
        labels: normalized labels.
    '''
    assert(scores.shape[-1] == self.score_size
            ), '{} vs {}'.format(scores.shape[-1], self.score_size)
    with tf.variable_scope('loss', values=[scores, labels]):
      loss_tensor = self._loss(scores, labels)
      if weights is None:
        loss = tf.reduce_mean(loss_tensor)
      else:
        w = tf.reduce_sum(weights) + 1e-6
        loss =tf.reduce_sum(loss_tensor * weights)/w
      #unit_ind = scores.name.find('Unit')
      #util.add_extra_tensor('loss_' + scores.name[unit_ind:unit_ind+5], loss)
      return loss, loss_tensor

  @abstractmethod
  def _loss(self, scores, labels):
    pass

class RankLoss(Loss):
  def __init__(self, k):
    super(RankLoss, self).__init__(True, 1, 'rank')
    self._k = k

  @property
  def score_size(self):
    return 1
  def _energy(self, scores):
    return scores[..., 0]

  def _normalize_labels(self, class_hist, class_hist0,
                        class_hist1, neg_class_hist):
    rank = _pairwise_scores(class_hist0,
                            class_hist1,
                            neg_class_hist,
                            type='sum')
    is_fg = tf.equal(rank, int(self._k/2)**2)
    add_extra_tensor('is_fg', is_fg)
    add_extra_tensor('rank', rank)
    return rank, is_fg

  def _loss(self, scores, labels):
    scores = tf.squeeze(scores, axis=-1)
    diff_scores = scores[..., tf.newaxis] - scores[:,tf.newaxis]
    diff_labes = labels[..., tf.newaxis] - labels[:, tf.newaxis]
    diff_labes = (1.0 + diff_labes/(self._k/2.0)**2)/2.0
    add_extra_tensor('diff_labes', diff_labes)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=diff_labes,
                                                   logits=diff_scores)
    return tf.reduce_mean(loss, axis=-1)


class PairwiseEstimateLossV2(Loss):
  def __init__(self, class_agnostic,
               num_classes, k, loss_type='l2',
               is_training=None):
    '''
      It classifies each co-object to its assign label independently.
    '''
    assert(class_agnostic == True and k > 1)
    super(PairwiseEstimateLossV2, self).__init__(class_agnostic, num_classes,
                                               'pairwise_estimate_v2', is_training)
    self._k = k
    self._loss_type = loss_type

  @property
  def score_size(self):
    return 1

  @property
  def _num_pairwise(self):
    return int ((self._k/2)**2)

  def _energy(self, scores):
    return self._num_pairwise * scores[..., 0]

  def _normalize_labels(self, class_hist, class_hist0,
                        class_hist1, neg_class_hist):
    neg_class_hist = None
    energy = _pairwise_scores(class_hist0,
                              class_hist1,
                              neg_class_hist,
                              type='max')

    is_fg = tf.equal(energy, self._num_pairwise)
    return energy[...,tf.newaxis]/self._num_pairwise, is_fg

  def _loss(self, scores, labels):
    add_extra_tensor('pw_scores_k{}'.format(self._k), scores)
    add_extra_tensor('pw_labels_k{}'.format(self._k), labels)

    if self._loss_type == 'cross_entropy':
      loss = tf.nn.sigmoid_cross_entropy_with_logits(
                      labels=labels, logits=scores)
    elif self._loss_type == 'l2':
      diff = scores - labels
      loss = 0.5 * tf.multiply(diff, diff)

    return tf.reduce_sum(loss, axis=-1)


class PairwiseEstimateLoss(Loss):
  def __init__(self, class_agnostic,
               num_classes, k, is_training=None):
    '''
      It classifies each co-object to its assign label independently.
    '''
    assert(class_agnostic == True and k > 1)
    super(PairwiseEstimateLoss, self).__init__(class_agnostic, num_classes,
                                               'pairwise_estimate', is_training)
    self._k = k

  @property
  def score_size(self):
    return int((self._k/2)**2)

  def _energy(self, scores):
    if False:
      scores = tf.log(tf.nn.sigmoid(scores))
    return tf.reduce_sum(scores, axis=-1)
  def _normalize_labels(self, class_hist, class_hist0,
                        class_hist1, neg_class_hist):

    add_extra_tensor('class_hist_k{}'.format(self._k), class_hist)
    add_extra_tensor('class_hist0_k{}'.format(self._k), class_hist0)
    add_extra_tensor('class_hist1_k{}'.format(self._k), class_hist1)
    neg_class_hist = None
    energy = _pairwise_scores(class_hist0,
                              class_hist1,
                              neg_class_hist,
                              type='max')
    labels = []
    for i in range(self.score_size):
      labs = tf.greater(energy, i)
      labels.append(labs)
    labels = tf.cast(tf.stack(labels, axis=-1),
                     dtype=tf.float32)

    add_extra_tensor('labels_k{}'.format(self._k), labels)
    add_extra_tensor('labs_k{}'.format(self._k), labs)
    add_extra_tensor('energy_k{}'.format(self._k), energy)
    return labels, labs

  def _loss(self, scores, labels):
    add_extra_tensor('pw_scores_k{}'.format(self._k), scores)
    add_extra_tensor('pw_labels_k{}'.format(self._k), labels)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
                      labels=labels, logits=scores)

    if True:
      gamma = 2.0
      alpha = 1.0
      sigm = tf.nn.sigmoid(scores)
      weights = tf.where(tf.cast(labels, dtype=tf.bool),
                                      1.0 - sigm, sigm)

      weights = alpha * tf.pow(weights, gamma)
      loss = weights * loss
      add_extra_tensor('weights', weights)
      add_extra_tensor('sigm', sigm)
      add_extra_tensor('labels', labels)
      add_extra_tensor('loss_k{}'.format(self._k), loss)

    return tf.reduce_sum(loss, axis=-1)

class SigmoidCrossEntropyLoss(Loss):
  def __init__(self, agnostic, num_classes, k, focal_loss=False):
    '''
      It classifies each co-object to its assign label independently.
    '''
    super(SigmoidCrossEntropyLoss, self).__init__(agnostic, num_classes,
                                                  'sigmoid_cross_entropy')
    self._k = k
    self._focal_loss = focal_loss

  @property
  def score_size(self):
    if self._class_agnostic:
      return 1
    else:
      return self._num_classes

  def _energy(self, scores):
    assert(len(scores.shape) == 3)
    return tf.reduce_max(scores, axis=-1)

  def _normalize_labels(self, class_hist, class_hist0,
                        class_hist1, neg_class_hist):
    # Match happens when at least min_match_frac*k objects
    # belongs to the same foreground class.
    labels, fg, bg = _onehot_labels(class_hist, self._k, neg_class_hist)
    add_extra_tensor('class_hist_{}'.format(self._k), class_hist)

    if self._class_agnostic:
      return (tf.to_float(fg),
              tf.squeeze(fg, axis=-1))
    else:
      return (tf.to_float(labels),
              tf.squeeze(fg, axis=-1))

  def _loss(self, scores, labels):
    def _target_vals(tensor):
      if self._bag_target_class is None:
        return tensor
      score_inds = tf.cast(self._bag_target_class[..., tf.newaxis], tf.int32) - 1
      tr_tensor = tf.transpose(tensor, [0,2,1])
      tr_tensor = util.batched_gather(score_inds, tr_tensor)
      return tf.transpose(tr_tensor, [0,2,1])

    add_extra_tensor('scores_b{}'.format(self._k), scores)
    add_extra_tensor('labels_b{}'.format(self._k), labels)

    scores = _target_vals(scores)
    labels = _target_vals(labels)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(
              labels=labels,
              logits=scores)
    add_extra_tensor('scores_{}'.format(self._k), scores)
    add_extra_tensor('labels_{}'.format(self._k), labels)
    add_extra_tensor('loss_{}'.format(self._k), loss)

    if self._focal_loss:
      gamma = 2.0
      alpha = 1.0
      sigm = tf.nn.sigmoid(scores)
      weights = tf.where(tf.cast(labels, dtype=tf.bool),
                                      1.0 - sigm, sigm)

      weights = alpha * tf.pow(weights, gamma)
      loss = weights * loss
      add_extra_tensor('weights_{}'.format(self._k), weights)
      add_extra_tensor('sigm_{}'.format(self._k), sigm)

    return tf.reduce_sum(loss, axis=-1)

class SoftSigmoidCrossEntropyLoss(SigmoidCrossEntropyLoss):
  def __init__(self, k, label_type='entropy'):
    super(SoftSigmoidCrossEntropyLoss, self).__init__(True, 1, k)
    self._label_type = label_type

  def _normalize_labels(self, class_hist, class_hist0,
                        class_hist1, neg_class_hist):
    soft_labels = _soft_labels(class_hist,
                               neg_class_hist,
                               self._k,
                               label_type=self._label_type)

    #add_extra_tensor('soft_labels_{}'.format(self._k), soft_labels)
    #add_extra_tensor('class_hist_{}'.format(self._k), class_hist)
    return (soft_labels[..., tf.newaxis],
            tf.less_equal(0.98, soft_labels))

class SoftmaxCrossEntropyLoss(Loss):
  def __init__(self, class_agnostic,
               num_classes, k,
               min_match_frac=1.0,
               name='softmax_cross_entropy'):
    '''
      It classifies each co-object to its assign label independently.
    '''
    super(SoftmaxCrossEntropyLoss, self).__init__(class_agnostic, num_classes,
                                                  name)
    self._min_nmatch = np.round(k*min_match_frac)

  def _energy(self, scores):
    assert(len(scores.shape) == 3)
    eq_scores = tf.reduce_logsumexp(scores[..., 1:], axis=-1)
    return eq_scores - scores[..., 0]

  def _multiclass_energy(self, scores):
    if self._class_agnostic:
      return None
    return scores

  def _normalize_labels(self, class_hist, class_hist0,
                        class_hist1, neg_class_hist):
    '''
      If the highest value in the class_hist is more than the required
      threshold the label would be the coresspoing class for the highest
      value. Otherwise, it would be background. Lenght of the one_hot label
      might be num_classes + 1 or 2 depending on the class_agnostic value.
    '''
    # Match happens when at least min_match_frac*k objects
    # belongs to the same foreground class.
    labels, fg, bg = _onehot_labels(class_hist, self._min_nmatch, neg_class_hist)
    if self._class_agnostic:
      onehot_labels = tf.concat([bg, fg], axis=-1)
    else:
      onehot_labels = tf.concat([bg, labels], axis=-1)
    return (tf.to_float(onehot_labels),
            tf.squeeze(fg, axis=-1))

  def _loss(self, scores, labels):
    return tf.nn.softmax_cross_entropy_with_logits(
              labels=labels,
              logits=scores)

class OnlyForTestingAverage(SoftmaxCrossEntropyLoss):
  def __init__(self, class_agnostic,
               num_classes, k,
               min_match_frac=1.0):
    super(OnlyForTestingAverage, self).__init__(class_agnostic, num_classes,
                                              k, min_match_frac,
                                              'only_for_testing_average')
  def _energy(self, scores):
      return tf.reduce_max(scores, axis=-1)

class L2Loss(Loss):
  def __init__(self, class_agnostic,
               num_classes, k,
               min_match_frac=1.0):
    '''
      It classifies each co-object to its assign label independently.
    '''
    assert(class_agnostic)
    super(L2Loss, self).__init__(class_agnostic, num_classes,
                                 'l2')
    self._min_nmatch = np.round(k*min_match_frac)

  @property
  def score_size(self):
    return 1

  def _normalize_scores(self, scores):
    return tf.squeeze(scores, axis=-1)

  def _normalize_labels(self, class_hist,
                        class_hist0, class_hist1,
                        neg_class_hist):
    '''
      If the highest value in the class_hist is more than the required
      threshold the label would be the coresspoing class for the highest
      value. Otherwise, it would be background. Lenght of the one_hot label
      might be num_classes + 1 or 2 depending on the class_agnostic value.
    '''
    # Match happens when at least min_match_frac*k objects
    # belongs to the same foreground class.
    labels, fg, bg = _onehot_labels(class_hist, self._min_nmatch, neg_class_hist)
    return (tf.to_float(fg if self._class_agnostic else labels),
            tf.squeeze(fg, axis=-1))

  def _loss(self, scores, labels):
    #return tf.reduce_mean(tf.square(scores - labels)/2.0, axis=-1)
    #return tf.reduce_mean(tf.abs(scores - labels), axis=-1)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                               logits=scores),
                          axis=-1)

class ApproxKLDivergenceLoss(Loss):
  def __init__(self, class_agnostic,
               num_classes,
               k, witness_rate,
               min_match_frac=1.0,
               partial_loss=False,
               skip_normalization=False):
    '''
     It treats the scores for all the co-objects as a distriubtion and
     compute the kl divergence between the predicted distribution and
     target distribution.
    '''
    assert(class_agnostic), 'For ApproxKLDivergenceLoss class agnostic should be True'
    super(ApproxKLDivergenceLoss, self).__init__(class_agnostic, num_classes,
                                                 'kl_divergence')
    self._partial_loss = partial_loss
    self._k = k
    self._witness_rate = witness_rate
    self._min_nmatch = np.round(k*min_match_frac)
    self._skip_normalization = skip_normalization

  @property
  def score_size(self):
    return 1

  def _normalize_labels(self, class_hist,
                        class_hist0, class_hist1,
                        neg_class_hist):
    '''
      If the highest value in the class_hist is more than the required
      threshold the label would be the coresspoing class for the highest
      value. Otherwise, it would be background. Lenght of the one_hot label
      might be num_classes + 1 or 2 depending on the class_agnostic value.
    '''
    # Match happens when at least min_match_frac*k objects
    # belongs to the same foreground class.
    labels, fg, bg = _onehot_labels(class_hist, self._min_nmatch, neg_class_hist)

    partitions = tf.reduce_max(class_hist[..., 1:], axis=-1) - 1
    partitions = tf.cast(partitions, tf.int32)
    util.add_extra_tensor('partitions_orig', partitions)
    util.add_extra_tensor('class_hist', class_hist)

    fg = tf.squeeze(fg, axis=-1)

    labels = [tf.to_float(fg), partitions]
    return (labels, fg)

  def _energy(self, scores):
    scores = tf.nn.softmax(scores[..., 0])
    return scores

  def _loss(self, scores, labels):
    p, partitions = labels
    scores = tf.squeeze(scores, axis=-1)
    p_shape = shape_utils.combined_static_and_dynamic_shape(p)

    partitions_dist = _partitions_dist(self._k, self._witness_rate)
    scale_factor = tf.reduce_min(partitions_dist)
    loss = approx_kl_divergence(p,
                                scores,
                                partitions=partitions,
                                partitions_dist=partitions_dist,
                                partial_loss=self._partial_loss,
                                partitions_dist_scale=scale_factor,
                                skip_normalization=self._skip_normalization)
    # Since the caller compute the  mean of the values
    # But we need to take sum
    loss = loss * p_shape[-1]
    return loss
