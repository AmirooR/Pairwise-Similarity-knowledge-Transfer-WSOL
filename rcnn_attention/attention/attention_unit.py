import tensorflow as tf
import util
import numpy as np
import object_detection.utils.shape_utils as shape_utils
from object_detection.core import balanced_positive_negative_sampler as sampler
from object_detection.utils import ops
from common_objects import CommonObjects
from attention_unit_result import AttentionUnitResult
import losses
#import nms_tf_wrapper

slim = tf.contrib.slim

class AttentionUnit(object):
  def __init__(self, ncobj_proposals, max_ncobj_proposals,
      positive_balance_fraction, k,
      pre_match_convline=None,
      post_match_convline=None, cross_similarity_cal=None,
      loss_cal=None,
      is_training=False,
      use_orig_feas_in_post_match_convline=True,
      subsample_hard_examples=False,
      stratified_subsampling=False,
      loss_positive_balance_fraction=0.5,
      loss_minibatch_size=128,
      parallel_iterations=16,
      loss_weight=1.0,
      negative_example_weight=1.0,
      compute_scores_after_matching=False,
      overwrite_fea_by_scores=False,
      negative_convline=None,
      is_calibration=False,
      calibration_type='MULTI_SCALES',
      unary_energy_scale=1.0,
      transfered_objectness_weight=0.0):

    self._ncobj_proposals = ncobj_proposals
    self._max_ncobj_proposals = max_ncobj_proposals
    self._positive_balance_fraction = positive_balance_fraction
    self._k = k
    self._pre_match_convline = pre_match_convline
    self._cross_similarity_cal = cross_similarity_cal
    self._loss_cal = loss_cal
    self._is_training = is_training
    self._use_orig_fea_in_post_match_convline = use_orig_feas_in_post_match_convline
    self._subsample_hard_examples = subsample_hard_examples
    self._stratified_subsampling = stratified_subsampling
    self._loss_positive_balance_fraction = loss_positive_balance_fraction
    self._loss_minibatch_size = loss_minibatch_size
    self._parallel_iterations = parallel_iterations
    self._negative_example_weight = negative_example_weight
    self._loss_weight = loss_weight
    self._post_match_convline = post_match_convline
    self._compute_scores_after_matching = compute_scores_after_matching
    if compute_scores_after_matching:
      # matching will be handled in the cross_similarity_cal
      cross_similarity_cal.set_joining_convline(post_match_convline)
    self._overwrite_fea_by_scores = overwrite_fea_by_scores
    self._negative_convline = negative_convline
    self._is_calibration = is_calibration
    self._calibration_type = calibration_type
    #self._calibration_loss = losses.SigmoidCrossEntropyLoss(k)
    self._calibration_loss = losses.SoftSigmoidCrossEntropyLoss(k,
                                             label_type='entropy')
    self._unary_energy_scale = unary_energy_scale

    assert(transfered_objectness_weight <= 1. and transfered_objectness_weight >= 0.)
    self._transfered_objectness_weight = transfered_objectness_weight

  def build(self, cobj0, cobj1,
                  neg_pooled_fea=None,
                  neg_target_cls=None,
                  neg_matched_class=None,
                  reuse_vars=False,
    ## Visualization code here
                  scope=None):
    values = cobj0.values
    if cobj1 is not None:
      values += cobj1.values
    with tf.variable_scope(scope, reuse=reuse_vars,
                          values=values):
      return self._build(cobj0, cobj1, neg_pooled_fea,
                         neg_target_cls, neg_matched_class, scope)

  def _build(self, cobj0, cobj1, neg_pooled_fea,
             neg_target_cls, neg_matched_class,
             scope):
    matched_class0, matched_class1 = None, None
    is_target0, is_target1 = None, None
    if self._negative_convline is not None and neg_pooled_fea is not None:
      neg_pooled_fea = self._negative_convline.build(neg_pooled_fea, scope='NegConvline')
    if cobj1 is None:
      fea0 = cobj0.get('fea')
      if self._pre_match_convline:
        fea0 = self._pre_match_convline.build(fea0, scope='Convline0')

      if not self._use_orig_fea_in_post_match_convline:
        cobj0.set('fea', fea0)
      ind0 = None
      ind1 = None
      if cobj0.has_key('ind'):
        ind0 = cobj0.get('ind')
      fea1 = None
      if cobj0.has_key('matched_class'):
        matched_class0 = cobj0.get('matched_class')
        is_target0 = cobj0.get('is_target')
    else:
      fea0 = cobj0.get('fea')
      fea1 = cobj1.get('fea')
      ind0 = None
      ind1 = None
      if cobj0.has_key('ind') and cobj1.has_key('ind'):
        ind0 = cobj0.get('ind')
        ind1 = cobj1.get('ind')
      if self._pre_match_convline:
        fea = tf.concat([fea0, fea1], 0)
        fea = self._pre_match_convline.build(fea, scope='Convline0')
        fea0, fea1 = tf.split(fea, 2)

      if not self._use_orig_fea_in_post_match_convline:
        cobj0.set('fea', fea0)
        cobj1.set('fea', fea1)

    score_size = self._loss_cal.score_size

    score_inds = None
    if not self._is_training and cobj0.has_key('bag_target_class'):
      bag_target_class = cobj0.get('bag_target_class')
      score_inds = tf.cast(bag_target_class, dtype=tf.int32) - 1

    (scores, pairs, joined_fea, negative_loss
        ) = self._cross_similarity_cal.build(fea0, fea1,
                                             ind0, ind1,
                                             score_size,
                                             neg_pooled_fea,
                                             matched_class0,
                                             neg_matched_class,
                                             target_score_inds=score_inds)

    (cobj, cobj_energy, loss, sampled_scores
        ) = self._sample_matches(cobj0, cobj1,
                                 scores,
                                 pairs,
                                 joined_fea,
                                 neg_target_cls)

    if self._k > 1:
      assert(neg_pooled_fea is None and
             neg_matched_class is None and
             negative_loss is None and
             neg_target_cls is None), 'Didn\'t expect that :('

    if negative_loss is not None and loss is not None:
      loss = loss + negative_loss

    if self._overwrite_fea_by_scores:
      scores_shape = shape_utils.combined_static_and_dynamic_shape(sampled_scores)
      assert(len(scores_shape)==3), 'scores shape should be 3D'
      nscores = tf.reshape(sampled_scores, scores_shape[:2]+[1,1, scores_shape[2]])
      cobj.set('fea', tf.nn.softmax(nscores)) #logit to probability

    ret = AttentionUnitResult(cobj, self._k,
                              self._loss_cal._class_agnostic,
                              fg_scores=cobj_energy,
                              loss=loss,
                              attention_unit=self,
                              scores=sampled_scores,
                              cross_similarity_scores=scores,
                              cross_similarity_pairs=pairs)

    return ret

  def score_instances(self, instance_fea,
                      instance_labels,
                      is_target,
                      unit_results,
                      min_match_frac=0.0,
                      rescore_convline=None,
                      rescore_fc_hyperparams=None):
    raise NotImplemented('Does not work with the new loss class')

  def _loss_weights(self, labels, is_fg):
    loss_positive_balance_fraction = self._loss_positive_balance_fraction

    ## If true sample uniformly from all the scores
    if self._loss_positive_balance_fraction <= 0:
      loss_positive_balance_fraction = 1.0

    loss_sampler = sampler.BalancedPositiveNegativeSampler(
                    positive_fraction=loss_positive_balance_fraction)

    def _minibatch_subsample_fn(labels):
      indicators = tf.ones_like(labels)
      ## If true samples uniformly from scores
      if self._loss_positive_balance_fraction <= 0:
        labels = indicators

      return loss_sampler.subsample(
          indicators,
          self._loss_minibatch_size, labels)

    ffg = tf.to_float(is_fg)
    weights = (1-ffg) * self._negative_example_weight + ffg

    ## if _loss_minibatch_size == 0 do not do subsampling
    if self._loss_minibatch_size > 0:
      batch_sampled_indices = tf.to_float(tf.map_fn(
          _minibatch_subsample_fn,
          is_fg,
          dtype=tf.bool,
          parallel_iterations=self._parallel_iterations,
          back_prop=True))
      weights *= batch_sampled_indices
    return weights * self._loss_weight

  def _subsample_from_topk(self, is_target,
                           scores,
                           difficulties,
                           positive_fraction):
    if self._subsample_hard_examples:
      _, topk = tf.nn.top_k(difficulties, k=self._ncobj_proposals)
      return topk
    else:
      n_proposals = scores.shape[1]
      if n_proposals is None:
        n_proposals = tf.shape(scores)[1]
        ntop = tf.minimum(self._max_ncobj_proposals, n_proposals)
      else:
        ntop = min(self._max_ncobj_proposals, n_proposals)

      ### TODO: it was k=n_proposals for cvpr experiments
      ## that allowed the method to sample from all
      ## the cobjs. It is a good practice to see how
      ## it changes the cvpr resuls?
      _, topk = tf.nn.top_k(scores, k=ntop)

      def fn(x, positive_fraction=positive_fraction):
        '''
          pick indicies from valid_inds. If
          positive_fraction is not None use labels
          to do balanecd sampling.
        '''
        labels, valid_inds = x
        indicators = ops.indices_to_dense_vector(valid_inds,
                                                 tf.shape(labels)[0])
        indicators = tf.cast(indicators, tf.bool)
        if positive_fraction is None:
          positive_fraction = 1.0
          labels = tf.ones_like(labels, tf.bool)
        return util.balanced_subsample(indicators,
                                       labels,
                                       self._ncobj_proposals,
                                       positive_fraction)
      return tf.map_fn(fn, [is_target, topk],
                       dtype=tf.int64,
                       back_prop=True,
                       name='batched_subsample')

  def _subsample_stratified(self, difficulties,
                            matched_class,
                            target_class):
    n = self._k + 1
    strata = tf.range(n)
    assert(self._ncobj_proposals % n == 0
        ), 'Only works for ncobj_proposals % k+1 == 0'
    num_samples = self._ncobj_proposals // n

    assert(self._ncobj_proposals >= n), 'Not enought ncobj_proposals for uniform sampling'

    # Sample uniformly from co-objects with
    # 0,..., k+1 target instances
    def fn(x):
      matched_cls, target_ind, difficulties = x
      num_matches = util.batched_gather(target_ind[..., tf.newaxis],
                                        matched_cls)[..., 0]
      # Sample num_samples from co-objects in
      # which there are exaclty m target instances
      def _sample_from_m_matches(m):
        indicators = tf.equal(num_matches, tf.cast(m, tf.float32))
        ### debug
        # hist = tf.bincount(tf.cast(num_matches, tf.int32), minlength=n, maxlength=n)
        #indicators = tf.Print(indicators, [m, self._k, num_matches, hist],
        #                      summarize=1000)
        ####
        if self._subsample_hard_examples:
          return util.topk_or_pad_inds_with_resampling(indicators, difficulties, num_samples)
        else:
          return util.subsample(indicators, num_samples)

      inds = tf.map_fn(_sample_from_m_matches,
                       [strata],
                       dtype=tf.int64,
                       back_prop=False,
                       name='sample_from_k_matches')

      return tf.reshape(inds, [-1])

    # target_ind = 1 if there is no target_class
    target_ind = tf.argmax(tf.cast(target_class[..., 1:], tf.int32), -1) + 1
    sampled_inds = tf.map_fn(fn,
                        [matched_class, target_ind, difficulties],
                        dtype=tf.int64,
                        back_prop=False,
                        name='sample_cobjs')
    return sampled_inds

  def _subsample_inds(self, cross_co, instance_loss):
    cobj_energy = cross_co.get('energy')
    if self._is_training:
      if self._stratified_subsampling:
        ## RANDOM/HARD_EXAMPLE SAMPLING FROM STRATA
        return self._subsample_stratified(instance_loss,
                                          cross_co.get('matched_class'),
                                          cross_co.get('target_class'))
      else:
        ## RANDOM/HARD_EXAMPLE (optionally +BALANCED) SAMPLING FROM TOP-K
        positive_fraction = self._positive_balance_fraction
        return self._subsample_from_topk(cross_co.get('is_target'),
                                         cobj_energy,
                                         instance_loss,
                                         positive_fraction)
    else:
      #if cross_co.has_key('boxes'):
      #  nms_idx = nms_tf_wrapper.batch_nms(cross_co.get('boxes'),
      #                                     cobj_energy,
      #                                     self._ncobj_proposals,
      #                                     threshold=0.5, use_py_fn=True)
        #util.add_extra_tensor('boxes_{}'.format(self._k), cross_co.get('boxes'))
        #util.add_extra_tensor('energy_{}'.format(self._k), cobj_energy)
        #util.add_extra_tensor('nms_idx_{}'.format(self._k), nms_idx)
        #nms_idx2 = nms_tf_wrapper.batch_nms(cross_co.get('boxes'),
        #                                    cobj_energy,
        #                                    self._ncobj_proposals,
        #                                    threshold=0.8,
        #                                    use_py_fn=False)
        #util.add_extra_tensor('nms_idx2_{}'.format(self._k), nms_idx2)

      ncobj = cobj_energy.shape[1]
      assert(ncobj is not None and ncobj >= self._ncobj_proposals)
      if self._ncobj_proposals == ncobj:
          return None
      _, topk_idx = tf.nn.top_k(cobj_energy,
                                k=self._ncobj_proposals,
                                sorted=False)
      #all_true = tf.ones_like(cobj_energy, dtype=tf.bool)
      #topk_idx = util.batched_topk_or_pad_inds_with_resampling(
      #                    all_true, cobj_energy,
      #                    self._ncobj_proposals)
      #from IPython import embed;embed()
      util.add_extra_tensor('topk_idx_{}'.format(self._k), topk_idx)
      return topk_idx

  def _sample_matches(self, cobj0, cobj1, scores,
                      pairs, joined_fea, neg_target_cls):

    def _unary_energy_scale(constant=True):
      scale = self._unary_energy_scale
      if constant:
        return scale
      else:
        var_name = 'k{}_unary_w'.format(self._k)
        return util.get_temperature_variable(var_name,
                                    initializer=scale)
    def _ek(k):
      return 'k{}_energy'.format(k)

    cross_co0 = cobj0.gather(pairs[..., 0])
    cross_co = cross_co0.copy()
    subproblems_energy = 0.0
    if cobj1 is not None:
      assert(self._k > 1)
      cross_co1 = cobj1.gather(pairs[..., 1])
      cross_co.join(cross_co1)

      # compute subproblems_energy
      if self._is_calibration:
        for i in range(int(np.log2(self._k))):
          k = 2**i
          if cross_co.has_key(_ek(k)):
            e = tf.reduce_sum(cross_co.get(_ek(k)), axis=-1)
            if k == 1:
              subproblems_energy += _unary_energy_scale() * e
            else:
              subproblems_energy += e
        assert(subproblems_energy.shape[:2] == scores.shape[:2]
                                             ), '{},{}'.format(
                                             subproblems_energy.shape,
                                             scores.shape)

    # compute unit_energy
    unit_energy = self._loss_cal.energy(scores)


    w = self._transfered_objectness_weight
    if cobj1 is None and self._is_calibration and w >= 0. and cross_co.has_key(_ek(0)):
      # transfered objectness will be added to unary energy for k1 only
      e0 = tf.reduce_sum(cross_co.get(_ek(0)), axis=-1)
      unit_energy = w * e0 + (1. - w) * unit_energy


    cross_co.set(_ek(self._k), unit_energy[..., tf.newaxis])
    # compute total_energy
    total_energy = subproblems_energy + unit_energy
    ## Compute the loss
    loss, instance_loss = None, None
    if self._is_training:
      matched_class = cross_co.get('matched_class')
      matched_class0 = cross_co0.get('matched_class')

      matched_class1 = (None if cobj1 is None else
                        cross_co1.get('matched_class'))

      if self._k > 1 and not self._is_calibration:
        neg_target_cls = None

      loss_cal = self._loss_cal

      if cobj0.has_key('bag_target_class'):
        bag_target_class = cross_co.get('bag_target_class')
        loss_cal.provide_bag_target_class(bag_target_class)

      if self._is_calibration and self._k > 1:
        loss_cal = self._calibration_loss
        scores = total_energy[..., tf.newaxis]

      labels, is_fg = loss_cal.normalize_labels(matched_class,
                                                matched_class0,
                                                matched_class1,
                                                neg_target_cls)
      weights = self._loss_weights(labels, is_fg)

      loss, instance_loss = loss_cal.loss(scores,
                                          labels,
                                          weights=weights)
    ###
    multiclass_energy = self._loss_cal.multiclass_energy(scores)
    cross_co.set('energy', total_energy)
    if multiclass_energy is not None:
      cross_co.set('multiclass_energy', multiclass_energy)
    top_indices = self._subsample_inds(cross_co, instance_loss)

    if top_indices is None:
        top_cobj = cross_co
    else:
        top_cobj = cross_co.gather(top_indices)

    if self._compute_scores_after_matching:
      if top_indices is None:
        top_joined_fea = joined_fea
      else:
        top_joined_fea = util.batched_gather(top_indices,
                                              joined_fea)
      top_joined_fea = top_joined_fea[:, :, tf.newaxis,
                                      tf.newaxis]
    else:
      top_joined_fea = top_cobj.get('fea')
      if self._post_match_convline is not None:
          top_joined_fea = self._post_match_convline.build(
              top_joined_fea,
              scope='joining_convline')

    top_cobj.set('fea', top_joined_fea)
    sampled_multiclass_energy = None
    if top_cobj.has_key('multiclass_energy'):
      sampled_multiclass_energy = top_cobj.pop('multiclass_energy')

    return top_cobj, top_cobj.pop('energy'), loss, sampled_multiclass_energy
