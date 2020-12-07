import tensorflow as tf
import util
import numpy as np
import object_detection.utils.shape_utils as shape_utils
from common_objects import CommonObjects
from attention_unit import AttentionUnit
from rcnn_attention.attention.cross_similarity import PairwiseCrossSimilarity

class AttentionTree(object):
  def __init__(self, attention_units_fn, k_shot,
              num_negative_bags=0,
              is_training=False,
              stop_features_gradient=False,
              preprocess_convline=None,
              num_classes=1,
              rescore_instances=False,
              rescore_min_match_frac=0.0,
              rescore_convline=None,
              rescore_fc_hyperparams=None,
              negative_preprocess_convline=None,
              negative_postprocess_convline=None,
              training_subsampler_ncobj=None,
              training_subsampler_pos_balance_frac=None,
              training_subsampler_agnostic=False,
              fea_split_ind=None):

    self._k_shot = k_shot
    depth = np.log2(k_shot)
    assert(depth.is_integer()), 'k_shot is not power of 2'
    assert(len(attention_units_fn) <= depth + 1
        ), 'units lenght is more than depth'

    self._depth = int(depth)

    def _pad_to_depth(l):
      if not isinstance(l, list):
        l = [l]
      if len(l) == 0:
        l = [None]
      return l + [l[-1]] * (1 + self._depth - len(l))

    self._attention_units_fn = _pad_to_depth(attention_units_fn)
    self._num_negative_bags = num_negative_bags
    self._is_training = is_training
    self._stop_features_gradient = stop_features_gradient
    self._preprocess_convline = preprocess_convline
    self._num_classes = num_classes
    self._rescore_instances = rescore_instances
    self._rescore_convline  = rescore_convline
    self._rescore_min_match_frac = rescore_min_match_frac
    self._rescore_fc_hyperparams = rescore_fc_hyperparams
    self._negative_preprocess_convline = negative_preprocess_convline
    self._negative_postprocess_convline = negative_postprocess_convline
    self._training_subsampler_ncobj = training_subsampler_ncobj
    self._training_subsampler_pos_balance_frac = training_subsampler_pos_balance_frac
    # if True every non-background box is considered as positive sample in the sampler
    # otherwise only common object are considered as positive
    self._training_subsampler_agnostic = training_subsampler_agnostic
    util.reset_static_values()
    self._fea_split_ind = fea_split_ind

  def _create_negative_objects(self, neg_fea, neg_matched_class=None, do_avg=False):
    '''
      pools negative features and their corresponding classes.
      Args:
        neg_fea: a [MBS*num_negative_bags, B, 1, 1, d] float tensor.
        neg_matched_class: optional [MBS*num_negative_bags, B, num_classes + 1] tensor
          representing the negative bag classes.
      returns:
        neg_fea: [MBS, 1, 1, 1, d] tensor if do_avg=True, and
                 [MBS, B*num_negative_bags, 1, 1, d] o.w
    '''
    if neg_matched_class is None:
      neg_target_cls = None
    else:
      #[MBS*num_negative_bags, B, num_classes + 1]
      neg_cls_shape = shape_utils.combined_static_and_dynamic_shape(neg_matched_class)
      #[MBS, num_negative_bags, B, num_classes + 1]
      neg_target_cls = tf.reshape(neg_matched_class,
          [neg_cls_shape[0]/self._num_negative_bags,
            self._num_negative_bags] + neg_cls_shape[1:])

      # or over all of the negative items
      # [MBS, num_classes+1]
      neg_target_cls = tf.reduce_sum(neg_target_cls, axis=[1,2])
      neg_target_cls = tf.cast(neg_target_cls, tf.bool)

    if self._negative_preprocess_convline is not None:
      neg_fea = self._negative_preprocess_convline.build(neg_fea, scope='NegPreprocess')
    #[MBS*num_negative_bags, B, 1,1, d]
    neg_fea_shape = shape_utils.combined_static_and_dynamic_shape(neg_fea)
    #[MBS,num_negative_bags*B, 1,1, d]
    neg_fea = tf.reshape(neg_fea, [neg_fea_shape[0]/self._num_negative_bags,
                                   -1] + neg_fea_shape[2:])
    #[MBS, 1, 1, 1, d]
    #may be we should define an op or sth else here rather than mean
    if do_avg:
      neg_fea = tf.reduce_mean(neg_fea, [1], keep_dims=True, name='NegAvg')
    if self._negative_postprocess_convline is not None:
      neg_fea = self._negative_postprocess_convline.build(neg_fea, scope='NegPostprocess')

    return neg_fea, neg_target_cls

  def _create_solo_common_objects(self, fea, matched_class,
                                  parallel_iterations,
                                  neg_target_cls,
                                  fea_boxes,
                                  problem_target_class,
                                  objectness_score):
    '''
      Create one CommonObjects for each box in the boxes tensor
    '''
    fea_shape = shape_utils.combined_static_and_dynamic_shape(fea)

    # Average Pooling [meta_batch_size*k , B, h, w, d]==>
    #[meta_batch_size*k, B, 1, 1, d]
    #fea = tf.reduce_mean(fea, [2, 3], keep_dims=True, name='AvgPool')

    # Keep indecies of proposals inside their images
    ind = tf.tile(tf.range(fea_shape[1])[tf.newaxis], [fea_shape[0], 1])

    # reshape to [meta_batch_size*k, B, 1]
    ind = ind[..., tf.newaxis]

    ## Used only for cobj nms in evaluation
    if not self._is_training and fea_boxes is not None:
        fea_boxes = fea_boxes[..., np.newaxis, :]
    else:
        fea_boxes = None

    if matched_class is None:
      is_target, target_cls = None, None
    else:
      # Compute is_target
      # matched_class shape [meta_batch_size*k, B, num_classes+1]
      cls_shape = shape_utils.combined_static_and_dynamic_shape(
                                                  matched_class)
      # shape [meta_batch_size, k, B, num_classes+1]
      target_cls = tf.reshape(matched_class, [cls_shape[0]/self._k_shot,
                                          self._k_shot] + cls_shape[1:])
      # or over items in each bag
      target_cls = tf.reduce_sum(target_cls, axis=2)
      # and over k_shot
      target_cls = tf.reduce_prod(target_cls, axis=1)

      # shape [meta_batch_size, num_classes+1]
      target_cls = tf.cast(target_cls, tf.bool)
      if neg_target_cls is not None:
        target_cls = tf.logical_and(target_cls,
                 tf.logical_not(neg_target_cls))
      # In case there are more than one target_cls sample one
      target_ind = tf.map_fn(lambda x: util.subsample(x, 1),
                             target_cls[..., 1:],
                             dtype=tf.int64,
                             back_prop=True,
                             name='select_target_cls')[..., 0] + 1
      util.add_extra_tensor('target_ind', target_ind)
      ntarget_cls = tf.cast(tf.one_hot(target_ind, cls_shape[-1]), tf.bool)
      util.add_extra_tensor('ntarget_cls', ntarget_cls)
      # Handle cases in which there is no target cls
      target_cls = tf.logical_and(ntarget_cls, target_cls)


      target_cls = tf.tile(target_cls[:, tf.newaxis, tf.newaxis],
                              [1, self._k_shot, cls_shape[1], 1])
      target_cls = tf.reshape(target_cls, cls_shape)

      is_target = tf.logical_and(target_cls,
                  tf.cast(matched_class, tf.bool))

      is_target = tf.reduce_any(is_target[..., 1:], axis=-1)
      util.add_extra_tensor('target_cls', target_cls)
      util.add_extra_tensor('is_target', is_target)

    bag_target_class = None
    if problem_target_class is not None:
      #NOTE: changed from the comment for trws where we have MBS > 1.
      #      Not sure what happens in the case of negative bags
      assert(self._num_negative_bags == 0), "Ridam passe kallat Shaban"
      #bag_target_class = tf.tile(problem_target_class[..., tf.newaxis],
      #                           [1, self._k_shot])
      bag_target_class = tf.tile(problem_target_class[..., tf.newaxis],
                                 [1, target_cls.shape[0]])

      bag_target_class = tf.reshape(bag_target_class, [-1])

    if objectness_score is not None:
      objectness_score = objectness_score[..., tf.newaxis]

    return CommonObjects(fea=fea,
                         matched_class=matched_class,
                         is_target=is_target,
                         ind=ind,
                         target_class=target_cls,
                         boxes=fea_boxes,
                         bag_target_class=bag_target_class,
                         k0_energy=objectness_score)

  def _subsample(self, cobj):
      if not self._is_training or not self._training_subsampler_ncobj:
          return cobj
      if self._training_subsampler_agnostic: # every obj is target
        # [batch_size*k, bag_size, num_classes+1]
        mc = cobj.get('matched_class')
        is_target = tf.reduce_any(tf.cast(mc[..., 1:], tf.bool),axis=2)
      else:
        # [batch_size*k, bag_size]
        is_target = cobj.get('is_target')

      def fn_target(labels):
        positive_fraction = 1.0
        indicators = labels # only samples from positive examples`
        return util.balanced_subsample(indicators,
                                       labels,
                                       self._training_subsampler_ncobj,
                                       positive_fraction)

      def fn_non_target(labels):
        positive_fraction=self._training_subsampler_pos_balance_frac
        indicators = tf.ones_like(labels, tf.bool)
        if positive_fraction is None:
          positive_fraction = 1.0
          labels = indicators
        return util.balanced_subsample(indicators,
                                       labels,
                                       self._training_subsampler_ncobj,
                                       positive_fraction)

      def fn(x):
        labels, is_bag_target = x
        return tf.cond(is_bag_target, lambda: fn_target(labels),
                       lambda: fn_non_target(labels))

      if cobj.has_key('bag_target_class'):
        bag_target_class = tf.cast(cobj.get('bag_target_class'), tf.int32)
        mc = cobj.get('matched_class')
        bag_mc = tf.reduce_sum(mc, axis=1)

        is_target_bag = util.batched_gather(bag_target_class[:, tf.newaxis],
                                            bag_mc)[:, 0]
        is_target_bag = tf.greater(is_target_bag, 0.0)
        util.add_extra_tensor('sub_bag_target_class', bag_target_class)
      else:
        n = shape_utils.combined_static_and_dynamic_shape(is_target)[0]
        is_target_bag = tf.zeros(n, tf.bool)

      inds = tf.map_fn(fn, (is_target, is_target_bag),
                       dtype=tf.int64,
                       back_prop=False,
                       name='batched_subsample')
      sub_cobj = cobj.gather(inds)

      util.add_extra_tensor('sub_is_target', is_target)
      util.add_extra_tensor('sub_is_target_bag', is_target_bag)
      util.add_extra_tensor('sub_inds', inds)
      util.add_extra_tensor('sub_matched_class', sub_cobj.get('matched_class'))
      util.add_extra_tensor('sub_orig_matched_class', cobj.get('matched_class'))

      return sub_cobj

  def apply_preprocess_convline(self, fea, neg_fea):
    def _split(fea):
      if fea is None:
        return None, None
      return fea[..., :self._fea_split_ind], fea[..., self._fea_split_ind:]

    def _single_fea_forward(fea, neg_fea):
      ## preprocess convline is applied on both fea and neg_fea
      if self._preprocess_convline is not None:
        x = fea if neg_fea is None else tf.concat([fea, neg_fea], 0)
        x = self._preprocess_convline.build(x, scope='Preprocess')
        if neg_fea is None:
          fea = x
        else:
          ind = shape_utils.combined_static_and_dynamic_shape(fea)[0]
          fea = x[:ind]
          neg_fea = x[ind:]
      return fea, neg_fea

    if self._fea_split_ind is None:
      return _single_fea_forward(fea, neg_fea)

    fea = _split(fea)
    neg_fea = _split(neg_fea)

    out_fea = []
    neg_out_fea = []
    if fea[0].shape[-1] > 0:
      with tf.variable_scope('A'):
        f, n = _single_fea_forward(fea[0], neg_fea[0])
        out_fea.append(f)
        if n is not None:
          neg_out_fea.append(n)

    if fea[1].shape[-1] > 0:
      with tf.variable_scope('B'):
        f, n = _single_fea_forward(fea[1], neg_fea[1])
        out_fea.append(f)
        if n is not None:
          neg_out_fea.append(n)

    if len(neg_out_fea) == 0:
      neg_out_fea = None
    else:
      neg_out_fea = tf.concat(neg_out_fea, axis=-1)

    return tf.concat(out_fea, axis=-1), neg_out_fea

  def build(self, fea, matched_class, parallel_iterations,
      neg_fea=None, neg_matched_class=None, fea_boxes=None,
      scope=None, problem_target_class=None, objectness_score=None):
    '''
      Args:
        fea: A tensor with size [meta_batch_size*k_shot, B, h, w, d]
        matched_class: An optional integer tensor with size
                       [meta_batch_size*k, B, num_classes+1]
        objectness_score: [meta_batch_size*k, B]
      Returns:
        top_scores: A tensor with size [meta_batch_size, self.ncobj_proposals]
        top_features: A tensor with size [meta_batch_size, self.ncobj_proposals,
                      h, w, d]
    '''
    util.add_extra_tensor('matched_class', matched_class)
    with tf.variable_scope(scope, 'attention_tree',
        values=[fea, matched_class]):
      if self._stop_features_gradient:
        fea = tf.stop_gradient(fea)
      self._tree_losses = dict()
      self._attention_unit_outputs = []

      def _add_result(res, height):
        self._attention_unit_outputs.append(res)
        self._tree_losses[height] = res.loss

      fea, neg_fea = self.apply_preprocess_convline(fea, neg_fea)

      # negative bags
      neg_pooled_fea, neg_target_cls = None, None
      do_avg = False #TODO: move to config, add assert for negative attention
      if neg_fea is not None:
        assert self._num_negative_bags > 0, \
          'negative feature is not None but '+ \
            'num_negative_bags({}) is not positive'.format(self._num_negative_bags)
        neg_pooled_fea, neg_target_cls = self._create_negative_objects(neg_fea,
                                              neg_matched_class, do_avg=do_avg)

      solo_cobj = self._create_solo_common_objects(fea, matched_class,
                                                   parallel_iterations,
                                                   neg_target_cls,
                                                   fea_boxes,
                                                   problem_target_class,
                                                   objectness_score)

      self._solo_cobj = self._subsample(solo_cobj)
      # Compute K1 scores and loss 
      attention_unit = self._attention_units_fn[0](1, self, parallel_iterations)
      res = attention_unit.build(self._solo_cobj, None,
                                 neg_pooled_fea,
                                 neg_target_cls,
                                 neg_matched_class,
                                 scope='Unit0')
      if do_avg is False:
        neg_pooled_fea = None

      _add_result(res, 0)
      cobj = self._solo_cobj
      for height in range(1, 1 + self._depth):
        cobj0, cobj1 = res.cobj.split()
        attention_unit = self._attention_units_fn[height](2**height, self,
                                                      parallel_iterations)
        res = attention_unit.build(cobj0, cobj1,
                                   scope='Unit{}'.format(height))
        _add_result(res, height)
      return res

  def loss(self):
    return self._tree_losses

  def tree_scores(self):
    '''
      For the top_n matches, returns an array of all the matching scores
      in the tree. It could be used to define addination auxiliary losses.
      Returns:
        top_scores: A tensor with size [meta_batch_size, top_n, k_shot-1]
    '''
    return self.cobj().get('scores')

  def debug_tensors(self, boxes, nms_fn, image_shape=None,
                    include_feas=False,
                    include_k2_cross_similarty=False,
                    apply_sigmoid_to_scores=False): #start bezan halle
    tensors_dict = dict()
    def _add_results(fg_scores=None, tag=None, class_agnostic=True, **kwargs):
      kwargs = {key:val for key, val in kwargs.items() if val is not None}
      results_dict= {'scores':fg_scores,
                     'class_agnostic':tf.constant(class_agnostic)}

      results_dict.update(kwargs)
      tensors_dict[tag] = results_dict

    for i, res in enumerate(self._attention_unit_outputs):
      (cobj_fea,
       cobj_fg_scores, cobj_cls_scores, cobj_match,
       cobj_boxes) = res.format_output(boxes)

      cobj_energy = -cobj_fg_scores
      # Doest chagne the order but make sure scores > 0.
      if apply_sigmoid_to_scores:
        cobj_fg_scores = tf.nn.sigmoid(cobj_fg_scores)
      kwargs = dict()

      #kwargs['instance_scores'] = res.instance_scores
      #kwargs['instance_fg_scores'] = res.instance_fg_scores
      #kwargs['match'] = cobj_match
      kwargs['proposal_inds'] = res.proposal_inds

      kwargs['boxes'] = cobj_boxes
      if not include_feas:
        cobj_fea = None

      kwargs['feas'] = cobj_fea
      kwargs['classes'] = tf.zeros_like(cobj_fg_scores)
      ctag = 'Tree_K{}'.format(res.k)
      _add_results(fg_scores=cobj_fg_scores, tag=ctag,
                        class_agnostic=True, **kwargs)
      if cobj_cls_scores is not None:
        kwargs.pop('classes')
        cobj_multiclass_scores = tf.reduce_max(cobj_cls_scores[...,1:], axis=-1)
        kwargs['classes'] = tf.argmax(cobj_cls_scores[...,1:], axis=-1)
        _add_results(fg_scores=cobj_multiclass_scores, tag=ctag+'_multiclass',
                                               class_agnostic=False, **kwargs)
      if nms_fn is not None:
        (nmsed_boxes, nmsed_scores, nmsed_classes,
            nmsed_feas, num_detections) = nms_fn(
                cobj_boxes[:, :, tf.newaxis],
                cobj_fg_scores[..., tf.newaxis],
                masks=cobj_fea)

        _add_results(fg_scores=nmsed_scores,
                     class_agnostic=True,
                     boxes=nmsed_boxes,
                     classes=nmsed_classes,
                     feas=nmsed_feas[:, :, tf.newaxis] if include_feas else None,
                     tag=ctag+'_nmsed')
        if cobj_cls_scores is not None:
          cobj_cls_scores = tf.nn.softmax(cobj_cls_scores)
          num_classes = shape_utils.combined_static_and_dynamic_shape(cobj_cls_scores)[-1]
          cobj_cls_boxes = tf.tile(cobj_boxes[:,:,tf.newaxis], [1,1,num_classes-1,1])
          cobj_cls_fea = None
          if include_feas:
            cobj_cls_fea = tf.tile(cobj_fea, [1, 1, num_classes-1, 1, 1])

          (cls_nmsed_boxes, cls_nmsed_scores, cls_nmsed_classes,
            cls_nmsed_fea, cls_num_detections) = nms_fn(
                cobj_cls_boxes,
                cobj_cls_scores[...,1:],
                masks=cobj_cls_fea)
          _add_results(fg_scores=cls_nmsed_scores,
                     class_agnostic=False,
                     boxes=cls_nmsed_boxes,
                     classes=cls_nmsed_classes,
                     feas=cls_nmsed_fea[:, :, tf.newaxis] if include_feas else None,
                     tag=ctag+'_multiclass_nmsed')

    meta_info = {'energy': cobj_energy}
    if hasattr(PairwiseCrossSimilarity, 'n_unique_scores'):
        total_k2_scores = self._k_shot*(self._k_shot - 1)/2.0
        bag_size = shape_utils.combined_static_and_dynamic_shape(self._solo_cobj.get('fea'))[1]
        ratio = tf.to_float(PairwiseCrossSimilarity.n_unique_scores)/total_k2_scores/bag_size/bag_size
        meta_info['evaluated_k2_scores_ratio'] = [ratio]

    ks = [r.k for r in self._attention_unit_outputs]
    if include_k2_cross_similarty and 2 in ks:
      res_k2 = self._attention_unit_outputs[ks.index(2)]
      meta_info['k2_cross_similarity_scores'] = res_k2.cross_similarity_scores
      meta_info['k2_cross_similarity_pairs'] = res_k2.cross_similarity_pairs

    tensors_dict['meta_info'] = meta_info
    return tensors_dict
