import util

class AttentionUnitResult(object):
  def __init__(self, cobj, k, class_agnostic, scores=None,
      fg_scores=None, loss=None, attention_unit=None,
      cross_similarity_scores=None,
      cross_similarity_pairs=None):
    self._cobj = cobj
    self._scores = scores
    self._loss = loss
    self._attention_unit = attention_unit
    self._k = k
    self._class_agnostic = class_agnostic
    self._cross_similarity_pairs = cross_similarity_pairs
    self._cross_similarity_scores = cross_similarity_scores
    self._proposal_inds = None
    self._fg_scores = fg_scores
    self._instance_scores = None
    self._instance_fg_scores = None
    self._instance_loss = None

  @property
  def cross_similarity_scores(self):
    return self._cross_similarity_scores

  @property
  def cross_similarity_pairs(self):
    return self._cross_similarity_pairs

  @property
  def instance_loss(self):
    return self._instance_loss

  @property
  def instance_fg_scores(self):
    return self._instance_fg_scores

  @property
  def instance_scores(self):
    return self._instance_scores

  @property
  def class_agnostic(self):
    return self._class_agnostic

  @property
  def cobj(self):
    return self._cobj

  @property
  def scores(self):
    return self._scores

  @property
  def loss(self):
    return self._loss

  @property
  def unit(self):
    return self._attention_unit

  @property
  def k(self):
    return self._k

  @property
  def fg_scores(self):
    return self._fg_scores

  @property
  def proposal_inds(self):
    if self._proposal_inds is not None:
      return self._proposal_inds
    proposal_inds = self.cobj.get('ind')
    self._proposal_inds = util.convert_proposal_inds(proposal_inds)
    return self._proposal_inds

  def format_output(self, *rpn_props):
    '''
      Args:
        *args: list of tensors with size [meta_batch_size*k_shot, M, ...]
      Returns:
        gathered_args: list of tensors with size [meta_batch_size*k_shot,
        self.ncobj_proposals, ...]
    '''
    fea = util.tile_and_reshape_cobj_prop(self.cobj.get('fea'),
                            self.k)
    scores = None
    if self.cobj.has_key('matched_class'):
      matched_class = util.tile_and_reshape_cobj_prop(
          self.cobj.get('matched_class'), self.k)
    else:
      matched_class = None
    fg_scores = util.tile_and_reshape_cobj_prop(self.fg_scores, self.k)
    if self.scores is not None:
      scores = util.tile_and_reshape_cobj_prop(self.scores, self.k)
    rpn_props = util.batched_gather(self.proposal_inds, *rpn_props)
    rpn_props = rpn_props if isinstance(
                rpn_props, list) else [rpn_props]
    return [fea,
            fg_scores, scores,
            matched_class] + rpn_props

