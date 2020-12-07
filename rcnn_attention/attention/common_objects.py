import tensorflow as tf
import util
import numpy as np

class CommonObjects(object):
  ''' A common object among J images is represented by J box coordinates,
      a 3D feature vector, and few other optional tensors.
      CommonObjects keep a batch of M "common object"s.

      Args:
        fea: with size [N, M, h, w, d]. Keeps the feature representation
             for each of the common objects

        matched_class: size [N, M, num_classes]. optional integer
             tensor indicating number of objects from each class
             in the common object.

        boxes: optional. with size [N, M, J, 4]. Keeps the coordinates
               of M common objects each among J images.
        ids: optinoal. size [N, M, J]. keeps the an id for each
              individual bounding box
        scores: size [N, M, V, num_classes]. Keeps a list of V scores
                for each common object.
  '''
  def __init__(self, **all_fields):
    self.data = dict()
    self.update_fields(**all_fields)
    # Check the tensor shapes
    n0, m0 = self._check_shape('fea')
    for key in self.data:
      shape = self._check_shape(key)
      assert(shape[0] == n0), 'Tensors shape mismatches {}, {}'.format(shape[0], n0)
      if len(shape) > 1:
        assert(shape[1] == m0), 'Tensors shape mismatches'

  def update_fields(self, **new_fields):
    for key, value in new_fields.items():
      if value is None:
        if self.has_key(key):
          del self.data[key]
      else:
        self.data[key] = value

  def _check_shape(self, name):
    tensor = self.data[name]
    ranks = dict([('boxes', 4), ('fea', 5), ('matched_class', 3),
      ('is_target', 2), ('target_class', 3), ('energy', 2), ('bag_target_class', 1)])
    if ranks.has_key(name):
      assert(len(tensor.shape) == ranks[name]), 'Tensor rank is not correct'
    else:
      assert(len(tensor.shape) > 2), 'Tensor rank is less than 3'

    rank = len(tensor.shape)

    shape = tensor.shape
    if isinstance(tensor, tf.Tensor):
      shape = shape.as_list()

    return shape[:min(2, rank)]

  @property
  def fields(self):
    return self.data

  @property
  def values(self):
    return self.data.values()

  def has_key(self, key):
    return self.data.has_key(key)

  def pop(self, key):
    value = self.get(key)
    self.set(key, None)
    return value

  def get(self, key):
    return self.data[key]

  def set(self, key, value):
    self.update_fields(**dict(([key, value],)))
    if value is not None:
      shape0 = self._check_shape('fea')
      shape = self._check_shape(key)
      assert(shape[0] == shape0[0]), 'Tensors shape mismatches'
      if len(shape) > 1:
        assert(shape[1] == shape0[1]), 'Tensors shape mismatches'



  def gather(self, indices):
    ''' Create a new CommonObjects instance by gathering the indices
        in the indices tensor.

        for each field in common object:
          for all k,i:
            field[k, i] = field[k, indices[k, i]]

        Args:
          indices: An int tensor with size [N, L]. Indices
                   of the selected objects.
        Returns:
           A new CommonObjects instance by gathering the given
                indices.
    '''
    with tf.variable_scope('gather', values=self.values):
      data = dict(self.data)
      # gather has no effect on bag_target_class
      bag_target_class = data.pop('bag_target_class', None)
      fields, values = zip(*data.items())
      batched_data = dict(zip(fields, util.batched_gather(indices, *values)))
      if bag_target_class is not None:
        batched_data['bag_target_class'] = bag_target_class
      return CommonObjects(**batched_data)

  def join(self, cobj, scores=None, joined_fea=None):
    already_handled  = ['boxes', 'matched_class',
                        'is_target', 'target_class',
                        'scores', 'bag_target_class']
    with tf.variable_scope('join', values=self.values):
      if self.has_key('boxes'):
        # Join boxes along second axis
        self.set('boxes', tf.concat([self.get('boxes'), cobj.get('boxes')], 2))

      if joined_fea is not None:
        self.set('fea', joined_fea)
        already_handled.append('fea')

      # Join match with logical and
      assert(self.has_key('matched_class') == cobj.has_key('matched_class'))
      if self.has_key('matched_class'):
        self.set('matched_class',
            self.get('matched_class') + cobj.get('matched_class'))
        self.set('is_target', tf.logical_and(self.get('is_target'),
                                             cobj.get('is_target')))
      if self.has_key('target_class'):
        assert(cobj.has_key('target_class'))
        eq_cond = tf.reduce_all(
                    tf.equal(self.get('target_class'),
                             cobj.get('target_class')))
        eq_assert = tf.Assert(eq_cond, ['target classes are not eq'])
        with tf.control_dependencies([eq_assert]):
          target_class = tf.identity(self.get('target_class'))
        self.set('target_class', target_class)

      # Join the new scores with others
      scores_list = []
      if self.has_key('scores'):
        scores_list.append(self.get('scores'))
      if cobj.has_key('scores'):
        scores_list.append(self.get('scores'))
      if scores is not None:
        assert(len(scores.shape) == 3), 'scores is not a rank three tensor'
        scores_list.append(scores)

      if scores_list:
        scores = tf.concat(scores_list, -1) if len(
            scores_list) > 1 else scores_list[0]
        self.set('scores', scores)

      if self.has_key('bag_target_class'):
        assert(cobj.has_key('bag_target_class'))
        eq_cond = tf.reduce_all(
                    tf.equal(self.get('bag_target_class'),
                             cobj.get('bag_target_class')))
        eq_assert = tf.Assert(eq_cond, ['target bag classes are not eq'])
        with tf.control_dependencies([eq_assert]):
          bag_target_class = tf.identity(self.get('bag_target_class'))
        self.set('bag_target_class', bag_target_class)

      # Join everything else along their last dimensions
      for key, value in self.data.items():
        if key in already_handled:
          continue
        self.set(key, tf.concat([self.get(key), cobj.get(key)], axis=-1))

  def split(self):
    fields, values = zip(*self.data.items())
    split0_data = dict(zip(fields, [elem[0::2] for elem in values]))
    split1_data = dict(zip(fields, [elem[1::2] for elem in values]))
    return CommonObjects(**split0_data), CommonObjects(**split1_data)

  def copy(self):
    return CommonObjects(**self.data)
