from util import *
import tensorflow as tf
import numpy as np
from IPython import embed

def test_batched_gather():
  a1_np = np.random.normal(size=(4,5))
  a2_np = np.random.normal(size=(4,5))
  inds_np = np.random.randint(5, size=(4,3))

  # gather using numpy
  ret_nps = [np.zeros(inds_np.shape) for _ in range(2)]
  for a_np, ret_np in zip([a1_np, a2_np], ret_nps):
    for i in range(a_np.shape[0]):
      ret_np[i] = a_np[i, inds_np[i]]

  # tf gather
  a1 = tf.constant(a1_np)
  a2 = tf.constant(a2_np)
  inds = tf.constant(inds_np, dtype=tf.int32)
  ret = batched_gather(inds, a1, a2)

  # check equality
  tensors, expected_vals = ret, ret_nps

  with tf.Session() as sess:
    vals = sess.run(tensors)
    check_equal(vals, expected_vals)
  return True

def test_subsample():
  k = np.random.randint(1, 5)
  indicators = tf.constant([True, True, True, False], dtype=tf.bool)
  labels = tf.constant([True, False, True, False], dtype=tf.bool)
  scores = tf.constant([0.0, 1.0, 2.0, 3.0])
  balance_fraction = .5

  inds = subsample(indicators, labels, scores, k, balance_fraction)

  with tf.Session() as sess:
    vals = sess.run(inds)
    from IPython import embed;embed()

def check_equal(vals, expected_vals):
  for val, expected_val in zip(vals, expected_vals):
    assert(np.all(val == expected_val))

if __name__ == '__main__':
  #test_batched_gather()
  test_subsample()
