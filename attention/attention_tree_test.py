import tensorflow as tf
from attention_tree import *
from cross_similarity import CosineCrossSimilarity
import numpy as np
from IPython import embed

def test_common_objects():
  def random_cobjs(n, m, j):
    boxes = np.random.randint(100, size=(n,m,j,4))
    fea = np.random.normal(size=(n, m, 1, 1, 10))
    scores = np.random.uniform(size=(n,m, 1))
    match = np.random.uniform(size=(n,m)) < .5
    return (CommonObjects(boxes=boxes,
                          fea=fea,
                          match=match,
                          scores=scores),
            CommonObjects(boxes=tf.constant(boxes),
                          fea=tf.constant(fea),
                          match=tf.constant(match),
                          scores=tf.constant(scores)))
  n = 4
  cobjs0_np, cobjs0 = random_cobjs(n, 2, 1)

  # Test split
  cobjs0_0, cobjs0_1 = cobjs0.split()

  # Test join
  cobjs1 = cobjs0_0.copy()
  cobjs1.join(cobjs0_1)

  #cobjs2 = cobjs1.gather
  with tf.Session() as sess:
    cobjs1_np = CommonObjects(**sess.run(cobjs1.fields))
    embed()

def test_attention_tree(with_tree_loss=True,
                        with_positive_balance=True):
  meta_batch_size = 3

  if with_positive_balance:
     positive_balance_fraction = .5
  else:
    positive_balance_fraction = None

  k = 4
  nproposals = 10

  if with_tree_loss:
    depth_loss_multiplier = 1.0
  else:
    depth_loss_multiplier = None

  pre_match_convline_fn = None
  post_match_convline_fn = None

  is_training = with_positive_balance or with_positive_balance
  tree = AttentionTree([4,2],
                       pre_match_convline_fn,
                       post_match_convline_fn,
                       CosineCrossSimilarity,
                       positive_balance_fraction, k,
                       use_orig_fea_in_post_match_convline=True,
                       depth_loss_multiplier=depth_loss_multiplier,
                       is_training=is_training)

  features = tf.random_normal((meta_batch_size*k, nproposals, 1, 1, 8))
  scores = tf.random_uniform((meta_batch_size*k, nproposals))
  proposals = tf.random_uniform((meta_batch_size*k, nproposals, 4))

  if is_training:
    match = tf.constant(np.random.uniform(size=(meta_batch_size*k, nproposals)) < .5,
        dtype=tf.bool)
  else:
    match = None

  (top_scores, top_feas) = tree.build(features, match)

  with tf.Session() as sess:
    gathered_proposals, num_proposals = tree.gather_proposals(proposals)
    tensors = [top_scores, top_feas, gathered_proposals, num_proposals]
    vals_np = sess.run(tensors)
    embed()

if __name__ == '__main__':
  #test_common_objects()
  test_attention_tree(with_tree_loss=True, with_positive_balance=True)
