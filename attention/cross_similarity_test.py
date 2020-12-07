from cross_similarity import *
import tensorflow as tf
from IPython import embed

if __name__ == '__main__':
  N, M = 10, 5
  cross_sim = CosineCrossSimilarity()
  fea0 = tf.random_normal((N, M, 1, 1, 6))
  fea1 = tf.random_normal((N, M, 1, 1, 6))

  sim_matrix = cross_sim._build(tf.squeeze(fea0), tf.squeeze(fea1),
                                reuse_vars=True, scope='Test_Cross_Sim__buid')

  match0 = tf.random_uniform((N, M)) < .5
  match1 = tf.random_uniform((N, M)) < .5

  scores, matches, pairs = cross_sim.build(fea0, fea1, match0, match1,
                        reuse_vars=False, scope='Test_Corss_Sim_build')
  with tf.Session() as sess:
    (sim_matrix_np, scores_np,
        matches_np, pairs_np,
        match0_np, match1_np) = sess.run([sim_matrix, scores,
                                          matches, pairs, match0,
                                          match1])
    embed()
