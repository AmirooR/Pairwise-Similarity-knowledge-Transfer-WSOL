import tensorflow as tf
from rcnn_attention.attention import util

def _flatt_bach(t):
  t = tf.convert_to_tensor(t)
  dim = t.shape[-1].value
  if dim is None:
    dim = tf.shape(t)[-1]
  return tf.reshape(t, [-1, dim])

def approx_kl_divergence(p, logits,
                         partitions,
                         partitions_dist,
                         scope=None,
                         partial_loss=True,
                         partitions_dist_scale=1.0,
                         skip_normalization=False):
  '''
    p: tensor with shape [..., N] in which elements are samples from
       (unormalized) target distribution
    logits: tensor with the same type and shape as p in which
              elements are samples from predicted logits
    partitions: integer tensor in which each element shows the index
              of the partition that each correponding element of p
              and q_logits are comming from. We assume all the
              element in one partition has the same value.
              max(partitions) < M
    partitions_dist: A tensor with shape [M] that shows
                   relative size of each different M partitions
  '''
  with tf.variable_scope(scope, 'approx_kl_divergence',
                         values=[p, logits, partitions, partitions]):

    util.add_extra_tensor('logits', logits)
    util.add_extra_tensor('partitions', partitions)
    util.add_extra_tensor('partitions_dist', partitions_dist)

    partitions_dist = tf.convert_to_tensor(partitions_dist)
    p, logits, partitions = [_flatt_bach(t) for t in [p, logits, partitions]]
    m = partitions_dist.shape[0].value

    ## Count the number of elements in each partition
    count = tf.map_fn(lambda arr: tf.bincount(arr, minlength=m, maxlength=m),
                      partitions)
    ## count shape = [B, M]
    count.set_shape([partitions.shape[0].value, m])

    ## Adjust the weights based on the counts
    ## inf values wont be showing up in weights...
    partitions_dist2 = tf.truediv(partitions_dist[tf.newaxis],
                                  tf.cast(count, tf.float32))
    weights = util.batched_gather(partitions, partitions_dist2)

    if skip_normalization:
      weights = tf.ones_like(weights)
      partitions_dist_scale = 1.0
    ## See tf.reduce_logsumexp implementation
    raw_max = tf.reduce_max(logits, axis=-1)
    my_max = tf.stop_gradient(
                tf.where(tf.is_finite(raw_max),
                         raw_max,
                         tf.zeros_like(raw_max)))

    logits = logits - my_max[..., tf.newaxis]
    q_normalizer = tf.reduce_sum(weights * tf.exp(logits),
                                 keep_dims=True,
                                 axis=-1)

    p_normalizer = tf.reduce_sum(weights * p,
                                 keep_dims=True,
                                 axis=-1)

    p = tf.truediv(p, p_normalizer)

    if partial_loss:
      loss_scale = partitions_dist_scale
    else:
      loss_scale = 1.0
      p = p*weights

    return -loss_scale * p * (logits - tf.log(q_normalizer) + tf.log(partitions_dist_scale))
