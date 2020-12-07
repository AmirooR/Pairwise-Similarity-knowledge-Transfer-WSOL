from approx_kl_divergence import *
import numpy as np

def get_full_values(arr_batch, partitions_batch, partitions_dist):
  full_len = sum(partitions_dist)
  assert(full_len.is_integer())
  full_len = int(full_len)
  n_partitions = len(partitions_dist)

  full_arr_batch = []
  for r in range(len(arr_batch)):
    arr = arr_batch[r]
    partitions = partitions_batch[r]
    count = np.bincount(partitions, minlength=n_partitions)
    full_arr = []
    for i in range(len(arr)):
      elem = arr[i]
      nrep = partitions_dist[partitions[i]]/count[partitions[i]]
      assert(nrep.is_integer())
      nrep = int(nrep)
      full_arr.extend([elem]*nrep)
    #full_arr += [0] * (full_len - len(full_arr))
    full_arr_batch.append(full_arr)

  return full_arr_batch

if __name__ == '__main__':
  batch_size = 10
  n_partitions = 3
  n_samples = 4
  partitions_dist_scale = 1e-6
  #np.random.seed(1357)

  ## Create valid p, logits, partitions, partitions_dist
  p = np.random.uniform(size=(batch_size, n_samples)).astype(np.float32)
  logits = np.random.normal(scale=10, size=(batch_size, n_samples)).astype(np.float32)

  partitions = np.random.randint(0, high=n_partitions, size=(batch_size, n_samples), dtype=np.int32)

  counts = map(lambda arr: np.bincount(arr, minlength=n_partitions), partitions)
  counts = np.array(counts, dtype=np.float32)
  counts[counts < 1] = 1.0

  partitions_dist = np.array([np.prod(np.unique(x)) for x in counts.T])

  ## Use the helper function to compute the gradient
  full_p = get_full_values(p, partitions, partitions_dist)
  full_p = [x/sum(x) for x in full_p]

  full_logits = get_full_values(logits, partitions, partitions_dist)

  ## Compute the full loss
  full_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=pp, logits=ll)
                                      for pp,ll in zip(full_p, full_logits)]


  full_loss = tf.stack(full_loss)

  ### Compute the approx loss
  approx_loss = approx_kl_divergence(p, logits, partitions,
                                     partitions_dist*partitions_dist_scale,
                                     partial_loss=False,
                                     partitions_dist_scale=partitions_dist_scale)

  approx_loss = tf.reduce_sum(approx_loss, axis=-1)

  #### For testing the gradient
  x = tf.random_normal((batch_size, n_samples), stddev=10.0)
  y = approx_kl_divergence(p, x, partitions, partitions_dist*partitions_dist_scale,
                           partial_loss=True, partitions_dist_scale=partitions_dist_scale)
  y = tf.reduce_sum(y, axis=-1)
  with tf.Session() as sess:
    full_loss_val, approx_loss_val = sess.run([full_loss, approx_loss])
    print('FULL LOSS FORWARD ERROR:', np.linalg.norm(full_loss_val - approx_loss_val))

    diff_err = tf.test.compute_gradient_error(x, (batch_size, n_samples), y, (batch_size,))
    print('DIFF ERROR:', diff_err)
