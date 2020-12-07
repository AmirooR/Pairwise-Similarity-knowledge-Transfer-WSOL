import tensorflow as tf
import numpy as np
import os, sys
from IPython import embed

def load_data(prefix):
  matches = np.load(prefix + '_matches.npy')
  unmatches = np.load(prefix + '_unmatches.npy')
  Np = matches.shape[0]
  Nn = unmatches.shape[0]
  yp = (Np + 1.0)/(Np + 2.0)
  yn = 1.0/(Nn+2.0)
  f = lambda x: np.log(x/(1.0-x))
  X = np.concatenate((matches, unmatches))
  X = f(X)
  Y = np.concatenate((np.ones_like(matches)*yp, np.ones_like(unmatches)*yn))
  return X,Y


def main(prefix):
  X_train,Y_train = load_data(prefix)
  num_samples = X_train.shape[0]
  indices = np.arange(num_samples)
  x = tf.placeholder(dtype=tf.float32, shape=[num_samples])
  y = tf.placeholder(dtype=tf.float32, shape=[num_samples])
  A = tf.get_variable('A',[1], initializer=tf.constant_initializer(1.0))
  B = tf.get_variable('B',[1], initializer=tf.constant_initializer(0.0))
  p_logit = A*x + B
  #p = 1.0/(1.0 + tf.exp(A*x+B))
  #loss = -tf.reduce_mean(y*tf.log(p) + (1-y)*tf.log(1-p))
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=p_logit))
  train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a, b = None, None
    for i in range(5000):
      np.random.shuffle(indices)
      _, loss_value,a,b = sess.run([train_op, loss,A,B], feed_dict={x:X_train[indices], y:Y_train[indices]})
      if (i+1) % 100 == 0:
        print('[{}] Loss: {}, -- a: {}, b: {}'.format(i+1, loss_value, a,b))

    np.save('platt_{}.npy'.format(prefix), [a,b])


if __name__ == '__main__':
  if len(sys.argv) != 2:
    print('Usage: {} prefix'.format(sys.argv[0]))
    sys.exit(1)
  prefix = sys.argv[1]
  main(prefix)
