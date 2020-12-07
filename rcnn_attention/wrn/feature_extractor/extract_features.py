import functools
import tensorflow as tf
import numpy as np
import os, sys
from functools import partial
from rcnn_attention.wrn.model.wide_resnet import wide_resnet_model
from dl_papers.datasets.miniimagenet import load_miniimagenet_data

trained_model_path = '../logs/mini_base/train/model.ckpt-200000'

def extract(split='train', wrn_depth=28, wrn_width=10, is_training=False, wrn_dropout_rate=0.0):
  with tf.Graph().as_default():
    model = partial(
        wide_resnet_model,
        depth=wrn_depth,
        width_factor=wrn_width,
        training=is_training,
        dropout_rate=wrn_dropout_rate,
        data_format='channels_last'
        )
    batch = 10
    input_images = tf.placeholder(tf.float32, shape=(batch,84,84,3)) #TODO complete
    with tf.variable_scope('FeatureExtractor'):
      features, activations = model(input_images)
    pooled_features = tf.reduce_mean(activations['post_relu'],[1,2], keep_dims=True, name='AvgPool')
    configproto = tf.ConfigProto()
    configproto.gpu_options.allow_growth = True
    variables_to_restore = tf.global_variables()
    var_map = {var.op.name: var for var in variables_to_restore}
    saver = tf.train.Saver(var_map)
    def init_fn(sess):
      saver.restore(sess, trained_model_path)
    with tf.Session(config=configproto) as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      init_fn(sess)
      data = load_miniimagenet_data(split=split, train_num=600)
      x_train, x_test, y_train, y_test, x_trean_mean, xtrain_std, files_train, files_test = data
      num_iter = x_train.shape[0]//batch
      for i in range(num_iter):
        if i+1 % 100 == 0:
          print('processed batch {0}/{1}'.format((i+1)/100,num_iter))
        np_pooled_features = sess.run(pooled_features, feed_dict={input_images:x_train[i*batch:(i+1)*batch]})
        names = files_train[i*batch:(i+1)*batch]
        fea_names = [name[:-4] for name in names]
        for fname,fea in zip(fea_names, np_pooled_features):
          idx = fname.rfind(split)
          save_name = fname[:idx]+'features/'+fname[idx:]
          root = save_name[:save_name.rfind('/')]
          if not os.path.exists(root):
            os.makedirs(root)
          np.save(save_name,fea)
        #from IPython import embed;embed()

if __name__ == '__main__':
  print('Extracting features of val')
  extract(split='val')
  print('Extracting features of test')
  extract(split='test')
  print('Extracting features of train')
  extract(split='train')
