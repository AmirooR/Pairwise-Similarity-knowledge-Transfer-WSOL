from tensorpack import *
import tensorflow as tf
slim = tf.contrib.slim

class DummyDataFlow(dataflow.RNGDataFlow):
  def __init__(self, size=100):
    self._data = range(100)

  def get_data(self):
    for i in range(self.size()):
      yield [float(self._data[i])]

  def size(self):
    return len(self._data)

def _create_input_queue(*args):
  ds = DummyDataFlow()
  ds = PrefetchDataZMQ(ds, 2)
  ds.reset_state()

  phs = [tf.placeholder(tf.float32, shape=())]
  queue = tf.FIFOQueue(10, [tf.float32])
  thread = graph_builder.EnqueueThread(queue, ds, phs)
  return queue, thread

def _create_losses(queue, **kwargs):
  dt0 = queue.dequeue()
  dt1 = queue.dequeue()

  dt0 = tf.Print(dt0, [dt0, dt1])
  name = 'SecondStageFeatureExtractor/resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights'
  shape=(1, 1, 512, 2048)
  weights = slim.variable(name,
                          shape=shape,
                          initializer=tf.truncated_normal_initializer(stddev=0.1),
                          regularizer=slim.l2_regularizer(0.000000001))
  tf.losses.add_loss(dt0)

if __name__ == '__main__':
  queue, thread = _create_input_queue()
  dt0 = queue.dequeue()
  dt1 = queue.dequeue()

  sess = tf.Session()

  with sess.as_default():
    thread.start()
    for i in range(10):
      res = sess.run([dt0, dt1])
      print(res)
