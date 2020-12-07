# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""CoLocDetection model trainer.

This file provides a generic training method that can be used to train a
DetectionModel.
"""

import functools

import tensorflow as tf

from object_detection.builders import optimizer_builder
from object_detection.builders import preprocessor_builder
from object_detection.core import batcher
from object_detection.core import preprocessor
from rcnn_attention.coloc import standard_fields as fields
from object_detection.utils import ops as util_ops
from object_detection.utils import variables_helper
from deployment import model_deploy
from IPython import embed
import rcnn_attention.learning as learning

slim = tf.contrib.slim

def _create_input_queue(batch_size_per_clone, create_tensor_dict_fn,
                        batch_queue_capacity, num_batch_queue_threads,
                        prefetch_queue_capacity):
  """Sets up reader, prefetcher and returns input queue.

  Args:
    batch_size_per_clone: batch size to use per clone.
    create_tensor_dict_fn: function to create tensor dictionary.
    batch_queue_capacity: maximum number of elements to store within a queue.
    num_batch_queue_threads: number of threads to use for batching.
    prefetch_queue_capacity: maximum capacity of the queue used to prefetch
                             assembled batches.
    data_augmentation_options: a list of tuples, where each tuple contains a
      data augmentation function and a dictionary containing arguments and their
      values (see preprocessor.py).

  Returns:
    input queue: a batcher.BatchQueue object holding enqueued tensor_dicts
      (which hold images, boxes and targets).  To get a batch of tensor_dicts,
      call input_queue.Dequeue().
  """
  tensor_dict, enqueue_thread = create_tensor_dict_fn()

  #tensor_dict[fields.CoLocInputDataFields.supportset] = tf.expand_dims(
  #    tensor_dict[fields.CoLocInputDataFields.supportset], 0)

  images = tensor_dict[fields.CoLocInputDataFields.supportset]
  float_images = tf.to_float(images)
  tensor_dict[fields.CoLocInputDataFields.supportset] = float_images

  input_queue = batcher.BatchQueue(
      tensor_dict,
      batch_size=batch_size_per_clone,
      batch_queue_capacity=batch_queue_capacity,
      num_batch_queue_threads=num_batch_queue_threads,
      prefetch_queue_capacity=prefetch_queue_capacity)
  return input_queue, enqueue_thread


def _get_inputs(input_queue, num_classes, k_shot, bag_size):
  """Dequeue batch and construct inputs to object detection model.

  Args:
    input_queue: BatchQueue object holding enqueued tensor_dicts.
    num_classes: Number of classes.
    k_shot: k_shot

  Returns:
    images: a list of 3-D float tensor of images.
    locations_list: a list of tensors of shape [num_boxes, 4]
      containing the corners of the groundtruth boxes.
  """
  read_data_list = input_queue.dequeue()
  label_id_offset = 0
  def extract_images_and_targets(read_data):
    image = read_data[fields.CoLocInputDataFields.supportset]
    boxes = read_data[fields.CoLocInputDataFields.groundtruth_boxes]
    classes = read_data[fields.CoLocInputDataFields.original_groundtruth_classes]
    classes = classes - label_id_offset
    classes = tf.one_hot(tf.to_int32(classes), num_classes+1)
    target_class = read_data[fields.CoLocInputDataFields.groundtruth_target_class]
    return image, boxes, classes, target_class

  return zip(*map(extract_images_and_targets, read_data_list))


def _create_losses(input_queue, create_model_fn,
    use_target_class_in_predictions=False):
  """Creates loss function for a DetectionModel.

  Args:
    input_queue: BatchQueue object holding enqueued tensor_dicts.
    create_model_fn: A function to create the DetectionModel.
  """
  detection_model = create_model_fn()
  k_shot = detection_model._k_shot + detection_model._num_negative_bags
  bag_size = detection_model._bag_size

  (images, boxes_list, classes_list, target_class_list
  ) = _get_inputs(input_queue, detection_model.num_classes, k_shot, bag_size)

  def _r(t):
    return tf.reshape(t, [-1,]+t.shape[2:].as_list())
  def _s(t):
    return tf.split(_r(t), k_shot*bag_size)
  images_c = [detection_model.preprocess(k)
              for image in images
              for k in _s(image)]
  images = tf.concat(images_c, 0)

  groundtruth_boxes_list = [k for boxes in boxes_list for k in _s(boxes)]
  groundtruth_classes_list = [k for classes in classes_list for k in _s(classes)]

  groundtruth_masks_list = None
  detection_model.provide_groundtruth(groundtruth_boxes_list,
                                      groundtruth_classes_list,
                                      groundtruth_masks_list)
  if use_target_class_in_predictions:
    target_class = tf.stack(target_class_list)
    detection_model._groundtruth_lists['target_class'] = target_class

  prediction_dict = detection_model.predict(images)
  losses_dict = detection_model.loss(prediction_dict)
  for loss_tensor in losses_dict.values():
    tf.losses.add_loss(loss_tensor)

def train(create_tensor_dict_fn, create_model_fn, train_config, master, task,
          num_clones, worker_replicas, clone_on_cpu, ps_tasks, worker_job_name,
          is_chief, train_dir):
  """Training function for detection models.

  Args:
    create_tensor_dict_fn: a function to create a tensor input dictionary.
    create_model_fn: a function that creates a DetectionModel and generates
                     losses.
    train_config: a train_pb2.TrainConfig protobuf.
    master: BNS name of the TensorFlow master to use.
    task: The task id of this training instance.
    num_clones: The number of clones to run per machine.
    worker_replicas: The number of work replicas to train with.
    clone_on_cpu: True if clones should be forced to run on CPU.
    ps_tasks: Number of parameter server tasks.
    worker_job_name: Name of the worker job.
    is_chief: Whether this replica is the chief replica.
    train_dir: Directory to write checkpoints and training summaries to.
  """

  detection_model = create_model_fn()
  #data_augmentation_options = [
  #    preprocessor_builder.build(step)
  #    for step in train_config.data_augmentation_options]
  data_augmentation_options = None
  with tf.Graph().as_default():
    # Build a configuration specifying multi-GPU and multi-replicas.
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=num_clones,
        clone_on_cpu=clone_on_cpu,
        replica_id=task,
        num_replicas=worker_replicas,
        num_ps_tasks=ps_tasks,
        worker_job_name=worker_job_name)

    # Place the global step on the device storing the variables.
    with tf.device(deploy_config.variables_device()):
      global_step = slim.create_global_step()

    with tf.device(deploy_config.inputs_device()):
      input_queue, enqueue_thread = _create_input_queue(train_config.batch_size // num_clones,
                                        create_tensor_dict_fn,
                                        train_config.batch_queue_capacity,
                                        train_config.num_batch_queue_threads,
                                        train_config.prefetch_queue_capacity)

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    global_summaries = set([])

    model_fn = functools.partial(_create_losses,
                                 create_model_fn=create_model_fn,
                                 use_target_class_in_predictions=train_config.use_target_class_in_predictions)
    clones = model_deploy.create_clones(deploy_config, model_fn, [input_queue])
    first_clone_scope = clones[0].scope

    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by model_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    with tf.device(deploy_config.optimizer_device()):
      training_optimizer = optimizer_builder.build(train_config.optimizer,
                                                   global_summaries)

    sync_optimizer = None
    if train_config.sync_replicas:
      training_optimizer = tf.SyncReplicasOptimizer(
          training_optimizer,
          replicas_to_aggregate=train_config.replicas_to_aggregate,
          total_num_replicas=train_config.worker_replicas)
      sync_optimizer = training_optimizer

    # Create ops required to initialize the model from a given checkpoint.
    init_fn = None
    if train_config.fine_tune_checkpoint:
      var_map = detection_model.restore_map(
          from_detection_checkpoint=train_config.from_detection_checkpoint,
          load_all_detection_checkpoint_vars=True)
      reader = tf.pywrap_tensorflow.NewCheckpointReader(train_config.fine_tune_checkpoint)
      checkpoint_var_shape = reader.get_variable_to_shape_map()

      available_var_map = {}
      for name,var in var_map.items():
        if name in checkpoint_var_shape:
          if checkpoint_var_shape[name] == var.shape.as_list():
            available_var_map[name] = var
          else:
            print('***********Warning: {} variable size does not match: '.format(name),
                  'Expected size: {}'.format(var.shape.as_list()),
                  'Size in the checkpoint: {}'.format(checkpoint_var_shape[name]))
      init_saver = tf.train.Saver(available_var_map)
      def initializer_fn(sess):
        init_saver.restore(sess, train_config.fine_tune_checkpoint)
      init_fn = initializer_fn

    with tf.device(deploy_config.optimizer_device()):
      total_loss, grads_and_vars = model_deploy.optimize_clones(
          clones, training_optimizer, regularization_losses=None)
      total_loss = tf.check_numerics(total_loss, 'LossTensor is inf or nan.')

      if train_config.grad_multiplier:
        grads_and_vars = variables_helper.multiply_gradients_matching_regex(
            grads_and_vars,
            train_config.grad_multiplier_regex,
            multiplier=train_config.grad_multiplier)

      # Optionally multiply bias gradients by train_config.bias_grad_multiplier.
      if train_config.bias_grad_multiplier:
        biases_regex_list = ['.*/biases']
        grads_and_vars = variables_helper.multiply_gradients_matching_regex(
            grads_and_vars,
            biases_regex_list,
            multiplier=train_config.bias_grad_multiplier)

      # Optionally freeze some layers by setting their gradients to be zero.
      if train_config.freeze_variables:
        grads_and_vars = variables_helper.freeze_gradients_matching_regex(
            grads_and_vars, train_config.freeze_variables)

      # Optionally clip gradients
      if train_config.gradient_clipping_by_norm > 0:
        with tf.name_scope('clip_grads'):
          grads_and_vars = slim.learning.clip_gradient_norms(
              grads_and_vars, train_config.gradient_clipping_by_norm)

      # Create gradient updates.
      grad_updates = training_optimizer.apply_gradients(grads_and_vars,
                                                        global_step=global_step)
      update_ops.append(grad_updates)
      update_op = tf.group(*update_ops)
      with tf.control_dependencies([update_op]):
        train_tensor = tf.identity(total_loss, name='train_op')

    # Add summaries.
    for model_var in slim.get_model_variables():
      global_summaries.add(tf.summary.histogram(model_var.op.name, model_var))
    for loss_tensor in tf.losses.get_losses():
      global_summaries.add(tf.summary.scalar(loss_tensor.op.name, loss_tensor))
    global_summaries.add(
        tf.summary.scalar('TotalLoss', tf.losses.get_total_loss()))
    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))
    summaries |= global_summaries

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    # Soft placement allows placing on CPU ops without GPU implementation.
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0,
                                allow_growth=True)
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False,
                                    gpu_options=gpu_options)

    # Save checkpoints regularly.
    keep_checkpoint_every_n_hours = train_config.keep_checkpoint_every_n_hours
    saver = tf.train.Saver(
        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

    debug = False
    if debug:
      kwargs = dict(train_step_fn=learning.train_step,
                    save_summaries_secs=100000,
                    save_interval_secs=100000)
    else:
      #kwargs = dict(save_summaries_secs=100,
      #              save_interval_secs=150)
      kwargs = dict(save_summaries_secs=70,
                    save_interval_secs=100)

    def session_wrapper(sess):
      with sess.as_default():
        enqueue_thread.start()
      return sess
    slim.learning.train(
        train_tensor,
        logdir=train_dir,
        master=master,
        is_chief=is_chief,
        session_config=session_config,
        startup_delay_steps=train_config.startup_delay_steps,
        init_fn=init_fn,
        summary_op=summary_op,
        number_of_steps=(
            train_config.num_steps if train_config.num_steps else None),
        sync_optimizer=sync_optimizer,
        saver=saver,
        session_wrapper=session_wrapper,
        **kwargs)
