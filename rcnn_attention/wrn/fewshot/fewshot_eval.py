"""Evaluation executable for coloc detection models.

This executable is used to evaluate CoLocDetectionModels. There are two ways of
configuring the eval job.

1) A single pipeline_pb2.TrainEvalPipelineConfig file maybe specified instead.
In this mode, the --eval_training_data flag may be given to force the pipeline
to evaluate on training data instead.

Example usage:
    ./eval \
        --logtostderr \
        --checkpoint_dir=path/to/checkpoint_dir \
        --eval_dir=path/to/eval_dir \
        --pipeline_config_path=pipeline_config.pbtxt

2) Three configuration files may be provided: a model_pb2.DetectionModel
configuration file to define what type of DetectionModel is being evaulated, an
input_reader_pb2.InputReader file to specify what data the model is evaluating
and an eval_pb2.EvalConfig file to configure evaluation parameters.

Example usage:
    ./eval \
        --logtostderr \
        --checkpoint_dir=path/to/checkpoint_dir \
        --eval_dir=path/to/eval_dir \
        --eval_config_path=eval_config.pbtxt \
        --model_config_path=model_config.pbtxt \
        --input_config_path=eval_input_config.pbtxt
"""
import functools
import json
import os
import tensorflow as tf
from google.protobuf import text_format

from rcnn_attention.wrn.fewshot import fewshot_evaluator as evaluator
from rcnn_attention.builders import dataflow_builder
from rcnn_attention.builders import attention_model_builder
from object_detection.protos import coloc_eval_pb2
from object_detection.protos import coloc_input_reader_pb2
from object_detection.protos import attention_model_pb2
from object_detection.protos import attention_pipeline_pb2
from object_detection.utils import label_map_util

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_boolean('eval_training_data', False,
                     'If training data should be evaluated for this job.')
flags.DEFINE_string('checkpoint_dir', '',
                    'Directory containing checkpoints to evaluate, typically '
                    'set to `train_dir` used in the training job.')
flags.DEFINE_string('eval_dir', '',
                    'Directory to write eval summaries to.')
flags.DEFINE_string('pipeline_config_path', '',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')
flags.DEFINE_string('eval_config_path', '',
                    'Path to an eval_pb2.EvalConfig config file.')
flags.DEFINE_string('input_config_path', '',
                    'Path to an input_reader_pb2.InputReader config file.')
flags.DEFINE_string('model_config_path', '',
                    'Path to a model_pb2.DetectionModel config file.')

flags.DEFINE_string('mrf_type', 'dense_trws', 'MRF type. chain, loop or dense_{map,trws,trbp,astar}.')
flags.DEFINE_boolean('add_mrf', False, 'whether to add results related to mrf or not')
flags.DEFINE_boolean('add_unaries', True, 'whether to add k1 scores as unaries or not')
flags.DEFINE_boolean('add_brute_force', False, 'whether to add k1 scores as unaries or not')
flags.DEFINE_boolean('calibration', False, 'perform calibration mode')
flags.DEFINE_float('pairwise_power', 0.0, 'scale of pairwise potentials')
# confidence calibration part in eval
flags.DEFINE_boolean('add_calibration', False, 'adds calibration function on output of network before evaluation')
flags.DEFINE_string('calib_prefix', 'val', 'calibration set to use one value from (val, test, train_seen, train_unseen)')
flags.DEFINE_string('calib_folder', 'histograms/pcvpr_sigmoid_loss', 'root folder for stored calibration data')
flags.DEFINE_boolean('add_calib_platt', True, 'adds platt calibration function on output')
flags.DEFINE_boolean('add_calib_isotonic', True, 'adds isotonic regression on histograms')
flags.DEFINE_boolean('add_calib_histogram', True, 'adds calibration based on raw histograms')
flags.DEFINE_integer('calib_nbins', 100, 'number of bins for histogram and isotonic mode')
flags.DEFINE_boolean('calib_histogram_normed', True, 'whether use normalized histograms or not')

FLAGS = flags.FLAGS


def get_configs_from_pipeline_file():
  """Reads evaluation configuration from a pipeline_pb2.TrainEvalPipelineConfig.

  Reads evaluation config from file specified by pipeline_config_path flag.

  Returns:
    model_config: a model_pb2.DetectionModel
    eval_config: a eval_pb2.EvalConfig
    input_config: a input_reader_pb2.InputReader
  """
  pipeline_config = attention_pipeline_pb2.AttentionTrainEvalPipelineConfig()
  with tf.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)

  model_config = pipeline_config.model
  if FLAGS.eval_training_data:
    eval_config = pipeline_config.train_config
  else:
    eval_config = pipeline_config.eval_config
  input_config = pipeline_config.eval_input_reader
  return model_config, eval_config, input_config


def get_configs_from_multiple_files():
  """Reads evaluation configuration from multiple config files.

  Reads the evaluation config from the following files:
    model_config: Read from --model_config_path
    eval_config: Read from --eval_config_path
    input_config: Read from --input_config_path

  Returns:
    model_config: a model_pb2.DetectionModel
    eval_config: a eval_pb2.EvalConfig
    input_config: a input_reader_pb2.InputReader
  """
  eval_config = coloc_eval_pb2.EvalConfig()
  with tf.gfile.GFile(FLAGS.eval_config_path, 'r') as f:
    text_format.Merge(f.read(), eval_config)

  model_config = attention_model_pb2.AttentionModel()
  with tf.gfile.GFile(FLAGS.model_config_path, 'r') as f:
    text_format.Merge(f.read(), model_config)

  input_config = coloc_input_reader_pb2.InputReader()
  with tf.gfile.GFile(FLAGS.input_config_path, 'r') as f:
    text_format.Merge(f.read(), input_config)

  return model_config, eval_config, input_config

def main(unused_argv):
  assert FLAGS.checkpoint_dir, '`checkpoint_dir` is missing.'
  assert FLAGS.eval_dir, '`eval_dir` is missing.'
  if FLAGS.pipeline_config_path:
    model_config, eval_config, input_config = get_configs_from_pipeline_file()
  else:
    model_config, eval_config, input_config = get_configs_from_multiple_files()

  input_k_shot = model_config.wrn_attention.k_shot
  tree_k_shot = 2 if FLAGS.add_mrf else input_k_shot
  model_fn = functools.partial(
      attention_model_builder.build,
      attention_model_config=model_config,
      is_training=False,
      is_calibration=FLAGS.calibration,
      k_shot=tree_k_shot)

  create_input_dict_fn_list = []
  input_config_names = []
  for ic in input_config:
    if ic.HasField('mil_input_reader'):
      reader_config = ic.mil_input_reader
      input_config_names.append('{0}-training_{1}'.format(
                                reader_config.split,
                                reader_config.is_training))
    elif ic.HasField('mil_det_fea_input_reader'):
      reader_config = ic.mil_det_fea_input_reader
      input_config_names.append(reader_config.support_db_name)
    else:
      raise Except('Unkown input reader')
    create_input_dict_fn_list.append(functools.partial(
        dataflow_builder.build,
        dataflow_config=ic,
        k_shot=input_k_shot,
        is_training=False,
        queue_size=5,
        bag_size=model_config.wrn_attention.bag_size,
        use_features=model_config.wrn_attention.use_features,
        num_negative_bags=model_config.wrn_attention.num_negative_bags))

  categories = None
  evaluator.evaluate(create_input_dict_fn_list, input_config_names,
                     model_fn, eval_config, categories,
                     FLAGS.checkpoint_dir, FLAGS.eval_dir,
                     input_k_shot)

if __name__ == '__main__':
  tf.app.run()
