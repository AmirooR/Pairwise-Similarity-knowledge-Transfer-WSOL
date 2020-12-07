import tensorflow as tf

from google.protobuf import text_format
from object_detection.protos import attention_pipeline_pb2

import os

flags = tf.app.flags
flags.DEFINE_string('config_template_path', '', 'Path to the template config')
flags.DEFINE_boolean('is_training', True, 'Whether the config is used for training')
flags.DEFINE_string('split', '', 'Name of the split for training or evaluation')
flags.DEFINE_integer('train_num_steps', 1000, 'Training steps')
flags.DEFINE_integer('eval_num_examples', 1000, 'number of examples for evaluation')
flags.DEFINE_boolean('aggregate', False, 'do aggregation?')
flags.DEFINE_string('aggregate_save_split', 'temp', 'save split name for aggregation')
flags.DEFINE_string('aggregate_update_split', 'temp', 'update split name for aggregation')
flags.DEFINE_string('write_path', '', 'config write path')
flags.DEFINE_integer('ncobj_proposals', 500, 'number of proposals for k2,k4')
flags.DEFINE_integer('eval_fold', None, 'evaluation fold')
flags.DEFINE_integer('num_folds', 10, 'num folds')
flags.DEFINE_float('transfer_objectness', None, 'objectness transfer')
flags.DEFINE_string('bcd_init', None, 'bcd_path')
flags.DEFINE_integer('k_shot', None, 'k shot')
flags.DEFINE_integer('save_eval_freq', None, 'icm epochs')
FLAGS= flags.FLAGS

def read_pipeline_config(config_path):
  pipeline_config = attention_pipeline_pb2.AttentionTrainEvalPipelineConfig()
  assert os.path.exists(config_path), 'Config file does not exist'
  with open(config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)
  return pipeline_config

def write_pipeline_config(config_path, config):
  with open(config_path, 'w') as f:
    f.write(text_format.MessageToString(config))

def main(_):
  config = read_pipeline_config(FLAGS.config_template_path)
  if FLAGS.is_training:
    config.train_input_reader.mil_det_fea_input_reader.split = FLAGS.split
    config.train_input_reader.mil_det_fea_input_reader.support_db_name = FLAGS.split
    config.train_config.num_steps = FLAGS.train_num_steps
    if FLAGS.eval_fold is not None:
      train_folds = [i for i in range(FLAGS.num_folds) if i != FLAGS.eval_fold]
      if FLAGS.num_folds == 1:
        train_folds = [0]
      config.train_input_reader.mil_det_fea_input_reader.folds.extend(train_folds)
  else:
    if FLAGS.k_shot is not None:
      config.model.wrn_attention.k_shot = FLAGS.k_shot
    if FLAGS.save_eval_freq is not None:
      config.eval_config.aggregation_params.save_eval_freq = FLAGS.save_eval_freq
    if FLAGS.transfer_objectness is not None:
      config.model.wrn_attention.attention_tree.unit[0].transfered_objectness_weight = FLAGS.transfer_objectness
    if len(config.model.wrn_attention.attention_tree.unit) > 1:
      config.model.wrn_attention.attention_tree.unit[1].ncobj_proposals = FLAGS.ncobj_proposals
    if len(config.model.wrn_attention.attention_tree.unit) > 2:
      config.model.wrn_attention.attention_tree.unit[2].ncobj_proposals = FLAGS.ncobj_proposals
    config.eval_config.num_examples = FLAGS.eval_num_examples
    config.eval_config.aggregation_params.aggregate = FLAGS.aggregate
    if FLAGS.aggregate:
      config.eval_config.aggregation_params.save_split = FLAGS.aggregate_save_split
      config.eval_config.aggregation_params.update_split = FLAGS.aggregate_update_split
    if FLAGS.eval_fold is not None:
      eval_folds = [FLAGS.eval_fold]
      config.eval_input_reader[0].mil_det_fea_input_reader.folds.extend(eval_folds)
    if FLAGS.bcd_init is not None:
      config.eval_config.bcd_init = FLAGS.bcd_init
  write_pipeline_config(FLAGS.write_path, config)

if __name__ == '__main__':
  tf.app.run()

