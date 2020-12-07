"""Function to build parameter learner from configuration."""

from rcnn_attention.coloc import parameter_learner
from object_detection.protos import parameter_learner_pb2
from rcnn_attention.builders import dynamic_hyperparams_builder

def build(argscope_fn, dynamic_argscope_fn, net_builder_fn,
          parameter_learner_config, is_training):
  """Builds parameter_learner based on the configuration.
  Args:
    argscope_fn: A function that takes the following inputs:
        * hyperparams_pb2.Hyperparams proto
        * a boolean indicating if the model is in training mode.
      and returns a tf slim argscope for Conv and FC hyperparameters.
    net_builder_fn:
    parameter_learner_config: parameter_learner_config.ParameterLearner proto containing
      configuration.
    is_training: Whether the models is in training mode.
  Returns:
    parameter_learner: parameter_learner.ParameterLearner object.
  Raises:
    ValueError: On unknown parameter learner.
  """
  if not isinstance(parameter_learner_config, parameter_learner_pb2.ParameterLearner):
    raise ValueError('parameter_learner_config not of type '
                     'parameter_learner_pb2.ParameterLearner.')
  parameter_learner_oneof = parameter_learner_config.WhichOneof('parameter_learner_oneof')
  if parameter_learner_oneof == 'weight_hashing_parameter_learner':
    weight_hashing_parameter_learner = parameter_learner_config.weight_hashing_parameter_learner
    fc_hyperparameters = argscope_fn(
      weight_hashing_parameter_learner.fc_hyperparameters, is_training)

    parameter_prediction_convline = net_builder_fn(argscope_fn, dynamic_argscope_fn,
      weight_hashing_parameter_learner.parameter_predictor_net, is_training)
    return parameter_learner.WeightHashingParameterLearner(
                              is_training,
                              output_scale=weight_hashing_parameter_learner.output_scale,
                              add_bias=weight_hashing_parameter_learner.add_bias,
                              one2one=weight_hashing_parameter_learner.one2one,
                              tanh_activation=weight_hashing_parameter_learner.tanh_activation,
                              parameter_prediction_convline=parameter_prediction_convline,
                              decompression_factor=weight_hashing_parameter_learner.decompression_factor,
                              fc_hyperparameters=fc_hyperparameters)

  raise ValueError('Unknown parameter learner: {}'.format(parameter_learner_oneof))
