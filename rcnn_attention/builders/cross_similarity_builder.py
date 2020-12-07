"""Function to build cross similarity from configuration."""

from rcnn_attention.attention import cross_similarity
from object_detection.protos import cross_similarity_pb2
from rcnn_attention.builders import convline_builder
from rcnn_attention.builders import negative_attention_builder

def build_cross_similarity(argscope_fn, cross_similarity_config, attention_tree, k, is_training):
  cross_similarity_oneof = cross_similarity_config.WhichOneof('cross_similarity_oneof')
  if cross_similarity_oneof == 'cosine_cross_similarity':
    cosine_cross_similarity = cross_similarity_config.cosine_cross_similarity
    return cross_similarity.CosineCrossSimilarity()
  elif cross_similarity_oneof == 'linear_cross_similarity':
    linear_cross_similarity = cross_similarity_config.linear_cross_similarity
    fc_hyperparameters = argscope_fn(
      linear_cross_similarity.fc_hyperparameters, is_training)
    return cross_similarity.LinearCrossSimilarity(fc_hyperparameters)
  elif cross_similarity_oneof == 'deep_cross_similarity':
    deep_cross_similarity = cross_similarity_config.deep_cross_similarity
    fc_hyperparameters = argscope_fn(
      deep_cross_similarity.fc_hyperparameters, is_training)
    convline = None
    if deep_cross_similarity.HasField('convline'):
      convline = convline_builder.build(argscope_fn,
                                        None,
                                        deep_cross_similarity.convline,
                                        is_training)
    negative_attention = None
    if deep_cross_similarity.HasField('negative_attention'):
      negative_attention = negative_attention_builder.build(argscope_fn,
                                                            deep_cross_similarity.negative_attention,
                                                            is_training)
    return cross_similarity.DeepCrossSimilarity(
        deep_cross_similarity.stop_gradient,
        fc_hyperparameters, convline,
        negative_attention,
        sum_output=deep_cross_similarity.sum_output)
  elif cross_similarity_oneof == 'average_cross_similarity':
    average_cross_similarity = cross_similarity_config.average_cross_similarity
    return cross_similarity.AverageCrossSimilarity()
  elif cross_similarity_oneof == 'euclidean_cross_similarity':
    return cross_similarity.EuclideanCrossSimilarity()
  elif cross_similarity_oneof == 'pairwise_cross_similarity':
    pairwise_cross_similarity = cross_similarity_config.pairwise_cross_similarity
    base_cross_similarity = build_cross_similarity(argscope_fn,
                                                   pairwise_cross_similarity.cross_similarity,
                                                   attention_tree,
                                                   k,
                                                   is_training)
    return cross_similarity.PairwiseCrossSimilarity(pairwise_cross_similarity.stop_gradient,
                                                    base_cross_similarity, k,
                                                    attention_tree=attention_tree)
  elif cross_similarity_oneof == 'k1_cross_similarity':
    k1_cross_similarity = cross_similarity_config.k1_cross_similarity
    base_cross_similarity = build_cross_similarity(argscope_fn,
                                                   k1_cross_similarity.cross_similarity,
                                                   attention_tree,
                                                   k,
                                                   is_training)
    return cross_similarity.K1CrossSimilarity(base_cross_similarity, k,
                    k1_cross_similarity.share_weights_with_pairwise_cs,
                                              k1_cross_similarity.mode,
                                              k1_cross_similarity.topk)
  elif cross_similarity_oneof == 'double_cross_similarity':
    double_cross_similarity = cross_similarity_config.double_cross_similarity
    main_cs = build_cross_similarity(argscope_fn,
                                     double_cross_similarity.main,
                                     attention_tree, k, is_training)
    if double_cross_similarity.HasField('transfered'):
      transfered_config = double_cross_similarity.transfered
    else:
      transfered_config = double_cross_similarity.main

    transfered_cs = build_cross_similarity(argscope_fn,
                                           transfered_config,
                                           attention_tree,
                                           k,
                                           is_training)

    ## PairwiseCrossSimilarity overrides the scope
    ## We need these to ensure main and transfered cs does not share variables
    if isinstance(main_cs, cross_similarity.PairwiseCrossSimilarity):
      main_cs._k2_scope_key = 'main_pairwise_cross_similarity'
    if isinstance(transfered_cs, cross_similarity.PairwiseCrossSimilarity):
      transfered_cs._k2_scope_key = 'transfered_pairwise_cross_similarity'

    return cross_similarity.DoubleCrossSimilarity(main_cs, transfered_cs,
                                                  double_cross_similarity.main_weight,
                                                  double_cross_similarity.fea_split_ind)

  raise ValueError('Unknown cross_similarity: {}'.format(cross_similarity_oneof))

def build(argscope_fn, cross_similarity_config, attention_tree, k, is_training):
  """Builds cross similarity method based on the configuration.
  Args:
    argscope_fn: A function that takes the following inputs:
        * hyperparams_pb2.Hyperparams proto
        * a boolean indicating if the model is in training mode.
      and returns a tf slim argscope for Conv and FC hyperparameters.
    cross_similarity_config: cross similarity configuration.
    is_training: Whether the models is in training mode.
  Returns:
    cross_similarity: attention.cross_similarity.CrossSimilarity object.
  Raises:
    ValueError: On unknown parameter learner.
  """
  if not isinstance(cross_similarity_config, cross_similarity_pb2.CrossSimilarity):
    raise ValueError('cross_similarity_config not of type '
                     'cross_similarity_pb2.CrossSimilarity.')

  return build_cross_similarity(argscope_fn, cross_similarity_config, attention_tree, k, is_training)
