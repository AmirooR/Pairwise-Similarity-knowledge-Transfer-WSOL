from object_detection.protos import attention_losses_pb2
from rcnn_attention.attention import losses

def build(loss_config, num_classes, k):
  if not isinstance(loss_config, attention_losses_pb2.AttentionLoss):
    raise ValueError('loss_config not of type '
                     'attention_losses_pb2.AttentionLoss.')

  attention_loss_oneof = loss_config.WhichOneof('attention_loss_oneof')
  class_agnostic = loss_config.class_agnostic
  if attention_loss_oneof == 'softmax_cross_entropy_loss':
    softmax_cross_entropy_loss = loss_config.softmax_cross_entropy_loss
    min_match_frac = softmax_cross_entropy_loss.min_match_frac
    return losses.SoftmaxCrossEntropyLoss(class_agnostic,
                                          num_classes,
                                          k, min_match_frac)
  elif attention_loss_oneof == 'only_for_testing_average_loss':
    only_for_testing_average_loss = loss_config.only_for_testing_average_loss
    min_match_frac = only_for_testing_average_loss.min_match_frac
    return losses.OnlyForTestingAverage(class_agnostic,
                                          num_classes,
                                          k, min_match_frac)
  elif attention_loss_oneof == 'l2_loss':
    l2_loss = loss_config.l2_loss
    min_match_frac = l2_loss.min_match_frac
    return losses.L2Loss(class_agnostic, num_classes,
                         k, min_match_frac)
  elif attention_loss_oneof == 'kl_divergence_loss':
    kl_divergence_loss = loss_config.kl_divergence_loss
    witness_rate = kl_divergence_loss.witness_rate
    min_match_frac = kl_divergence_loss.min_match_frac
    return losses.ApproxKLDivergenceLoss(class_agnostic,
                                         num_classes,
                                         k,
                                         witness_rate,
                                         min_match_frac,
                                         kl_divergence_loss.partial)
  elif attention_loss_oneof == 'pairwise_estimate_loss':
    pairwise_estimate_loss = loss_config.pairwise_estimate_loss
    return losses.PairwiseEstimateLoss(class_agnostic, num_classes, k)
  elif attention_loss_oneof == 'pairwise_estimate_v2_loss':
    pairwise_estimate_loss = loss_config.pairwise_estimate_loss
    return losses.PairwiseEstimateLossV2(class_agnostic, num_classes, k)
  elif attention_loss_oneof == 'soft_sigmoid_cross_entropy_loss':
    soft_sigmoid_cross_entropy_loss = loss_config.soft_sigmoid_cross_entropy_loss
    return losses.SoftSigmoidCrossEntropyLoss(k,
            label_type=soft_sigmoid_cross_entropy_loss.type)
  elif attention_loss_oneof == 'sigmoid_cross_entropy_loss':
    sigmoid_cross_entropy_loss = loss_config.sigmoid_cross_entropy_loss
    return losses.SigmoidCrossEntropyLoss(class_agnostic, num_classes, k,
                                          focal_loss=sigmoid_cross_entropy_loss.focal_loss)
  elif attention_loss_oneof == 'rank_loss':
    rank_loss = loss_config.rank_loss
    return losses.RankLoss(k)
  raise ValueError('Unkonwn loss: {}'.format(attention_loss_oneof))
