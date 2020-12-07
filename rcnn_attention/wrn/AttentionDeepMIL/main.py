from __future__ import print_function, division

import numpy as np
#import tensorflow as tf
import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

from dataloader import ImageBags
from model import Attention
from functools import partial
from rcnn_attention.wrn.misvm.data_reader import get_data_reader

parser = argparse.ArgumentParser(description='PyTorch MINIImageNet bags Example')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--seed', type=int, default=1357, metavar='S',
                    help='random seed (default: 1357)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--k', type=int, default=4, metavar='K',
                    help='k_shot (default: 4)')
parser.add_argument('--b', type=int, default=5, metavar='B',
                    help='bag size (default: 5)')
parser.add_argument('--eval_num', type=int, default=500, metavar='E',
                    help='num evals (default: 500)')
parser.add_argument('--neg', type=int, default=2, metavar='NE',
                    help='num negative bags (default: 2)')
parser.add_argument('--split', type=str, default="test", metavar='SP',
                    help='split to use for evaluation (default: test)')
parser.add_argument('--dataset', type=str, default='miniimagenet', metavar='DS',
                    help='name of the dataset (default: miniimagenet)')
#coco related
parser.add_argument('--data_root', default='/home/amir/CoLoc/rcnn_attention/feature_extractor/logs/coco/k1/extract{}', metavar='DR', help='path to features for coco')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

def add_results(result_dict, result_lists):
  for key,val in result_dict.items():
    if isinstance(val, dict):
      if key not in result_lists:
        result_lists[key] = dict()
      add_results(val, result_lists[key])
    else:
      if key not in result_lists:
        result_lists[key] = []
      result_lists[key].extend(val)

def main(reader, params):
  #from rcnn_attention import evaluator
  k_shot = params.k
  num_negative_bags = params.neg
  total_bags = k_shot + num_negative_bags
  result_lists = {}
  input_dim = 256 if params.dataset == 'omniglot' else 640

  for i, tensor_data in enumerate(reader.get_data()):
    if (i+1) % 100 == 0:
      print('Evaluating problem number %d/%d' % (i+1, params.eval_num))
    [feas, fea_boxes, fea_target_classes,
        fea_classes, imgs, target_class] = tensor_data[0:6]
    boxes_list = tensor_data[6:6+total_bags]
    class_list = tensor_data[6+total_bags:]
    bags = np.squeeze(feas)
    bag_labels = np.max(fea_target_classes, axis=1)
    input_labels = fea_target_classes.astype(np.int64)
    train_loader = data_utils.DataLoader(ImageBags(bags=bags, labels=input_labels),
                                         batch_size=1,
                                         shuffle=True,
                                         **loader_kwargs)
    test_loader = data_utils.DataLoader(ImageBags(bags=bags, labels=input_labels),
                                                         batch_size=1,
                                                         shuffle=False,
                                                         **loader_kwargs)
    model = Attention(input_dim=input_dim)
    if params.cuda:
      model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.999), weight_decay=params.reg)

    def train(epoch):
      model.train()
      train_loss = 0.
      train_error = 0.
      for batch_idx, (data, label) in enumerate(train_loader):
        bag_label = label[0]
        if params.cuda:
          data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        loss, _ = model.calculate_objective(data, bag_label)
        train_loss += loss.data[0]
        #error, _ = model.calculate_classification_error(data, bag_label)
        #train_error += error
        # backward pass
        loss.backward()
        # step
        optimizer.step()

      train_loss /= len(train_loader)
      #print('epoch: {}, loss: {}'.format(epoch, train_loss))
      #train_error /= len(train_loader)

    def test():
      model.eval()
      test_loss = 0.
      test_error = 0.
      num_success = 0
      scores = np.zeros_like(fea_classes[:params.k])
      for batch_idx, (data, label) in enumerate(test_loader):
        bag_label = label[0]
        instance_labels = label[1]
        if params.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        loss, attention_weights = model.calculate_objective(data, bag_label)
        test_loss += loss.data[0]
        #error, predicted_label = model.calculate_classification_error(data, bag_label)
        #test_error += error
        if batch_idx < params.k:
          scores[batch_idx] = attention_weights.cpu().data.numpy()[0]
          #argmax_pred = np.argmax(attention_weights.cpu().data.numpy()[0])
          #val = instance_labels.numpy()[0].tolist()[argmax_pred]
          #num_success += val
          #print('batch idx: {}, val: {}'.format(batch_idx, val))
      #print('scores: ', scores)
      res = {'boxes': fea_boxes[:params.k],
             'classes': np.ones_like(fea_classes[:params.k]),
             'scores': scores,
             'class_agnostic': True}
      return res

    gt = {}
    gt['boxes'] = boxes_list[:params.k]
    gt['classes'] = class_list[:params.k]
    gt['target_class'] = target_class
    for epoch in range(1, args.epochs + 1):
      train(epoch)
    res = test()
    result_dict = {'groundtruth': gt,
                   'atnmil': res}
    from rcnn_attention import evaluator
    evaluator._postprocess_result_dict(result_dict)
    result_dict.pop('groundtruth')
    add_results(result_dict, result_lists)
    if i+1 == params.eval_num:
      break
  metrics = {}
  from rcnn_attention import eval_util
  for method, result_list in result_lists.items():
    m = eval_util.evaluate_coloc_results(result_list, None)
    metrics[method] = m
  for k,v in metrics.items():
    print('{}: {}'.format(k,v))

if __name__ == '__main__':
  args.data_root = args.data_root.format(args.b)
  print('Loading dataset')
  reader = get_data_reader(args)
  print(' - done.')
  main(reader, args)
