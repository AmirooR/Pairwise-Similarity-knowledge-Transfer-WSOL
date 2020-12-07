"""Pytorch dataset object that loads MNIST dataset as bags."""
from __future__ import print_function, division

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from functools import partial



class ImageBags(data_utils.Dataset):
  def __init__(self, bags, labels):
    self._np_bags = bags
    self._np_labels = labels
    self.num_bags = len(bags)
    self.bags_list = []
    self.labels_list = []
    for bag in bags:
      self.bags_list.append(torch.from_numpy(bag))
    for label in labels:
      self.labels_list.append(torch.from_numpy(label))
  def __len__(self):
      return len(self.labels_list)

  def __getitem__(self, index):
      bag = self.bags_list[index]
      label = [max(self.labels_list[index]), self.labels_list[index]]
      return bag, label


