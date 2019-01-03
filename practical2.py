import torch
from torch import Tensor
import dlc_practical_prologue as prologue

#Exercise_1
def nearest_classification(train_input, train_target, x):
  dist_matrix = (train_input - x)*(train_input-x)
  index_shortest = int(torch.sort(torch.sum(dist_matrix,1))[1][0])
  label = int(train_target[index_shortest])
  return label