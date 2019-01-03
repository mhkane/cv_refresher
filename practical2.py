import torch
from torch import Tensor
import dlc_practical_prologue as prologue

#Exercise_1
def nearest_classification(train_input, train_target, x):
  dist_matrix = (train_input - x)*(train_input-x)
  index_shortest = int(torch.sort(torch.sum(dist_matrix,1))[1][0])
  label = int(train_target[index_shortest])
  return label


#Exercise_2
def compute_nb_errors(train_input, train_target, test_input, test_target, mean = None, proj = None):
	error_count =0
	if mean != None:
		train_input = train_input-mean
		test_input = test_input-mean
  	if proj != None:
    	train_input = torch.mm(train_input,torch.t(proj))
    	test_input = torch.mm(test_input,torch.t(proj))
  	for ind in range(len(test_input)):
    	predicted = nearest_classification(train_input,train_target,test)
    	if predicted != int(test_targert[ind]):
      		error_count+=1
	return error_count