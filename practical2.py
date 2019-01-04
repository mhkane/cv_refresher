import torch
from torch import Tensor
#import dlc_practical_prologue as prologue
from sklearn.decomposition import PCA 
import numpy as np

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

#Exercise 3
def PCA_2(x):
  assert(len(x)>1)
  mean = torch.mean(x,0)
  mean_shifted_x = x - mean
  n = len(x)
  cov = (1.0/(n-1)) * torch.mm(torch.t(mean_shifted_x),mean_shifted_x)
  eig_val,eig_vec = torch.eig(cov,True)
  sorted, indices = torch.sort(eig_val,0,descending=True)
  list_index = indices[:,0]
  eig_vec = eig_vec[list_index,:]
  return mean, eig_vec



#Test PCA implementation

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X)
print(pca.components_)


# Test mine 

Y = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = torch.FloatTensor(Y)
mean, eig_vec = PCA_2(Y)
print eig_vec