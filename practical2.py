import torch
from torch import Tensor
import dlc_practical_prologue as prologue
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
    if type(mean)!=type(None):
        train_input = train_input-mean
        test_input = test_input-mean
    if type(proj)!=type(None):
        train_input = torch.mm(train_input,torch.t(proj))
        test_input = torch.mm(test_input,torch.t(proj))
    for ind in range(len(test_input)):
        test = test_input[ind]
        predicted = nearest_classification(train_input,train_target,test)
        if predicted != int(test_target[ind]):
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


#Exercise 4 
#Load MNIST 
train_input_mnist , train_target_mnist, test_input_mnist, test_target_mnist = prologue.load_data(flatten=True)

#Load CIFAR 
train_input_cifar , train_target_cifar, test_input_cifar, test_target_cifar = prologue.load_data(cifar=True, flatten=True)

#PCA comparison for mnist
num_components = 16
pca = PCA(n_components=num_components)
pca.fit(train_input_mnist) #Learn num_components principal components from training set
projection_matrix_pca = torch.FloatTensor(pca.components_)
compute_nb_errors(torch.tensor(train_input_mnist),train_target_mnist,torch.tensor(test_input_mnist),test_target_mnist,proj=projection_matrix_pca) #Errors for PCA as projection matrix
random_projection_matrix =  torch.empty(projection_matrix_pca.size()[0],projection_matrix_pca.size()[1]).normal_()
compute_nb_errors(torch.tensor(train_input_mnist),train_target_mnist,torch.tensor(test_input_mnist),test_target_mnist,proj=random_projection_matrix)

#PCA comparison for CIFAR
num_components = 16
pca = PCA(n_components=num_components)
pca.fit(train_input_cifar) #Learn num_components principal components from training set
projection_matrix_pca = torch.FloatTensor(pca.components_)
compute_nb_errors(torch.tensor(train_input_cifar),train_target_cifar,torch.tensor(test_input_cifar),test_target_cifar,proj=projection_matrix_pca) #Errors for PCA as projection matrix
random_projection_matrix =  torch.empty(projection_matrix_pca.size()[0],projection_matrix_pca.size()[1]).normal_()
compute_nb_errors(torch.tensor(train_input_cifar),train_target_cifar,torch.tensor(test_input_cifar),test_target_cifar,proj=random_projection_matrix)



