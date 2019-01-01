import torch
import time 
#Exercise 1
x = torch.full((13,13),1)
x[:,(1,6,11)]= 2
x[3:5,3:5]= 3
x[3:5,8:10]= 3
x[8:10,3:5]= 3
x[8:10,8:10]= 3

#Exercise 2
M = torch.empty(20,20).normal_()
inv_M = torch.inverse(M) 
diag_M = torch.diag(torch.diag(M))
e = torch.eig(torch.mm(torch.mm(inv_M,diag_M),M))

#Exercise 3
U = torch.empty(5000,5000).normal_()
V = torch.empty(5000,5000).normal_()
start_time = time.time()
torch.mm(U,V)
print("--- %s seconds ---" % (time.time() - start_time))

#Exercise 4

def mul_row(x):
  for i in range(1,len(x)+1):
    x[i-1] = i*x[i-1]
  return x 

def mul_fast(u):
  r = torch.arange(1.0,float(len(u)+1)).view(len(u),1)
  return torch.mul(r,u)

m = torch.full((1000, 400), 2.0)
n = torch.full((1000, 400), 2.0)

start_time = time.time()
mul_row(m)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
mul_fast(n)
print("--- %s seconds ---" % (time.time() - start_time))