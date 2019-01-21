# -*- coding: utf-8 -*-
"""Practical3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AdB24NP8zG9n6aPltIgN2im1-G_8Vjcu
"""

# http://pytorch.org/
from os.path import exists
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'

!pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.1-{platform}-linux_x86_64.whl torchvision
import torch

def sigma(x):
  return torch.tanh(x)

u =  torch.empty(5).normal_()

u

def dsigma(x):
  return 1-torch.tanh(x).pow(2)

sigma(u)



dsigma(u)

dsigma(u)

def loss(v,t):
  return torch.sum((v-t).pow(2))

x = torch.tensor([2.0,2.0])

y = torch.tensor([0.0,0.0])

loss(x,y)

def dloss(v,t):
  return 2*v-2*t

u = torch.empty(5).normal_()

v = torch.empty(5).normal_()

c = loss(u,v)

print(c)

u

v

u-v

torch.sum((u-v)*(u-v))

c

torch.mm

loss(u,v)

def forward_pass(w1, b1, w2, b2, x):
  s_1 = torch.add(torch.matmul(w1,x),b1)
  x_1 = sigma(s_1)
  s_2 = torch.add(torch.matmul(w2,x_1),b2)
  x_2 = sigma(s_2)
  return (x,s_1,x_1,s_2,x_2)

def backward_pass(w1, b1, w2, b2, t, x, s1, x1, s2, x2, dl_dw1, dl_db1, dl_dw2, dl_db2):
  dl_dx2 = dloss(x2,t)
  dl_ds2 = dl_dx2 * dsigma(s2)
  dl_dw2 = torch.matmul(dl_dx2,torch.t(x1))
  dl_db2 = dl_ds2
  
  dl_dx1 = dl_ds2 * w2
  dl_ds1 = dl_dx1 * dsigma(s1)
  dl_dw1 = torch.matmul(dl_dx1, torch.t(x))
  dl_db1 = dl_ds1