######################################################################
#
# This is free and unencumbered software released into the public domain.
# 
# Anyone is free to copy, modify, publish, use, compile, sell, or
# distribute this software, either in source code form or as a compiled
# binary, for any purpose, commercial or non-commercial, and by any
# means.
# 
# In jurisdictions that recognize copyright laws, the author or authors
# of this software dedicate any and all copyright interest in the
# software to the public domain. We make this dedication for the benefit
# of the public at large and to the detriment of our heirs and
# successors. We intend this dedication to be an overt act of
# relinquishment in perpetuity of all present and future rights to this
# software under copyright law.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
# 
# For more information, please refer to <http://unlicense.org/>
#
######################################################################

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

import dlc_practical_prologue as prologue

train_input, train_target, test_input, test_target = \
    prologue.load_data(one_hot_labels = True, normalize = True, flatten = False)

class Net(nn.Module):
    def __init__(self,hidden=200):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(256, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x

train_input, train_target = Variable(train_input), Variable(train_target)

model, criterion = Net(), nn.MSELoss()
eta, mini_batch_size = 1e-1, 200


def filter_val(x,val):
  if x==val:
    return 1
  else:
    return 0
def error(output,target):
  error_count =0
  for ind in range(len(output)):
    if not(torch.equal(output[ind],target[ind])):
        error_count+=1
  return error_count
def max_out(x):
    '''Winner takes all transformation on vector x'''
    max_value = torch.max(x)
    v = [filter_val(i,max_value) for i in x] 
    return torch.Tensor(v)



def compute_nb_errors(model, input, target, mini_batch_size):
    num_errors = 0 
    for b in range(0, train_input.size(0), mini_batch_size):
        output = model(input.narrow(0, b, mini_batch_size))
        output_take_all = [max_out(u) for u in output]
        target_take_all = target.narrow(0,b,mini_batch_size)
        num_errors+=error(output_take_all,target_take_all)
    return float(num_errors)/len(target)


def train_model(model, train_input, train_target, mini_batch_size):
    # We do this with mini-batches
    for e in range(0, 25):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            sum_loss = sum_loss + loss.item()
            model.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.data.sub_(eta * p.grad.data)
        print(e,sum_loss)

train_model(model, train_input, train_target, mini_batch_size)

print(compute_nb_errors(model, test_input, test_target, mini_batch_size))
