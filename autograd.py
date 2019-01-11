import torch

x = torch.tensor([ 1., 2. ])

y = torch.tensor([ 4., 5. ])

z = torch.tensor([ 7., 3. ])

x.requires_grad

(x + y).requires_grad

z.requires_grad = True

(x + z).requires_grad

x = torch.tensor([1., 10.])

x.requires_grad = True

x = torch.tensor([1, 10])

w1 = torch.rand(20, 10).requires_grad_()
b1 = torch.rand(20).requires_grad_()
w2 = torch.rand(5, 20).requires_grad_()
b2 = torch.rand(5).requires_grad_()
x = torch.rand(10)
h = torch.tanh(w1 @ x + b1)
y = torch.tanh(w2 @ h + b2)
target = torch.rand(5)
loss = (y - target).pow(2).mean()