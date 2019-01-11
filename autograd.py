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