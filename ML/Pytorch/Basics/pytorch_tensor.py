import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

arr = np.array([1, 2, 3])
my_tensor = torch.tensor(arr)
print(my_tensor)
arr[0] = 11
print(arr)
print(my_tensor.tolist())
print(my_tensor.numpy())
print("---------")
print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.shape)
print(my_tensor.requires_grad)

print(torch.tensor(2))
x1 = torch.empty(size=(3, 3), device=device)
print(x1)
print(x1.long())

print(torch.rand(size=(3, 3), device=device))
print(torch.arange(3))
print(torch.arange(1, 3))
print(torch.arange(start=1, end=10, step=2))
print(torch.linspace(3, 10, steps=3))
print(torch.linspace(-1, 1, steps=5))
print(torch.logspace(-4, 0, steps=5, base=10))
print(torch.normal(mean=torch.tensor(1.0), std=torch.tensor(1.0)))
print(torch.empty(size=(3, 3)).normal_(10, 20))

print(torch.tensor([1, 2, 3]) + torch.tensor([4, 3, 1]))
print(torch.tensor([1, 2, 3]) + 2)
print(torch.tensor([1, 2, 3]).pow(2))
print(torch.tensor([1, 2, 3]) ** 2)

x = torch.randn(size=(5, 3))
print(torch.max(x, dim=1))
print(torch.argmax(x, dim=1))
print(torch.mean(x.float(), dim=0))
print(torch.clamp(x, min=0, max=0.5))

x = torch.tensor([0, 1, 0, 1, 1], dtype=torch.bool)
print(torch.all(x))
print(torch.any(x))
print(x[0])

x = torch.randn(size=(5, 3))
print(x[:, 0])
print(x[0, 0])
print(x[0, :])
print(x[0, 0:2])

x = torch.arange(10)
print(x[[2, 3, 4]])
print(x.remainder(2))

x = torch.randn(size=(5, 3))
print(x)
print(x[[1, 2], [1, 2]])

x = torch.arange(9)
print(x.view(3, 3))
print(x.reshape(3, 3))
print(x.reshape(3, 3).ndim)
print(x.reshape(3, 3).permute(1, 0))

x1 = torch.randn(size=(5, 3))
x2 = torch.randn(size=(5, 3))
print(torch.cat((x1, x2), dim=0).shape)
print(torch.cat((x1, x2), dim=1).shape)

print("---------")
x = torch.randn(size=(2, 2, 3))
print(x)
print("---------")
print(x.reshape(2, -1))
print("---------")
print(x.reshape(-1, 3))


print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)
print(x.squeeze(1).shape)
print(x.unsqueeze(0).unsqueeze(1).shape)
print(x.unsqueeze(0).unsqueeze(1).squeeze().shape)
print(x.unsqueeze(0).unsqueeze(1).squeeze(0).shape)

x = torch.tensor([0, 1, 0, 1, 1])
print(x.bool())
print(x.data)