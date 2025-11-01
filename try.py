
import torch
print(torch.__version__)  # Should print the installed PyTorch version
print(torch.cuda.is_available())  # Should return True if CUDA is enabled
print(torch.cuda.device_count())  # Should show the number of GPUs
print(torch.cuda.get_device_name(0))  # Should print your GPU name

x = torch.rand(3, 3)
x_cuda = x.to('cuda')
print(x_cuda)  # Should print a tensor on CUDA device

import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc1(x)

model = SimpleNN().to('cuda')
print(model)
