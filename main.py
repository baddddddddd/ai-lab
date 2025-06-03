import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from architectures import *

# model = FashionMNISTNet()
model = MLP([28 * 28, 64, 64, 10], dropout=0.2)
# model = LeNet5().to(device)
print(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)



from training.datasets import MNISTTrainer

train = MNISTTrainer(model, optimizer)
train.train(10)