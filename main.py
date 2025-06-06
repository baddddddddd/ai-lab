import torch
from torch import nn

from training.datasets import *
from architectures import *



model = PlaygroundNet()
print(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

trainer = CIFAR10Trainer(model, optimizer)
trainer.train(10)