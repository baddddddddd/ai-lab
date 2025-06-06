import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor

from training.trainer import Trainer

class CIFAR100Trainer:
    def __init__(self, model, optimizer, batch_size=64):
        self.train_dataset = datasets.CIFAR100(
            root='./data',
            train=True,
            transform=ToTensor(),
            download=True
        )

        self.test_dataset = datasets.CIFAR100(
            root='./data',
            train=False,
            transform=ToTensor(),
            download=True
        )

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size)

        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.trainer = Trainer(self.train_dataloader, self.test_dataloader, self.model, self.loss_fn, self.optimizer)


    def train(self, epochs):
        self.trainer.fit(epochs)