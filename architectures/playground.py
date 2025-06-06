import torch
import torch.nn as nn
import torch.nn.functional as F

class PlaygroundNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # (1, 28, 28)

            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1), # -> (64, 32, 32)
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), # -> (64, 32, 32)
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2), # -> (32, 16, 16)

            ##################
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # -> (64, 16, 16)
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # -> (64, 16, 16)
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2), # -> (64, 8, 8)
        )

        self.classification = nn.Sequential(
            nn.Flatten(),

            nn.Linear(64 * 8 * 8, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classification(x)
        return x