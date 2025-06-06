import torch
from torch import nn

# class VGGNet(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.features = nn.Sequential(
#             # 28x28
#             nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
#             nn.ReLU(),

#             # # 28x28
#             # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
#             # nn.ReLU(),

#             nn.MaxPool2d(kernel_size=2, stride=2),
#             # 14x14

#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(),

#             # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
#             # nn.ReLU(),

#             nn.MaxPool2d(kernel_size=2, stride=2),
#             # 7x7
#         )

#         self.classification = nn.Sequential(
#             nn.Flatten(),

#             nn.Linear(7 * 7 * 64, 128),
#             nn.ReLU(),

#             nn.Linear(128, 10),
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classification(x)
#         return x


class VGGNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # 28x28
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 14x14
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # # 7x7
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classification = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(7 * 7 * 128, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 10),
        )


    def forward(self, x):
        x = self.features(x)
        x = self.classification(x)
        return x
    