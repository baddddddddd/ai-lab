from torch import nn


# 28x28 version
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            # C1
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.Tanh(),

            # S2
            nn.AvgPool2d(kernel_size=2, stride=2),

            # C3
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0),
            nn.Tanh(),

            # S4
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            nn.Linear(5 * 5 * 16, 120),
            nn.Tanh(),

            nn.Linear(120, 84),
            nn.Tanh(),

            nn.Linear(84, 10)
        )


    def forward(self, x):
        return self.cnn(x)