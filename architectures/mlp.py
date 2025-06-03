from torch import nn

class MLP(nn.Module):
    def __init__(self, layers, batch_norm=False, dropout=0.0):
        super().__init__()

        modules = []

        for i in range(1, len(layers) - 1):
            modules.append(nn.Linear(layers[i - 1], layers[i]))

            if batch_norm:
                modules.append(nn.BatchNorm1d(layers[i]))

            modules.append(nn.ReLU())

            if dropout > 0.0:
                modules.append(nn.Dropout(dropout))

        modules.append(nn.Linear(layers[-2], layers[-1]))

        self.model = nn.Sequential(*modules)
        self.flatten = nn.Flatten()


    def forward(self, x):
        x = self.flatten(x)
        x = self.model(x)
        return x
