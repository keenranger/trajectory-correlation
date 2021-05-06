from torch import nn


class AnnModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_size, 64), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(32, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
