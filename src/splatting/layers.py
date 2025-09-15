import torch
import torch.nn.functional as F

class ClassificationLayer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.num_classes = out_channels

    def forward(self, x):
        x = self.linear(x)
        x = F.softmax(x, dim=1)
        return x
