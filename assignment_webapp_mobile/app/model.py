import torch
from torch import nn


class NonlinearModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_batch = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        return self.layer_batch(x)
