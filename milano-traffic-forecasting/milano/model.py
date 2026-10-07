"""Same GRU architecture for both feature sets."""
import torch
from torch import nn


class TrafficGRU(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.output(hidden[-1]).squeeze(-1)
