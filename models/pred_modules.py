import torch.nn as nn


class CTC(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# Postprocessing: Greedy, Beam Search


class Attention:
    def __init__(self, input_dim, hidden_dim, output_dim): ...

    def forward(self, x): ...
