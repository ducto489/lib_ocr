import torch
import torch.nn as nn
import torch.nn.functional as F
# CTC, Attention
class CTC_Predictor(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # torch.Size([16, 512, 2, 1])
        # x shape: [batch_size, dim, height, width]
        # b, c, h, w = x.size()
        return self.fc(x)
        
       
# Postprocessing: Greedy, Beam Search