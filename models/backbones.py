import torch.nn as nn
import timm

class resnet50(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = timm.create_model('resnet50', pretrained=True)
        core = list(backbone.children())[:-2]
        self.backbone = nn.Sequential(*core)    
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

    def forward(self, x):
        x = self.backbone(x)
        x = self.AdaptiveAvgPool(x.permute(0, 3, 1, 2))
        x = x.squeeze(3)
        return x

class resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = timm.create_model('resnet18', pretrained=True)
        core = list(backbone.children())[:-2]
        self.backbone = nn.Sequential(*core)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

    def forward(self, x):
        x = self.backbone(x)
        x = self.AdaptiveAvgPool(x.permute(0, 3, 1, 2))
        x = x.squeeze(3)
        return x