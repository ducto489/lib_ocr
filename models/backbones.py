import torch.nn as nn
import timm


class resnet50(nn.Module):
    def __init__(self, input_channels, output_channels=512):
        super().__init__()
        backbone = timm.create_model("resnet50", pretrained=True)
        core = list(backbone.children())[:-2]
        self.backbone = nn.Sequential(*core)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

    def forward(self, x):
        x = self.backbone(x)
        x = self.AdaptiveAvgPool(x.permute(0, 3, 1, 2))
        x = x.squeeze(3)
        return x


class resnet18(nn.Module):
    def __init__(self, input_channels, output_channels=512):
        super().__init__()
        backbone = timm.create_model("resnet18", pretrained=True)
        core = list(backbone.children())[:-2]
        self.backbone = nn.Sequential(*core)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

    def forward(self, x):
        x = self.backbone(x)
        x = self.AdaptiveAvgPool(x.permute(0, 3, 1, 2))
        x = x.squeeze(3)
        return x


class VGG(nn.Module):
    def __init__(self, input_channels, output_channels=512):
        super().__init__()
        self.output_channels = [
            int(output_channels / 8),
            int(output_channels / 4),
            int(output_channels / 2),
            output_channels,
        ]  # 64, 128, 256, 512
        self.ConvNet = nn.Sequential(  # Input: 3 x 100 x 420
            nn.Conv2d(input_channels, self.output_channels[0], 3, 1, 0),
            nn.ReLU(True),  # 64 x 98 x 418
            nn.MaxPool2d(2, 2),  # 64 x 49 x 209
            nn.Conv2d(self.output_channels[0], self.output_channels[1], 3, 1, 0),
            nn.ReLU(True),  # 128 x 48 x 208
            nn.MaxPool2d(2, 2),  # 128 x 24 x 104
            nn.Conv2d(self.output_channels[1], self.output_channels[2], 3, 1, 0),
            nn.ReLU(True),  # 256 x 22 x 102
            nn.Conv2d(self.output_channels[2], self.output_channels[2], 3, 1, 0),
            nn.ReLU(True),  # 256 x 20 x 100
            nn.MaxPool2d(2, 2),  # 256 x 10 x 50
            nn.Conv2d(self.output_channels[2], self.output_channels[3], 3, 1, 0),
            nn.ReLU(True),  # 512 x 8 x 48
            nn.BatchNorm2d(self.output_channels[3]),
            nn.ReLU(True),
            nn.Conv2d(self.output_channels[3], self.output_channels[3], 3, 1, 0),
            nn.ReLU(True),  # 512 x 6 x 46
            nn.BatchNorm2d(self.output_channels[3]),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 512 x 3 x 23
            nn.Conv2d(self.output_channels[3], self.output_channels[3], 2, 1, 0),
            nn.ReLU(True),  # 512 x 1 x 21
        )

    def forward(self, x):
        conv = self.ConvNet(x)  # batch_size x 512 x 5 x 104
        # Reshape to (batch_size, sequence_length, channels) for CTC
        conv = conv.permute(0, 3, 1, 2)  # [b, w, c, h]
        conv = conv.squeeze(-1)  # [b, w, c]
        return conv  # [batch_size, sequence_length, channels]
