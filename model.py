import torch
import torch.nn as nn

# 2D CNN Model
class CNN2D(nn.Module):
    def __init__(self, num_classes):
        super(CNN2D, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        x = self.conv_layers(x)
        x = x.view(B, T, -1)
        x = x.mean(dim=1)
        return self.classifier(x)

# R(2+1)D CNN Model
class R2Plus1D(nn.Module):
    def __init__(self, num_classes):
        super(R2Plus1D, self).__init__()
        self.block1 = self._make_r2plus1d_block(3, 32)
        self.block2 = self._make_r2plus1d_block(32, 64)
        self.block3 = self._make_r2plus1d_block(64, 128)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(128, num_classes)

    def _make_r2plus1d_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, in_c, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(),
            nn.Conv3d(in_c, out_c, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_c),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )

    def forward(self, x):  # x: [B, T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)  # [B, 128, 1, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 128]
        return self.fc(x)
