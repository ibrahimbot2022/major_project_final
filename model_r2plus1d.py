import torch.nn as nn

class R2Plus1D(nn.Module):
    def __init__(self, num_classes):
        super(R2Plus1D, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2))
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2))
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x)
        return x
