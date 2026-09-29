import torch
import torch.nn as nn
import torch.nn.functional as F
import parameter

parameter._init()


class ConvDownsample(nn.Module):
    """Standard convolutional downsampling used in the public pre-release backbone."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class GSSCNetPublic(nn.Module):
    """Public pre-acceptance backbone.

    This file intentionally contains only the generic multimodal feature-extraction
    and classification pipeline. The paper-specific core modules are not included
    in this pre-release repository.
    """

    def __init__(self, out_features, radar_channels, ms_channels):
        super().__init__()

        channels = parameter.get_value('channels')
        window_size = parameter.get_value('windowSize')

        if window_size == 8:
            last_kernel_size = 1
        else:
            last_kernel_size = 3
        last_padding = 0 if last_kernel_size == 1 else 1

        # HSI branch
        self.hsi_conv1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(9, 3, 3), padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.hsi_conv2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(7, 3, 3), padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.hsi_conv3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(5, last_kernel_size, last_kernel_size), padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.hsi_conv4 = nn.Sequential(
            nn.Conv2d(32 * (channels - 18), 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # SAR/LiDAR branch
        self.sar_conv1 = nn.Sequential(
            nn.Conv2d(radar_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.sar_down1 = ConvDownsample(64, 128)
        self.sar_conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.sar_down2 = ConvDownsample(128, 256)
        self.sar_conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=last_kernel_size, padding=last_padding),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # MS branch
        self.ms_conv1 = nn.Sequential(
            nn.Conv2d(ms_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.ms_down1 = ConvDownsample(64, 128)
        self.ms_conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.ms_down2 = ConvDownsample(128, 256)
        self.ms_conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=last_kernel_size, padding=last_padding),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Generic feature-level fusion and classifier
        self.dropout = nn.Dropout(0.6)
        self.classifier = nn.Linear(256 * 3, out_features)

    @staticmethod
    def _flatten_hsi_3d(x):
        return x.reshape(x.size(0), x.size(1) * x.size(2), x.size(3), x.size(4))

    def forward(self, hsi, sar, ms):
        h = self.hsi_conv1(hsi)
        h = self.hsi_conv2(h)
        h = self.hsi_conv3(h)
        h = self.hsi_conv4(self._flatten_hsi_3d(h))

        s = self.sar_conv1(sar)
        s = self.sar_down1(s)
        s = self.sar_conv2(s)
        s = self.sar_down2(s)
        s = self.sar_conv3(s)

        m = self.ms_conv1(ms)
        m = self.ms_down1(m)
        m = self.ms_conv2(m)
        m = self.ms_down2(m)
        m = self.ms_conv3(m)

        hsi_feat = F.adaptive_avg_pool2d(h, 1).flatten(1)
        sar_feat = F.adaptive_avg_pool2d(s, 1).flatten(1)
        ms_feat = F.adaptive_avg_pool2d(m, 1).flatten(1)

        fused = torch.cat([hsi_feat, sar_feat, ms_feat], dim=1)
        logits = self.classifier(self.dropout(fused))
        return logits
