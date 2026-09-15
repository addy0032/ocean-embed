"""
Temporal U-Net baseline for OceanEmbed.

Input:
    [B, T, C, H, W]

    B = batch
    T = number of days
    C = 7 surface variables
    H = 101 latitude points
    W = 241 longitude points

Output:
    [B, 15, H, W]

    15 channels correspond to the 15 target depths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """
    3D convolutional block used for temporal-spatial feature extraction.

    Input:
        [B, C, T, H, W]

    Output:
        [B, out_channels, T, H, W]
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvBlock2D(nn.Module):
    """
    Standard 2D convolutional block for the spatial U-Net.

    Input:
        [B, C, H, W]

    Output:
        [B, out_channels, H, W]
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TemporalEncoder(nn.Module):
    """
    Extract temporal-spatial features from the 7-day input sequence.

    Input:
        [B, 7, T, H, W]

    Output:
        [B, base_channels, H, W]

    The temporal dimension is collapsed using adaptive
    average pooling after the 3D convolutions.
    """

    def __init__(self, in_channels, base_channels):
        super().__init__()

        self.conv1 = ConvBlock3D(
            in_channels,
            base_channels,
        )

        self.conv2 = ConvBlock3D(
            base_channels,
            base_channels,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # Collapse temporal dimension:
        #
        # [B, C, T, H, W]
        #       ↓
        # [B, C, H, W]
        x = x.mean(dim=2)

        return x


class TemporalUNet(nn.Module):
    """
    Baseline Temporal U-Net.

    Input:
        [B, T, 7, H, W]

    Output:
        [B, 15, H, W]
    """

    def __init__(
        self,
        in_channels=7,
        out_channels=15,
        base_channels=32,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        # --------------------------------------------------------
        # Temporal feature extraction.
        #
        # Input after permutation:
        # [B, 7, T, H, W]
        # --------------------------------------------------------
        self.temporal_encoder = TemporalEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
        )

        # --------------------------------------------------------
        # Spatial U-Net encoder.
        # --------------------------------------------------------
        self.enc1 = ConvBlock2D(
            base_channels,
            base_channels,
        )

        self.enc2 = ConvBlock2D(
            base_channels,
            base_channels * 2,
        )

        self.enc3 = ConvBlock2D(
            base_channels * 2,
            base_channels * 4,
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # --------------------------------------------------------
        # Bottleneck.
        # --------------------------------------------------------
        self.bottleneck = ConvBlock2D(
            base_channels * 4,
            base_channels * 8,
        )

        # --------------------------------------------------------
        # Decoder.
        # --------------------------------------------------------
        self.dec3 = ConvBlock2D(
            base_channels * 8 + base_channels * 4,
            base_channels * 4,
        )

        self.dec2 = ConvBlock2D(
            base_channels * 4 + base_channels * 2,
            base_channels * 2,
        )

        self.dec1 = ConvBlock2D(
            base_channels * 2 + base_channels,
            base_channels,
        )

        # --------------------------------------------------------
        # Final prediction layer.
        #
        # One channel per target depth.
        # --------------------------------------------------------
        self.output = nn.Conv2d(
            base_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape [B, T, 7, H, W]

        Returns
        -------
        torch.Tensor
            Shape [B, 15, H, W]
        """

        if x.ndim != 5:
            raise ValueError(
                f"Expected 5D input [B,T,C,H,W], "
                f"got shape {tuple(x.shape)}"
            )

        if x.shape[2] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, "
                f"got {x.shape[2]}"
            )

        # --------------------------------------------------------
        # PyTorch Conv3d expects:
        #
        # [B, C, T, H, W]
        #
        # Dataset provides:
        #
        # [B, T, C, H, W]
        # --------------------------------------------------------
        x = x.permute(0, 2, 1, 3, 4)

        # --------------------------------------------------------
        # Temporal encoder.
        #
        # [B, 7, T, H, W]
        #       ↓
        # [B, 32, H, W]
        # --------------------------------------------------------
        x = self.temporal_encoder(x)

        # --------------------------------------------------------
        # U-Net encoder.
        # --------------------------------------------------------
        e1 = self.enc1(x)

        p1 = self.pool(e1)

        e2 = self.enc2(p1)

        p2 = self.pool(e2)

        e3 = self.enc3(p2)

        p3 = self.pool(e3)

        # --------------------------------------------------------
        # Bottleneck.
        # --------------------------------------------------------
        b = self.bottleneck(p3)

        # --------------------------------------------------------
        # Decoder level 3.
        #
        # Upsample to e3 spatial dimensions explicitly.
        # This handles odd spatial dimensions safely.
        # --------------------------------------------------------
        d3 = F.interpolate(
            b,
            size=e3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        d3 = torch.cat(
            [d3, e3],
            dim=1,
        )

        d3 = self.dec3(d3)

        # --------------------------------------------------------
        # Decoder level 2.
        # --------------------------------------------------------
        d2 = F.interpolate(
            d3,
            size=e2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        d2 = torch.cat(
            [d2, e2],
            dim=1,
        )

        d2 = self.dec2(d2)

        # --------------------------------------------------------
        # Decoder level 1.
        # --------------------------------------------------------
        d1 = F.interpolate(
            d2,
            size=e1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        d1 = torch.cat(
            [d1, e1],
            dim=1,
        )

        d1 = self.dec1(d1)

        # --------------------------------------------------------
        # Final output.
        #
        # [B, 32, H, W]
        #       ↓
        # [B, 15, H, W]
        # --------------------------------------------------------
        return self.output(d1)