import torch
import torch.nn.functional as F
from torch import nn


class DecoderBlock(nn.Module):
    """
    Upsample + skip fusion + convolutional refinement.

    The decoder operates in standard [B,C,H,W] format.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),

            nn.Dropout2d(dropout),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:

        # Match the skip connection's exact spatial dimensions.
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat([x, skip], dim=1)

        return self.conv(x)


class OceanDecoder(nn.Module):
    """
    U-Net style decoder.

    Encoder:

        bottleneck: 256 channels, ~13 x 31
        skip2:      128 channels, ~26 x 61
        skip1:       64 channels, ~51 x 121

    Decoder:

        256 -> 128
        128 -> 64
        64  -> 64

    Finally interpolate back to the original 101 x 241 grid.
    """

    def __init__(
        self,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.decode2 = DecoderBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128,
            dropout=dropout,
        )

        self.decode1 = DecoderBlock(
            in_channels=128,
            skip_channels=64,
            out_channels=64,
            dropout=dropout,
        )

        # Final refinement after returning to full resolution.
        self.final_refine = nn.Sequential(
            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
        )

    def forward(
        self,
        bottleneck: torch.Tensor,
        skip2: torch.Tensor,
        skip1: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:

        x = self.decode2(
            bottleneck,
            skip2,
        )

        x = self.decode1(
            x,
            skip1,
        )

        # 51 x 121 → 101 x 241
        x = F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        x = self.final_refine(x)

        return x