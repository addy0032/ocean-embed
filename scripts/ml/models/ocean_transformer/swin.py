from typing import Sequence

import torch
from torch import nn

from torchvision.models.swin_transformer import (
    SwinTransformerBlock,
    PatchMerging,
)


class SwinStage(nn.Module):
    """
    A sequence of official TorchVision Swin Transformer blocks.

    Input/output layout:
        [B, H, W, C]
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        depth: int,
        window_size: int = 7,
        dropout: float = 0.1,
        stochastic_depth_prob: float = 0.1,
    ):
        super().__init__()

        blocks = []

        for i in range(depth):
            # Alternates:
            #
            # block 0: normal window attention
            # block 1: shifted window attention
            # block 2: normal
            # block 3: shifted
            #
            shift = 0 if i % 2 == 0 else window_size // 2

            # Linearly distribute stochastic depth across blocks.
            if depth == 1:
                sd_prob = stochastic_depth_prob
            else:
                sd_prob = stochastic_depth_prob * i / (depth - 1)

            blocks.append(
                SwinTransformerBlock(
                    dim=channels,
                    num_heads=num_heads,
                    window_size=[window_size, window_size],
                    shift_size=[shift, shift],
                    mlp_ratio=4.0,
                    dropout=dropout,
                    attention_dropout=dropout,
                    stochastic_depth_prob=sd_prob,
                    norm_layer=nn.LayerNorm,
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class SwinEncoder(nn.Module):
    """
    Hierarchical Swin encoder for ocean fields.

    Spatial progression for a 101 x 241 input:

        patch embedding:
            101 x 241
                ↓
            51 x 121

        stage 1:
            51 x 121, 64 channels

        patch merging:
            ~26 x 61

        stage 2:
            ~26 x 61, 128 channels

        patch merging:
            ~13 x 31

        stage 3:
            ~13 x 31, 256 channels

        bottleneck:
            ~13 x 31, 256 channels

    TorchVision internally handles odd spatial dimensions and
    pads/unpads windows where necessary.
    """

    def __init__(
        self,
        in_channels: int = 64,
        channels: Sequence[int] = (64, 128, 256),
        depths: Sequence[int] = (2, 2, 4),
        num_heads: Sequence[int] = (4, 8, 16),
        window_size: int = 7,
        dropout: float = 0.1,
        stochastic_depth_prob: float = 0.1,
    ):
        super().__init__()

        if not (
            len(channels)
            == len(depths)
            == len(num_heads)
            == 3
        ):
            raise ValueError(
                "channels, depths and num_heads must all have length 3"
            )

        # Initial patch embedding.
        #
        # 101 x 241
        #      ↓ kernel=2, stride=2
        # 51 x 121
        self.patch_embed = nn.Sequential(
            nn.Conv2d(
                in_channels,
                channels[0],
                kernel_size=2,
                stride=2,
            ),
            nn.BatchNorm2d(channels[0]),
        )

        self.stage1 = SwinStage(
            channels=channels[0],
            num_heads=num_heads[0],
            depth=depths[0],
            window_size=window_size,
            dropout=dropout,
            stochastic_depth_prob=stochastic_depth_prob,
        )

        self.merge1 = PatchMerging(
            dim=channels[0],
            norm_layer=nn.LayerNorm,
        )

        self.stage2 = SwinStage(
            channels=channels[1],
            num_heads=num_heads[1],
            depth=depths[1],
            window_size=window_size,
            dropout=dropout,
            stochastic_depth_prob=stochastic_depth_prob,
        )

        self.merge2 = PatchMerging(
            dim=channels[1],
            norm_layer=nn.LayerNorm,
        )

        self.stage3 = SwinStage(
            channels=channels[2],
            num_heads=num_heads[2],
            depth=depths[2],
            window_size=window_size,
            dropout=dropout,
            stochastic_depth_prob=stochastic_depth_prob,
        )

        # Additional bottleneck processing at 256 channels.
        self.bottleneck = SwinStage(
            channels=channels[2],
            num_heads=num_heads[2],
            depth=2,
            window_size=window_size,
            dropout=dropout,
            stochastic_depth_prob=stochastic_depth_prob,
        )

    def forward(self, x: torch.Tensor):
        """
        Input:
            [B, C, H, W]

        Returns:
            bottleneck
            skip2
            skip1
        """

        # ---------------------------------------------------------
        # Patch embedding
        # ---------------------------------------------------------

        x = self.patch_embed(x)

        # Convert:
        # [B, C, H, W]
        # →
        # [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()

        # ---------------------------------------------------------
        # Stage 1
        # ---------------------------------------------------------

        skip1 = self.stage1(x)

        # Downsample 64 → 128.
        x = self.merge1(skip1)

        # ---------------------------------------------------------
        # Stage 2
        # ---------------------------------------------------------

        skip2 = self.stage2(x)

        # Downsample 128 → 256.
        x = self.merge2(skip2)

        # ---------------------------------------------------------
        # Stage 3
        # ---------------------------------------------------------

        x = self.stage3(x)

        # ---------------------------------------------------------
        # Bottleneck
        # ---------------------------------------------------------

        x = self.bottleneck(x)

        # Return in [B,C,H,W] format.
        bottleneck = x.permute(
            0, 3, 1, 2
        ).contiguous()

        skip2 = skip2.permute(
            0, 3, 1, 2
        ).contiguous()

        skip1 = skip1.permute(
            0, 3, 1, 2
        ).contiguous()

        return bottleneck, skip2, skip1