import torch
from torch import nn

from .positional import CoordinateFeatures
from .temporal_attention import TemporalAttention
from .swin import SwinEncoder
from .decoder import OceanDecoder


class OceanTransformer(nn.Module):
    """
    Causal spatiotemporal CNN-Swin Transformer U-Net.

    Input:
        [B, T=7, C=7, H=101, W=241]

    Output:
        [B, 15, H=101, W=241]

    Pipeline:

        Physical variables
              +
        latitude / longitude
              ↓
        per-day CNN embedding
              ↓
        causal temporal attention
              ↓
        hierarchical Swin encoder
              ↓
        U-Net decoder
              ↓
        1x1 output projection
              ↓
        15 depth levels
    """

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 15,
        temporal_length: int = 7,
        feature_channels: int = 64,
        temporal_heads: int = 8,
        swin_channels: tuple[int, int, int] = (
            64,
            128,
            256,
        ),
        swin_depths: tuple[int, int, int] = (
            2,
            2,
            4,
        ),
        swin_heads: tuple[int, int, int] = (
            4,
            8,
            16,
        ),
        window_size: int = 7,
        dropout: float = 0.1,
        stochastic_depth_prob: float = 0.1,
        use_temporal_attention: bool = True,
        use_coordinate_features: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temporal_length = temporal_length

        self.use_temporal_attention = (
            use_temporal_attention
        )

        self.use_coordinate_features = (
            use_coordinate_features
        )

        # ---------------------------------------------------------
        # Coordinate features
        # ---------------------------------------------------------

        if use_coordinate_features:
            self.coordinate_features = CoordinateFeatures()
            spatial_input_channels = in_channels + 2
        else:
            self.coordinate_features = None
            spatial_input_channels = in_channels

        # ---------------------------------------------------------
        # Per-day feature embedding
        #
        # Each day's 7 physical variables (+ 2 coordinates)
        # are converted into a 64-channel representation.
        # ---------------------------------------------------------

        self.input_projection = nn.Sequential(
            nn.Conv2d(
                spatial_input_channels,
                feature_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(feature_channels),
            nn.GELU(),

            nn.Conv2d(
                feature_channels,
                feature_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(feature_channels),
            nn.GELU(),
        )

        # ---------------------------------------------------------
        # Temporal processing
        # ---------------------------------------------------------

        if use_temporal_attention:
            self.temporal_attention = TemporalAttention(
                channels=feature_channels,
                num_heads=temporal_heads,
                dropout=dropout,
                max_time=temporal_length,
            )
        else:
            self.temporal_attention = None

        # ---------------------------------------------------------
        # Swin encoder
        # ---------------------------------------------------------

        if swin_channels[0] != feature_channels:
            raise ValueError(
                "The first Swin channel count must equal "
                "feature_channels."
            )

        self.encoder = SwinEncoder(
            in_channels=feature_channels,
            channels=swin_channels,
            depths=swin_depths,
            num_heads=swin_heads,
            window_size=window_size,
            dropout=dropout,
            stochastic_depth_prob=stochastic_depth_prob,
        )

        # ---------------------------------------------------------
        # U-Net decoder
        # ---------------------------------------------------------

        self.decoder = OceanDecoder(
            dropout=dropout,
        )

        # ---------------------------------------------------------
        # Final depth prediction head
        #
        # 64 feature maps → 15 depth maps.
        #
        # This is a 1x1 convolution, so each pixel gets its
        # own learned mapping from the 64 learned features
        # to the 15 output depth levels.
        # ---------------------------------------------------------

        self.output_head = nn.Conv2d(
            64,
            out_channels,
            kernel_size=1,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize the CNN/output layers.

        Swin blocks contain their own initialization logic.
        """

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _encode_each_day(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the CNN input projection independently to each day.

        Input:
            [B, T, C, H, W]

        Output:
            [B, T, 64, H, W]
        """

        b, t, c, h, w = x.shape

        # Flatten the temporal dimension into batch.
        x = x.reshape(
            b * t,
            c,
            h,
            w,
        )

        x = self.input_projection(x)

        x = x.reshape(
            b,
            t,
            -1,
            h,
            w,
        )

        return x

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 5:
            raise ValueError(
                "Expected input shape [B,T,C,H,W], "
                f"got {tuple(x.shape)}"
            )

        b, t, c, h, w = x.shape

        if t != self.temporal_length:
            raise ValueError(
                f"Expected T={self.temporal_length}, "
                f"got T={t}"
            )

        if c != self.in_channels:
            raise ValueError(
                f"Expected C={self.in_channels}, "
                f"got C={c}"
            )

        # ---------------------------------------------------------
        # Add static coordinate information.
        #
        # Coordinates are identical for every temporal slice.
        # ---------------------------------------------------------

        if self.use_coordinate_features:
            # Coordinates based on spatial dimensions.
            coords = self.coordinate_features(
                x[:, 0]
            )

            coords = coords.unsqueeze(1).expand(
                -1,
                t,
                -1,
                -1,
                -1,
            )

            x = torch.cat(
                [x, coords],
                dim=2,
            )

        # ---------------------------------------------------------
        # Per-day CNN embedding
        # ---------------------------------------------------------

        x = self._encode_each_day(x)

        # ---------------------------------------------------------
        # Temporal processing
        # ---------------------------------------------------------

        if self.use_temporal_attention:
            x = self.temporal_attention(x)

        else:
            # Ablation/baseline:
            # reproduce the old temporal-collapse idea.
            x = x.mean(dim=1)

        # x:
        # [B, 64, H, W]
        # ---------------------------------------------------------

        # ---------------------------------------------------------
        # Swin encoder
        # ---------------------------------------------------------

        bottleneck, skip2, skip1 = self.encoder(x)

        # ---------------------------------------------------------
        # U-Net decoder
        # ---------------------------------------------------------

        decoded = self.decoder(
            bottleneck=bottleneck,
            skip2=skip2,
            skip1=skip1,
            output_size=(h, w),
        )

        # ---------------------------------------------------------
        # Predict 15 standard depth levels
        # ---------------------------------------------------------

        output = self.output_head(decoded)

        return output