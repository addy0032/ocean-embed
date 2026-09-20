import torch
from torch import nn


class CoordinateFeatures(nn.Module):
    """
    Creates static latitude/longitude coordinate features.

    Input:
        x: [B, C, H, W]

    Output:
        coords: [B, 2, H, W]
            channel 0 = normalized latitude
            channel 1 = normalized longitude
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        device = x.device
        dtype = x.dtype

        # Normalized geographic coordinates:
        # latitude  : [-1, 1]
        # longitude : [-1, 1]
        lat = torch.linspace(
            -1.0, 1.0, h, device=device, dtype=dtype
        ).view(1, 1, h, 1)

        lon = torch.linspace(
            -1.0, 1.0, w, device=device, dtype=dtype
        ).view(1, 1, 1, w)

        lat = lat.expand(x.shape[0], -1, -1, w)
        lon = lon.expand(x.shape[0], -1, h, -1)

        return torch.cat([lat, lon], dim=1)