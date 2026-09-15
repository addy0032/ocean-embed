"""
Masked losses for OceanEmbed.

The target contains invalid cells where either:

    1. the surface master mask is invalid, or
    2. GLORYS does not provide a valid target at that depth.

The Dataset combines those two masks and returns:

    mask: [B, 15, H, W]

The loss is computed only over mask == True.
"""

import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """
    Mean squared error over valid target cells only.

    Parameters
    ----------
    prediction : torch.Tensor
        Model output.
        Shape: [B, D, H, W]

    target : torch.Tensor
        Normalized target.
        Shape: [B, D, H, W]

    mask : torch.Tensor
        Boolean validity mask.
        Shape: [B, D, H, W]

    Returns
    -------
    torch.Tensor
        Scalar masked MSE loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, prediction, target, mask):

        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target shapes do not match.\n"
                f"Prediction: {tuple(prediction.shape)}\n"
                f"Target:     {tuple(target.shape)}"
            )

        if mask.shape != prediction.shape:
            raise ValueError(
                "Mask and prediction shapes do not match.\n"
                f"Mask:       {tuple(mask.shape)}\n"
                f"Prediction: {tuple(prediction.shape)}"
            )

        if mask.dtype != torch.bool:
            raise TypeError(f"Expected boolean mask, got {mask.dtype}")

        # --------------------------------------------------------
        # Compute the loss in float32 for numerical stability.
        #
        # The model can run in float16 under AMP, but the final
        # reduction over many spatial/depth cells should remain
        # float32.
        # --------------------------------------------------------
        prediction_float = prediction.float()
        target_float = target.float()
        mask_float = mask.to(
            device=prediction.device,
            dtype=torch.float32,
        )

        squared_error = (prediction_float - target_float) ** 2

        masked_error = squared_error * mask_float

        valid_count = mask_float.sum()

        if valid_count.item() == 0:
            raise ValueError("MaskedMSELoss received a mask with zero valid cells.")

        loss = masked_error.sum() / valid_count

        return loss