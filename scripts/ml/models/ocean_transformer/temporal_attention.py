import torch
from torch import nn


class TemporalAttention(nn.Module):
    """
    Temporal self-attention over the T-day input sequence.

    Input:
        x: [B, T, C, H, W]

    Output:
        x: [B, C, H, W]

    The representation from the final day is returned.
    Because attention is causal, the final day's representation
    can use t-6 ... t, but never any future timestep.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_time: int = 7,
    ):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.channels = channels
        self.max_time = max_time

        self.norm = nn.LayerNorm(channels)

        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Learnable temporal position embeddings.
        # Shape: [1, T, C]
        self.temporal_embedding = nn.Parameter(
            torch.zeros(1, max_time, channels)
        )

        nn.init.trunc_normal_(
            self.temporal_embedding,
            std=0.02,
        )

        self.dropout = nn.Dropout(dropout)

        self.output_norm = nn.LayerNorm(channels)

    def _causal_mask(
        self,
        t: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        True means that attention is blocked.

        Result:
            [T, T]

        Example for T=4:

        [[False, True,  True,  True ],
         [False, False, True,  True ],
         [False, False, False, True ],
         [False, False, False, False]]
        """

        return torch.triu(
            torch.ones(t, t, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"Expected [B, T, C, H, W], got {tuple(x.shape)}"
            )

        b, t, c, h, w = x.shape

        if c != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {c}"
            )

        if t > self.max_time:
            raise ValueError(
                f"Sequence length {t} exceeds max_time={self.max_time}"
            )

        # ---------------------------------------------------------
        # Convert:
        #
        # [B, T, C, H, W]
        #       ↓
        # [B, H, W, T, C]
        #       ↓
        # [B*H*W, T, C]
        #
        # Each spatial location has its own temporal sequence.
        # ---------------------------------------------------------

        x = x.permute(0, 3, 4, 1, 2).contiguous()
        x = x.reshape(b * h * w, t, c)

        # ---------------------------------------------------------
        # Add temporal position information.
        # ---------------------------------------------------------

        x = x + self.temporal_embedding[:, :t]

        # ---------------------------------------------------------
        # Causal mask.
        #
        # True = attention is blocked.
        #
        # Therefore:
        #
        # t0 -> t0
        # t1 -> t0,t1
        # t2 -> t0,t1,t2
        # ...
        # t6 -> t0,...,t6
        # ---------------------------------------------------------

        causal_mask = self._causal_mask(
            t=t,
            device=x.device,
        )

        # CUDA's efficient attention kernel has a maximum
        # attention-batch size of 65535.
        #
        # After flattening, x has:
        #
        #     B * H * W
        #
        # independent temporal sequences.
        #
        # For B=4:
        #
        #     4 * 101 * 241 = 97,364
        #
        # which exceeds 65,535.
        #
        # We therefore process at most 16,000 sequences per
        # attention call. This is independent of the physical
        # training batch size because B has already been folded
        # into the flattened sequence dimension above.
        #
        # Chunking does not change the mathematical operation:
        # temporal attention is performed independently for every
        # spatial location.
        
        max_spatial_sequences = 32_000

        outputs = []

        for start in range(
            0,
            x.shape[0],
            max_spatial_sequences,
        ):
            end = min(
                start + max_spatial_sequences,
                x.shape[0],
            )

            x_chunk = x[start:end]

            # -----------------------------------------------------
            # Pre-normalization
            # -----------------------------------------------------

            residual = x_chunk

            x_norm = self.norm(x_chunk)

            # -----------------------------------------------------
            # Causal temporal self-attention
            # -----------------------------------------------------

            attended, _ = self.attention(
                x_norm,
                x_norm,
                x_norm,
                attn_mask=causal_mask,
                need_weights=False,
            )

            # -----------------------------------------------------
            # Residual connection
            # -----------------------------------------------------

            x_chunk = (
                residual
                + self.dropout(attended)
            )

            x_chunk = self.output_norm(x_chunk)

            outputs.append(x_chunk)

        # Reassemble all spatial sequences.
        x = torch.cat(outputs, dim=0)

        # ---------------------------------------------------------
        # We only need the representation of the FINAL day.
        #
        # Because attention is causal, this representation contains
        # information from:
        #
        # t-6, t-5, ..., t
        #
        # but never any future timestep.
        # ---------------------------------------------------------

        x = x[:, -1]

        # ---------------------------------------------------------
        # Restore spatial layout:
        #
        # [B*H*W, C]
        #       ↓
        # [B, H, W, C]
        #       ↓
        # [B, C, H, W]
        # ---------------------------------------------------------

        x = x.reshape(
            b,
            h,
            w,
            c,
        )

        x = x.permute(
            0,
            3,
            1,
            2,
        ).contiguous()

        return x