import torch
import torch.nn as nn


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=4
    ):
        super().__init__()

        self.norm = nn.GroupNorm(
            num_groups=8,
            num_channels=channels
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

        self.output = nn.Linear(
            channels,
            channels
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        residual = x

        x = self.norm(x)

        x = x.reshape(
            batch_size,
            channels,
            height * width
        )

        x = x.permute(
            0,
            2,
            1
        )

        attention_output, _ = self.attention(
            x,
            x,
            x,
            need_weights=False
        )

        attention_output = self.output(
            attention_output
        )

        attention_output = attention_output.permute(
            0,
            2,
            1
        )

        attention_output = attention_output.reshape(
            batch_size,
            channels,
            height,
            width
        )

        return residual + attention_output