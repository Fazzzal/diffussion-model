import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_embedding_dim
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(
            num_groups=8,
            num_channels=in_channels
        )

        self.activation = nn.SiLU()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.time_projection = nn.Linear(
            time_embedding_dim,
            out_channels
        )

        self.norm2 = nn.GroupNorm(
            num_groups=8,
            num_channels=out_channels
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        if in_channels != out_channels:
            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x, time_embedding):
        residual = self.skip(x)

        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)

        time = self.time_projection(time_embedding)

        time = time.unsqueeze(-1).unsqueeze(-1)

        x = x + time

        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)

        return x + residual