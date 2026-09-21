import torch
import torch.nn as nn

from diffusion.blocks import ResidualBlock
from diffusion.embeddings import SinusoidalTimeEmbedding


class DiffusionUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=32,
        time_embedding_dim=128
    ):
        super().__init__()

        self.time_embedding = SinusoidalTimeEmbedding(
            time_embedding_dim
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(
                time_embedding_dim,
                time_embedding_dim
            ),
            nn.SiLU(),
            nn.Linear(
                time_embedding_dim,
                time_embedding_dim
            )
        )

        self.input_conv = nn.Conv2d(
            in_channels,
            base_channels,
            kernel_size=3,
            padding=1
        )

        self.down_block = ResidualBlock(
            base_channels,
            base_channels,
            time_embedding_dim
        )

        self.downsample = nn.Conv2d(
            base_channels,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.middle_block = ResidualBlock(
            base_channels * 2,
            base_channels * 2,
            time_embedding_dim
        )

        self.upsample = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.up_block = ResidualBlock(
            base_channels * 2,
            base_channels,
            time_embedding_dim
        )

        self.output_norm = nn.GroupNorm(
            num_groups=8,
            num_channels=base_channels
        )

        self.output_activation = nn.SiLU()

        self.output_conv = nn.Conv2d(
            base_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x, timesteps):
        time_embedding = self.time_embedding(
            timesteps
        )

        time_embedding = self.time_mlp(
            time_embedding
        )

        x = self.input_conv(x)

        skip = self.down_block(
            x,
            time_embedding
        )

        x = self.downsample(skip)

        x = self.middle_block(
            x,
            time_embedding
        )

        x = self.upsample(x)

        x = torch.cat(
            [x, skip],
            dim=1
        )

        x = self.up_block(
            x,
            time_embedding
        )

        x = self.output_norm(x)
        x = self.output_activation(x)

        return self.output_conv(x)