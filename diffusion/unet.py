import torch
import torch.nn as nn

from diffusion.embeddings import SinusoidalTimeEmbedding
from diffusion.blocks import ResidualBlock
from diffusion.attention import SelfAttentionBlock


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=32,
        time_embedding_dim=128
    ):
        super().__init__()

        self.time_embedding = SinusoidalTimeEmbedding(
            time_embedding_dim
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )

        self.input_conv = nn.Conv2d(
            in_channels,
            base_channels,
            kernel_size=3,
            padding=1
        )

        self.down1 = ResidualBlock(
            base_channels,
            base_channels,
            time_embedding_dim
        )

        self.downsample1 = nn.Conv2d(
            base_channels,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.down2 = ResidualBlock(
            base_channels * 2,
            base_channels * 2,
            time_embedding_dim
        )

        self.attention2 = SelfAttentionBlock(
            base_channels * 2
        )

        self.downsample2 = nn.Conv2d(
            base_channels * 2,
            base_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.middle1 = ResidualBlock(
            base_channels * 4,
            base_channels * 4,
            time_embedding_dim
        )

        self.middle_attention = SelfAttentionBlock(
            base_channels * 4
        )

        self.middle2 = ResidualBlock(
            base_channels * 4,
            base_channels * 4,
            time_embedding_dim
        )

        self.upsample2 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.up2 = ResidualBlock(
            base_channels * 4,
            base_channels * 2,
            time_embedding_dim
        )

        self.attention_up2 = SelfAttentionBlock(
            base_channels * 2
        )

        self.upsample1 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.up1 = ResidualBlock(
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
        time_embedding = self.time_embedding(timesteps)
        time_embedding = self.time_mlp(time_embedding)

        x = self.input_conv(x)

        skip1 = self.down1(
            x,
            time_embedding
        )

        x = self.downsample1(skip1)

        skip2 = self.down2(
            x,
            time_embedding
        )

        x = self.attention2(skip2)

        x = self.downsample2(x)

        x = self.middle1(
            x,
            time_embedding
        )

        x = self.middle_attention(x)

        x = self.middle2(
            x,
            time_embedding
        )

        x = self.upsample2(x)

        x = torch.cat(
            [x, skip2],
            dim=1
        )

        x = self.up2(
            x,
            time_embedding
        )

        x = self.attention_up2(x)

        x = self.upsample1(x)

        x = torch.cat(
            [x, skip1],
            dim=1
        )

        x = self.up1(
            x,
            time_embedding
        )

        x = self.output_norm(x)
        x = self.output_activation(x)
        x = self.output_conv(x)

        return x