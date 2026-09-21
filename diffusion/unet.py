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

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

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
            c1,
            kernel_size=3,
            padding=1
        )

        self.down1 = ResidualBlock(
            c1,
            c1,
            time_embedding_dim
        )

        self.downsample1 = nn.Conv2d(
            c1,
            c2,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.down2 = ResidualBlock(
            c2,
            c2,
            time_embedding_dim
        )

        self.attention2 = SelfAttentionBlock(
            c2
        )

        self.downsample2 = nn.Conv2d(
            c2,
            c3,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.down3 = ResidualBlock(
            c3,
            c3,
            time_embedding_dim
        )

        self.attention3 = SelfAttentionBlock(
            c3
        )

        self.downsample3 = nn.Conv2d(
            c3,
            c4,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.middle1 = ResidualBlock(
            c4,
            c4,
            time_embedding_dim
        )

        self.middle_attention = SelfAttentionBlock(
            c4
        )

        self.middle2 = ResidualBlock(
            c4,
            c4,
            time_embedding_dim
        )

        self.upsample3 = nn.ConvTranspose2d(
            c4,
            c3,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.up3 = ResidualBlock(
            c3 * 2,
            c3,
            time_embedding_dim
        )

        self.attention_up3 = SelfAttentionBlock(
            c3
        )

        self.upsample2 = nn.ConvTranspose2d(
            c3,
            c2,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.up2 = ResidualBlock(
            c2 * 2,
            c2,
            time_embedding_dim
        )

        self.attention_up2 = SelfAttentionBlock(
            c2
        )

        self.upsample1 = nn.ConvTranspose2d(
            c2,
            c1,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.up1 = ResidualBlock(
            c1 * 2,
            c1,
            time_embedding_dim
        )

        self.output_norm = nn.GroupNorm(
            num_groups=8,
            num_channels=c1
        )

        self.output_activation = nn.SiLU()

        self.output_conv = nn.Conv2d(
            c1,
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

        skip1 = self.down1(
            x,
            time_embedding
        )

        x = self.downsample1(
            skip1
        )

        skip2 = self.down2(
            x,
            time_embedding
        )

        skip2 = self.attention2(
            skip2
        )

        x = self.downsample2(
            skip2
        )

        skip3 = self.down3(
            x,
            time_embedding
        )

        skip3 = self.attention3(
            skip3
        )

        x = self.downsample3(
            skip3
        )

        x = self.middle1(
            x,
            time_embedding
        )

        x = self.middle_attention(x)

        x = self.middle2(
            x,
            time_embedding
        )

        x = self.upsample3(x)

        x = torch.cat(
            [x, skip3],
            dim=1
        )

        x = self.up3(
            x,
            time_embedding
        )

        x = self.attention_up3(x)

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