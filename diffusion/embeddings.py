import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        half_dim = self.embedding_dim // 2

        scale = math.log(10000) / (half_dim - 1)

        frequencies = torch.exp(
            torch.arange(
                half_dim,
                device=timesteps.device
            ) * -scale
        )

        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)

        embedding = torch.cat(
            [
                torch.sin(angles),
                torch.cos(angles)
            ],
            dim=1
        )

        return embedding