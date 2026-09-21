import torch

from diffusion.blocks import ResidualBlock


batch_size = 4
in_channels = 32
out_channels = 64
height = 28
width = 28
time_embedding_dim = 128

block = ResidualBlock(
    in_channels=in_channels,
    out_channels=out_channels,
    time_embedding_dim=time_embedding_dim
)

x = torch.randn(
    batch_size,
    in_channels,
    height,
    width
)

time_embedding = torch.randn(
    batch_size,
    time_embedding_dim
)

output = block(
    x,
    time_embedding
)

print("Input shape:", x.shape)
print("Time embedding shape:", time_embedding.shape)
print("Output shape:", output.shape)