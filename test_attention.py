import torch

from diffusion.attention import SelfAttentionBlock


attention = SelfAttentionBlock(
    channels=64,
    num_heads=4
)

x = torch.randn(
    4,
    64,
    14,
    14
)

output = attention(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

print(
    "Number of parameters:",
    sum(
        parameter.numel()
        for parameter in attention.parameters()
    )
)