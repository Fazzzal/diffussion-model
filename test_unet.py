import torch

from diffusion.unet import DiffusionUNet


model = DiffusionUNet()

x = torch.randn(
    4,
    1,
    28,
    28
)

timesteps = torch.tensor(
    [0, 100, 500, 999]
)

output = model(
    x,
    timesteps
)

print("Input shape:", x.shape)
print("Timestep shape:", timesteps.shape)
print("Output shape:", output.shape)

print(
    "Number of parameters:",
    sum(p.numel() for p in model.parameters())
)