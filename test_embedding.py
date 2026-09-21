import torch

from diffusion.embeddings import SinusoidalTimeEmbedding


embedding = SinusoidalTimeEmbedding(128)

timesteps = torch.tensor([0, 100, 500, 999])

output = embedding(timesteps)

print("Input shape:", timesteps.shape)
print("Output shape:", output.shape)
print("Embedding for t=0:")
print(output[0])