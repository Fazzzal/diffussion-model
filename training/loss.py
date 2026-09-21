import torch.nn.functional as F


def diffusion_loss(predicted_noise, actual_noise):
    return F.mse_loss(
        predicted_noise,
        actual_noise
    )