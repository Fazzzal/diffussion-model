import torch


class DiffusionScheduler:
    def __init__(
        self,
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device="cpu"
    ):
        self.num_timesteps = num_timesteps
        self.device = device

        self.betas = torch.linspace(
            beta_start,
            beta_end,
            num_timesteps,
            device=device
        )

        self.alphas = 1.0 - self.betas

        self.alpha_bars = torch.cumprod(
            self.alphas,
            dim=0
        )

        self.sqrt_alpha_bars = torch.sqrt(
            self.alpha_bars
        )

        self.sqrt_one_minus_alpha_bars = torch.sqrt(
            1.0 - self.alpha_bars
        )

        self.sqrt_reciprocal_alphas = torch.sqrt(
            1.0 / self.alphas
        )

        self.posterior_variance = (
            self.betas
            * (1.0 - torch.cat([
                torch.ones(1, device=device),
                self.alpha_bars[:-1]
            ]))
            / (1.0 - self.alpha_bars)
        )

        self.posterior_variance[0] = 0.0

    def add_noise(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar = self.sqrt_alpha_bars[t]
        sqrt_one_minus_alpha_bar = (
            self.sqrt_one_minus_alpha_bars[t]
        )

        sqrt_alpha_bar = sqrt_alpha_bar.reshape(
            -1, 1, 1, 1
        )

        sqrt_one_minus_alpha_bar = (
            sqrt_one_minus_alpha_bar.reshape(
                -1, 1, 1, 1
            )
        )

        x_t = (
            sqrt_alpha_bar * x_0
            + sqrt_one_minus_alpha_bar * noise
        )

        return x_t, noise

    def step(
        self,
        predicted_noise,
        timestep,
        x_t
    ):
        beta_t = self.betas[timestep]
        alpha_t = self.alphas[timestep]
        alpha_bar_t = self.alpha_bars[timestep]

        sqrt_reciprocal_alpha = (
            self.sqrt_reciprocal_alphas[timestep]
        )

        predicted_mean = sqrt_reciprocal_alpha * (
            x_t
            - (
                beta_t
                / torch.sqrt(1.0 - alpha_bar_t)
            ) * predicted_noise
        )

        if timestep == 0:
            return predicted_mean

        variance = self.posterior_variance[timestep]

        noise = torch.randn_like(x_t)

        return (
            predicted_mean
            + torch.sqrt(variance) * noise
        )