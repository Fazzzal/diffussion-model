import matplotlib.pyplot as plt
import torch
from pathlib import Path

from diffusion.scheduler import DiffusionScheduler
from diffusion.unet import UNet
from diffusion.sampler import sample


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    project_root = Path(__file__).resolve().parent.parent

    checkpoint_path = (
        project_root
        / "checkpoints"
        / "cifar10_attention_unet.pth"
    )

    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        time_embedding_dim=128
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    scheduler = DiffusionScheduler(
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        device=device
    )

    generated_images, trajectory = sample(
        model=model,
        scheduler=scheduler,
        num_samples=1,
        image_size=32,
        channels=3,
        device=device
    )

    fig, axes = plt.subplots(
        1,
        len(trajectory),
        figsize=(15, 3)
    )

    for ax, (timestep, images) in zip(
        axes,
        trajectory
    ):
        image = images[0].cpu()

        image = (
            image.clamp(-1, 1) + 1
        ) / 2

        image = image.permute(
            1,
            2,
            0
        )

        ax.imshow(image)

        ax.set_title(
            f"t = {timestep}"
        )

        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()