import matplotlib.pyplot as plt
import torch
from pathlib import Path

from diffusion.scheduler import DiffusionScheduler
from diffusion.unet import DiffusionUNet
from diffusion.sampler import sample


def main():
    device = torch.device("cpu")

    model = DiffusionUNet().to(device)

    project_root = Path(__file__).resolve().parent.parent

    checkpoint_path = (
        project_root
        / "checkpoints"
        / "diffusion_mnist.pth"
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device
        )
    )

    scheduler = DiffusionScheduler(
        device=device
    )

    generated_images, trajectory = sample(
        model=model,
        scheduler=scheduler,
        num_samples=1,
        image_size=28,
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
        image = images[0, 0].cpu()

        image = (image.clamp(-1, 1) + 1) / 2

        ax.imshow(
            image,
            cmap="gray"
        )

        ax.set_title(f"t = {timestep}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()