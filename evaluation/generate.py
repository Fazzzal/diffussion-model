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

    sample_dir = (
        project_root
        / "samples"
    )

    sample_dir.mkdir(
        parents=True,
        exist_ok=True
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

    generated_images, _ = sample(
        model=model,
        scheduler=scheduler,
        num_samples=16,
        image_size=32,
        channels=3,
        device=device
    )

    generated_images = (
        generated_images.clamp(-1, 1) + 1
    ) / 2

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(8, 8)
    )

    for ax, image in zip(
        axes.flatten(),
        generated_images
    ):
        image = image.cpu().permute(
            1,
            2,
            0
        )

        ax.imshow(image)
        ax.axis("off")

    plt.tight_layout()

    output_path = (
        sample_dir
        / "cifar10_generated_grid.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"Generated images saved to: {output_path}"
    )


if __name__ == "__main__":
    main()