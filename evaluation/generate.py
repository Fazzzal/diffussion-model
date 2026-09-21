import matplotlib.pyplot as plt
import torch
from pathlib import Path

from diffusion.scheduler import DiffusionScheduler
from diffusion.unet import UNet
from diffusion.sampler import sample


def load_model(checkpoint, device):
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        time_embedding_dim=128
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return model


def load_ema_model(checkpoint, device):
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        time_embedding_dim=128
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in checkpoint["ema_state_dict"]:
                parameter.copy_(
                    checkpoint["ema_state_dict"][name]
                )

    return model


def generate_images(
    model,
    scheduler,
    device,
    num_samples=16
):
    images, _ = sample(
        model=model,
        scheduler=scheduler,
        num_samples=num_samples,
        image_size=32,
        channels=3,
        device=device
    )

    return (
        images.clamp(-1, 1) + 1
    ) / 2


def plot_grid(images, title, output_path):
    fig, axes = plt.subplots(
        4,
        4,
        figsize=(8, 8)
    )

    for ax, image in zip(
        axes.flatten(),
        images
    ):
        image = image.cpu().permute(
            1,
            2,
            0
        )

        ax.imshow(image)
        ax.axis("off")

    fig.suptitle(
        title,
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)


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

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    scheduler = DiffusionScheduler(
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        device=device
    )

    normal_model = load_model(
        checkpoint,
        device
    )

    ema_model = load_ema_model(
        checkpoint,
        device
    )

    normal_images = generate_images(
        normal_model,
        scheduler,
        device
    )

    ema_images = generate_images(
        ema_model,
        scheduler,
        device
    )

    plot_grid(
        normal_images,
        "CIFAR-10 DDPM - Standard Model",
        sample_dir / "cifar10_standard_grid.png"
    )

    plot_grid(
        ema_images,
        "CIFAR-10 DDPM - EMA Model",
        sample_dir / "cifar10_ema_grid.png"
    )


if __name__ == "__main__":
    main()