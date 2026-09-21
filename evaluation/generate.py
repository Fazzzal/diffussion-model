import matplotlib.pyplot as plt
import torch
from pathlib import Path

from diffusion.scheduler import DiffusionScheduler
from diffusion.unet import UNet
from diffusion.sampler import sample


def load_standard_model(checkpoint, device):
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
    seed,
    num_samples=16
):
    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    images, _ = sample(
        model=model,
        scheduler=scheduler,
        num_samples=num_samples,
        image_size=32,
        channels=3,
        device=device
    )

    images = (
        images.clamp(-1, 1) + 1
    ) / 2

    return images


def plot_comparison(
    standard_images,
    ema_images,
    output_path
):
    fig, axes = plt.subplots(
        4,
        8,
        figsize=(16, 8)
    )

    for i in range(16):
        row = i // 4
        pair = i % 4

        standard_column = pair * 2
        ema_column = standard_column + 1

        standard_image = (
            standard_images[i]
            .cpu()
            .permute(1, 2, 0)
        )

        ema_image = (
            ema_images[i]
            .cpu()
            .permute(1, 2, 0)
        )

        axes[row, standard_column].imshow(
            standard_image
        )

        axes[row, ema_column].imshow(
            ema_image
        )

        axes[row, standard_column].axis("off")
        axes[row, ema_column].axis("off")

    axes[0, 0].set_title("Standard")
    axes[0, 1].set_title("EMA")

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
        "cuda" if torch.cuda.is_available()
        else "cpu"
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

    standard_model = load_standard_model(
        checkpoint,
        device
    )

    ema_model = load_ema_model(
        checkpoint,
        device
    )

    seed = 42

    standard_images = generate_images(
        standard_model,
        scheduler,
        device,
        seed
    )

    ema_images = generate_images(
        ema_model,
        scheduler,
        device,
        seed
    )

    output_path = (
        sample_dir
        / "standard_vs_ema.png"
    )

    plot_comparison(
        standard_images,
        ema_images,
        output_path
    )

    print(
        f"Comparison saved to: {output_path}"
    )


if __name__ == "__main__":
    main()