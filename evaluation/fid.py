import torch
from pathlib import Path
from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance

from diffusion.scheduler import DiffusionScheduler
from diffusion.unet import UNet
from diffusion.sampler import sample


NUM_SAMPLES = 1000
BATCH_SIZE = 64


def load_model(checkpoint, device, use_ema):
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        time_embedding_dim=128
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if use_ema:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in checkpoint["ema_state_dict"]:
                    parameter.copy_(
                        checkpoint["ema_state_dict"][name]
                    )

    model.eval()

    return model


@torch.no_grad()
def generate_batch(
    model,
    scheduler,
    device,
    batch_size
):
    images, _ = sample(
        model=model,
        scheduler=scheduler,
        num_samples=batch_size,
        image_size=32,
        channels=3,
        device=device
    )

    images = (
        images.clamp(-1, 1) + 1
    ) / 2

    images = (
        images * 255
    ).round().to(torch.uint8)

    return images


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

    print(f"Device: {device}")
    print(f"Number of samples: {NUM_SAMPLES}")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    real_dataset = datasets.CIFAR10(
        root=str(
            project_root / "data"
        ),
        train=False,
        download=True,
        transform=transform
    )

    print("Preparing FID evaluator...")

    fid_standard = FrechetInceptionDistance(
        feature=2048,
        normalize=False
    ).to(device)

    fid_ema = FrechetInceptionDistance(
        feature=2048,
        normalize=False
    ).to(device)

    standard_model = load_model(
        checkpoint,
        device,
        use_ema=False
    )

    ema_model = load_model(
        checkpoint,
        device,
        use_ema=True
    )

    print()
    print("Generating samples and calculating FID...")
    print()

    generated = 0

    while generated < NUM_SAMPLES:
        current_batch_size = min(
            BATCH_SIZE,
            NUM_SAMPLES - generated
        )

        real_images = torch.stack([
            real_dataset[i][0]
            for i in range(
                generated,
                generated + current_batch_size
            )
        ])

        real_images = (
            real_images * 255
        ).round().to(torch.uint8)

        real_images = real_images.to(device)

        standard_images = generate_batch(
            standard_model,
            scheduler,
            device,
            current_batch_size
        )

        ema_images = generate_batch(
            ema_model,
            scheduler,
            device,
            current_batch_size
        )

        fid_standard.update(
            real_images,
            real=True
        )

        fid_standard.update(
            standard_images,
            real=False
        )

        fid_ema.update(
            real_images,
            real=True
        )

        fid_ema.update(
            ema_images,
            real=False
        )

        generated += current_batch_size

        print(
            f"Processed {generated}/{NUM_SAMPLES}"
        )

    standard_score = fid_standard.compute()
    ema_score = fid_ema.compute()

    print()
    print("=" * 50)
    print("FID RESULTS")
    print("=" * 50)
    print(
        f"Standard model FID: "
        f"{standard_score.item():.4f}"
    )
    print(
        f"EMA model FID: "
        f"{ema_score.item():.4f}"
    )
    print("=" * 50)


if __name__ == "__main__":
    main()