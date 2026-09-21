import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import load_config, PROJECT_ROOT
from data_loaders.mnist import get_mnist_dataloader
from data_loaders.cifar10 import get_cifar10_dataloader
from diffusion.scheduler import DiffusionScheduler
from diffusion.unet import UNet
from training.loss import diffusion_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    config = load_config()

    set_seed(config["training"]["seed"])

    device = torch.device(
        config.get("device", "cpu")
    )

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but no CUDA GPU is available."
        )

    dataset_name = config["data"]["dataset"]

    data_dir = (
        PROJECT_ROOT /
        config["data"]["data_dir"]
    )

    batch_size = config["training"]["batch_size"]

    if dataset_name == "mnist":
        dataloader = get_mnist_dataloader(
            batch_size=batch_size,
            data_dir=str(data_dir)
        )
    elif dataset_name == "cifar10":
        dataloader = get_cifar10_dataloader(
            batch_size=batch_size,
            data_dir=str(data_dir)
        )
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}"
        )

    channels = config["data"]["channels"]

    model = UNet(
        in_channels=channels,
        out_channels=channels,
        base_channels=config["model"]["base_channels"],
        time_embedding_dim=config["model"]["time_embedding_dim"]
    ).to(device)

    scheduler = DiffusionScheduler(
        num_timesteps=config["diffusion"]["timesteps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        device=device
    )

    optimizer = Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    epochs = config["training"]["epochs"]

    checkpoint_dir = (
        PROJECT_ROOT /
        config["paths"]["checkpoint_dir"]
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = (
        checkpoint_dir /
        f"{dataset_name}_attention_unet.pth"
    )

    log_dir = (
        PROJECT_ROOT /
        config["paths"]["log_dir"] /
        dataset_name
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    writer = SummaryWriter(
        log_dir=str(log_dir)
    )

    start_epoch = 0

    if checkpoint_path.exists():
        print(
            f"Found checkpoint: {checkpoint_path}"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):
            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(
                    checkpoint["optimizer_state_dict"]
                )

            start_epoch = checkpoint.get(
                "epoch",
                0
            )

            print(
                f"Resuming from epoch {start_epoch}"
            )
        else:
            model.load_state_dict(checkpoint)

            print(
                "Loaded model weights."
            )

    print()
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Starting epoch: {start_epoch + 1}")
    print(f"Batch size: {batch_size}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"TensorBoard logs: {log_dir}")
    print()

    global_step = start_epoch * len(dataloader)

    model.train()

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for batch_idx, (images, _) in enumerate(
            progress_bar
        ):
            images = images.to(device)

            current_batch_size = images.shape[0]

            timesteps = torch.randint(
                0,
                scheduler.num_timesteps,
                (current_batch_size,),
                device=device,
                dtype=torch.long
            )

            noisy_images, noise = scheduler.add_noise(
                images,
                timesteps
            )

            predicted_noise = model(
                noisy_images,
                timesteps
            )

            loss = diffusion_loss(
                predicted_noise,
                noise
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            loss_value = loss.item()

            epoch_loss += loss_value

            writer.add_scalar(
                "Loss/Batch",
                loss_value,
                global_step
            )

            writer.add_scalar(
                "Training/Learning_Rate",
                optimizer.param_groups[0]["lr"],
                global_step
            )

            global_step += 1

            progress_bar.set_postfix(
                loss=f"{loss_value:.4f}"
            )

        average_loss = (
            epoch_loss /
            len(dataloader)
        )

        writer.add_scalar(
            "Loss/Epoch",
            average_loss,
            epoch
        )

        print(
            f"Epoch {epoch + 1} Average Loss: "
            f"{average_loss:.4f}"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "loss": average_loss,
            "config": config
        }

        torch.save(
            checkpoint,
            checkpoint_path
        )

        print(
            f"Checkpoint saved after epoch "
            f"{epoch + 1}: {checkpoint_path}"
        )

    writer.close()

    print()
    print("Training complete.")
    print(f"Final checkpoint: {checkpoint_path}")
    print(f"TensorBoard logs: {log_dir}")


if __name__ == "__main__":
    main()