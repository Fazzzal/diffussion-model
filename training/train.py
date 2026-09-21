from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from config import load_config
from datasets.mnist import get_mnist_dataloader
from diffusion.scheduler import DiffusionScheduler
from diffusion.unet import DiffusionUNet
from training.loss import diffusion_loss


def set_seed(seed):
    torch.manual_seed(seed)


def main():
    config = load_config()

    set_seed(config["training"]["seed"])

    device = torch.device(
        config["device"]
    )

    project_root = Path(__file__).resolve().parent.parent

    log_dir = (
        project_root
        / config["paths"]["log_dir"]
        / "mnist"
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    writer = SummaryWriter(
        log_dir=str(log_dir)
    )

    dataloader = get_mnist_dataloader(
        batch_size=config["training"]["batch_size"],
        data_dir=config["data"]["data_dir"]
    )

    scheduler = DiffusionScheduler(
        num_timesteps=config["diffusion"]["timesteps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        device=device
    )

    model = DiffusionUNet(
        in_channels=config["data"]["channels"],
        out_channels=config["data"]["channels"],
        base_channels=config["model"]["base_channels"],
        time_embedding_dim=config["model"]["time_embedding_dim"]
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    model.train()

    epochs = config["training"]["epochs"]

    global_step = 0

    for epoch in range(epochs):

        total_loss = 0.0

        for batch_idx, (images, _) in enumerate(dataloader):

            images = images.to(device)

            batch_size = images.shape[0]

            timesteps = torch.randint(
                0,
                scheduler.num_timesteps,
                (batch_size,),
                device=device
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

            total_loss += loss_value

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

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"Batch {batch_idx}/{len(dataloader)} "
                    f"Loss: {loss_value:.4f}"
                )

        average_loss = (
            total_loss / len(dataloader)
        )

        writer.add_scalar(
            "Loss/Epoch",
            average_loss,
            epoch
        )

        print(
            f"Epoch {epoch + 1} "
            f"Average Loss: {average_loss:.4f}"
        )

    checkpoint_dir = (
        project_root
        / config["paths"]["checkpoint_dir"]
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = (
        checkpoint_dir
        / "diffusion_mnist.pth"
    )

    torch.save(
        model.state_dict(),
        checkpoint_path
    )

    writer.close()

    print(
        f"Model saved to: {checkpoint_path}"
    )

    print(
        f"TensorBoard logs saved to: {log_dir}"
    )


if __name__ == "__main__":
    main()