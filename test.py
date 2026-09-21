import matplotlib.pyplot as plt
import torch

from datasets.mnist import get_mnist_dataloader
from diffusion.scheduler import DiffusionScheduler


def main():
    dataloader = get_mnist_dataloader(batch_size=1)

    images, labels = next(iter(dataloader))

    image = images[0:1]

    scheduler = DiffusionScheduler()

    timesteps = [0, 100, 300, 500, 700, 999]

    fig, axes = plt.subplots(1, len(timesteps), figsize=(15, 3))

    for ax, timestep in zip(axes, timesteps):
        t = torch.tensor([timestep])

        noisy_image, _ = scheduler.add_noise(
            image,
            t
        )

        display_image = noisy_image[0, 0].detach().numpy()

        ax.imshow(
            display_image,
            cmap="gray",
            vmin=-1,
            vmax=1
        )

        ax.set_title(f"t = {timestep}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()