import torch

from data_loaders.cifar10 import get_cifar10_dataloader


def main():
    dataloader = get_cifar10_dataloader(
        batch_size=4,
        data_dir="data"
    )

    images, labels = next(iter(dataloader))

    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)
    print("Image dtype:", images.dtype)
    print("Minimum:", images.min().item())
    print("Maximum:", images.max().item())
    print("Labels:", labels.tolist())


if __name__ == "__main__":
    main()