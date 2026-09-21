from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_cifar10_dataloader(batch_size=64, data_dir="data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    return dataloader