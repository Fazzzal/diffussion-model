import torch

from diffusion.unet import UNet


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        time_embedding_dim=128
    ).to(device)

    x = torch.randn(
        4,
        3,
        32,
        32,
        device=device
    )

    timesteps = torch.randint(
        0,
        1000,
        (4,),
        device=device
    )

    output = model(
        x,
        timesteps
    )

    print(
        "Input shape:",
        x.shape
    )

    print(
        "Output shape:",
        output.shape
    )

    print(
        "Number of parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )


if __name__ == "__main__":
    main()