import torch

from diffusion.unet import UNet


def main():
    device = "cpu"

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

    t = torch.randint(
        0,
        1000,
        (4,),
        device=device
    )

    with torch.no_grad():
        output = model(x, t)

    print("Input shape:", x.shape)
    print("Timestep shape:", t.shape)
    print("Output shape:", output.shape)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Number of parameters:", total_parameters)


if __name__ == "__main__":
    main()