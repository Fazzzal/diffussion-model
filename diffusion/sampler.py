import torch


@torch.no_grad()
def sample(
    model,
    scheduler,
    num_samples=16,
    image_size=32,
    channels=3,
    device="cpu",
    use_amp=False
):
    model.eval()

    x = torch.randn(
        num_samples,
        channels,
        image_size,
        image_size,
        device=device
    )

    trajectory = []

    for timestep in reversed(range(scheduler.num_timesteps)):
        t = torch.full(
            (num_samples,),
            timestep,
            device=device,
            dtype=torch.long
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp and str(device).startswith("cuda")
        ):
            predicted_noise = model(
                x,
                t
            )

        predicted_noise = predicted_noise.float()

        x = scheduler.step(
            predicted_noise,
            timestep,
            x
        )

        if timestep in [999, 800, 600, 400, 200, 0]:
            trajectory.append(
                (timestep, x.clone())
            )

    return x, trajectory