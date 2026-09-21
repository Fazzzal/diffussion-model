import torch


@torch.no_grad()
def sample(
    model,
    scheduler,
    num_samples=16,
    image_size=28,
    device="cpu"
):
    model.eval()

    x = torch.randn(
        num_samples,
        1,
        image_size,
        image_size,
        device=device
    )

    trajectory = []

    for timestep in reversed(
        range(scheduler.num_timesteps)
    ):
        t = torch.full(
            (num_samples,),
            timestep,
            device=device,
            dtype=torch.long
        )

        predicted_noise = model(
            x,
            t
        )

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