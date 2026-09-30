"""
FID evaluation (resumable, time-capped).

Usage (from project root):
    python -m evaluation.fid1 --num-samples 50000 --batch-size 500 --max-hours 8
    python -m evaluation.fid1 --num-samples 10000 --no-amp   # quick check

Generated images are saved in chunks under samples/fid_chunks_<tag>/ so an
interrupted session can resume where it stopped. Pass --max-hours to stop
cleanly between chunks before Kaggle's session limit hits; rerun the same
command afterward to continue.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance

from config import load_config, PROJECT_ROOT
from diffusion.scheduler import DiffusionScheduler
from diffusion.unet import UNet
from diffusion.sampler import sample


def load_model(checkpoint, device, use_ema):
    cfg = checkpoint["config"]

    model = UNet(
        in_channels=cfg["data"]["channels"],
        out_channels=cfg["data"]["channels"],
        base_channels=cfg["model"]["base_channels"],
        time_embedding_dim=cfg["model"]["time_embedding_dim"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if use_ema:
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.copy_(checkpoint["ema_state_dict"][name])

    model.eval()
    return model


def to_uint8(images):
    images = (images.clamp(-1, 1) + 1) / 2
    return (images * 255).round().to(torch.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--max-hours", type=float, default=None,
        help="Stop cleanly between chunks once this many hours have "
             "elapsed. Rerun the same command to resume from the next "
             "missing chunk."
    )
    args = parser.parse_args()

    session_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    config = load_config()

    ckpt_path = (
        Path(args.checkpoint) if args.checkpoint
        else PROJECT_ROOT / config["paths"]["checkpoint_dir"]
        / config["model"]["checkpoint_name"]
    )

    checkpoint = torch.load(
        ckpt_path, map_location=device, weights_only=False
    )

    use_ema = not args.no_ema
    tag = "ema" if use_ema else "standard"

    print(f"Checkpoint: {ckpt_path}")
    print(f"Epoch: {checkpoint.get('epoch')}  Weights: {tag}  AMP: {use_amp}")

    model = load_model(checkpoint, device, use_ema)

    diff = checkpoint["config"]["diffusion"]
    scheduler = DiffusionScheduler(
        num_timesteps=diff["timesteps"],
        beta_start=diff["beta_start"],
        beta_end=diff["beta_end"],
        device=device
    )

    chunk_dir = (
        PROJECT_ROOT / config["paths"]["sample_dir"] / f"fid_chunks_{tag}"
    )
    chunk_dir.mkdir(parents=True, exist_ok=True)

    num_chunks = (args.num_samples + args.batch_size - 1) // args.batch_size

    stopped_early = False
    chunk_time = None

    for i in range(num_chunks):
        chunk_path = chunk_dir / f"chunk_{i:05d}.npy"
        if chunk_path.exists():
            continue

        # Stop before starting a chunk we likely can't finish in budget.
        if args.max_hours:
            elapsed = time.time() - session_start
            projected = (
                elapsed + chunk_time * 1.2
                if chunk_time is not None
                else elapsed
            )
            if projected > args.max_hours * 3600:
                stopped_early = True
                print(
                    f"Time budget reached after {elapsed / 3600:.2f}h "
                    f"({i}/{num_chunks} chunks done this run). "
                    "Rerun the same command to resume."
                )
                break

        chunk_start = time.time()

        n = min(args.batch_size, args.num_samples - i * args.batch_size)
        torch.manual_seed(1000 + i)

        images, _ = sample(
            model=model,
            scheduler=scheduler,
            num_samples=n,
            image_size=32,
            channels=3,
            device=device,
            use_amp=use_amp
        )

        np.save(chunk_path, to_uint8(images).cpu().numpy())
        chunk_time = time.time() - chunk_start
        elapsed = time.time() - session_start
        print(
            f"Saved chunk {i + 1}/{num_chunks}  "
            f"(chunk took {chunk_time / 60:.1f} min, "
            f"session {elapsed / 3600:.2f}h)",
            flush=True
        )

    if stopped_early:
        print("Exiting without computing FID (not all chunks generated yet).")
        return

    missing = [
        i for i in range(num_chunks)
        if not (chunk_dir / f"chunk_{i:05d}.npy").exists()
    ]
    if missing:
        print(f"Still missing {len(missing)} chunks; rerun to finish sampling.")
        return

    # ---- All chunks present: compute FID vs CIFAR-10 TRAIN set ----
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    real = datasets.CIFAR10(
        root=str(PROJECT_ROOT / config["data"]["data_dir"]),
        train=True,
        download=True,
        transform=transforms.PILToTensor()
    )

    print("Feeding real images...")
    for start in range(0, len(real), 500):
        batch = torch.stack([
            real[j][0] for j in range(start, min(start + 500, len(real)))
        ]).to(device)
        fid.update(batch, real=True)

    print("Feeding generated images...")
    for i in range(num_chunks):
        chunk = torch.from_numpy(
            np.load(chunk_dir / f"chunk_{i:05d}.npy")
        ).to(device)
        fid.update(chunk, real=False)

    score = fid.compute().item()

    print("=" * 50)
    print(f"FID ({tag}, {args.num_samples} samples): {score:.3f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
