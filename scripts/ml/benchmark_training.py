"""
Benchmark the OceanEmbed training input pipeline.

This does NOT train the model or modify checkpoints.

It compares DataLoader worker counts while keeping
batch size fixed.

Run:

    uv run python -m scripts.ml.benchmark_training
"""

import time

import torch

from scripts.ml.dataloader import create_dataloaders
from scripts.ml.losses import MaskedMSELoss
from scripts.ml.models.temporal_unet import TemporalUNet


BATCH_SIZE = 8

NUM_WORKERS_LIST = [0, 2, 4]

NUM_BATCHES = 20


def benchmark(batch_size, num_workers):

    print()
    print("-" * 70)
    print(
        f"Batch size: {batch_size} | "
        f"Workers: {num_workers}"
    )
    print("-" * 70)

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------
    train_loader, _, _ = create_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = TemporalUNet(
        in_channels=7,
        out_channels=15,
        base_channels=32,
    ).to(device)

    loss_fn = MaskedMSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    model.train()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------------
    # Create iterator once.
    # --------------------------------------------------------
    iterator = iter(train_loader)

    # --------------------------------------------------------
    # Warm-up batch.
    # --------------------------------------------------------
    warmup = next(iterator)

    x = warmup["x"].to(
        device,
        non_blocking=True,
    )

    y = warmup["y"].to(
        device,
        non_blocking=True,
    )

    mask = warmup["mask"].to(
        device,
        non_blocking=True,
    )

    optimizer.zero_grad(set_to_none=True)

    prediction = model(x)

    loss = loss_fn(
        prediction,
        y,
        mask,
    )

    loss.backward()
    optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # --------------------------------------------------------
    # Timed batches.
    # --------------------------------------------------------
    data_time = 0.0
    compute_time = 0.0

    previous_time = time.perf_counter()

    for _ in range(NUM_BATCHES):

        batch = next(iterator)

        batch_ready = time.perf_counter()

        data_time += (
            batch_ready - previous_time
        )

        x = batch["x"].to(
            device,
            non_blocking=True,
        )

        y = batch["y"].to(
            device,
            non_blocking=True,
        )

        mask = batch["mask"].to(
            device,
            non_blocking=True,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        compute_start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

        prediction = model(x)

        loss = loss_fn(
            prediction,
            y,
            mask,
        )

        loss.backward()

        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()

        compute_end = time.perf_counter()

        compute_time += (
            compute_end - compute_start
        )

        previous_time = compute_end

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------
    avg_data = data_time / NUM_BATCHES
    avg_compute = compute_time / NUM_BATCHES

    total_time = (
        avg_data + avg_compute
    )

    throughput = (
        batch_size / total_time
    )

    print()
    print(
        f"Data time / batch:    "
        f"{avg_data:.4f} s"
    )

    print(
        f"Compute time / batch: "
        f"{avg_compute:.4f} s"
    )

    print(
        f"Approx total / batch: "
        f"{total_time:.4f} s"
    )

    print(
        f"Throughput:           "
        f"{throughput:.2f} samples/s"
    )

    if device.type == "cuda":

        peak_memory = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        print(
            f"Peak GPU memory:      "
            f"{peak_memory:.2f} GB"
        )


def main():

    print("=" * 70)
    print("OCEANEMBED DATALOADER BENCHMARK")
    print("=" * 70)

    print()
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Workers: {NUM_WORKERS_LIST}")
    print(f"Timed batches: {NUM_BATCHES}")

    for num_workers in NUM_WORKERS_LIST:

        try:

            benchmark(
                batch_size=BATCH_SIZE,
                num_workers=num_workers,
            )

        except RuntimeError as exc:

            if (
                "out of memory"
                in str(exc).lower()
            ):

                print()
                print(
                    f"Workers {num_workers}: "
                    f"CUDA OUT OF MEMORY"
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            else:
                raise

    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()