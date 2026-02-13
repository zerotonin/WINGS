"""
W.I.N.G.S. GPU Benchmark — Growth scenario
===========================================
Tests the simulation starting from 50 beetles growing to carrying capacity
under different density-dependent mortality modes.

Run interactively on a GPU node:
    srun --partition=aoraki_gpu_L40 --gpus-per-task=1 --mem=16G --time=00:30:00 \
         --account=geuba03p --pty bash
    conda activate wings-gpu
    python benchmark_gpu.py
"""

import torch
import time
from gpu_simulation import GPUSimulation, SimConfig


def benchmark(mortality_mode, n_days=120):
    """Run a growth simulation from 50 beetles and report dynamics + timing."""
    cfg = SimConfig(
        initial_population=50,
        max_population=20_000,
        max_eggs=800_000,
        grid_size=500,
        mating_backend='brute',    # brute is fine for small→medium N
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=42,
        mortality_mode=mortality_mode,
        mortality_beta=2.0,
        cannibalism_rate=0.0001,
        wolbachia_effects={
            'cytoplasmic_incompatibility': True,
            'male_killing': False,
            'increased_exploration_rate': True,
            'increased_eggs': True,
            'reduced_eggs': False,
        },
    )

    print(f"\n{'='*65}")
    print(f"  Mortality: {mortality_mode}  |  {n_days} days  |  device: {cfg.device}")
    print(f"  Start: {cfg.initial_population}  →  K = {cfg.max_population}")
    print(f"{'='*65}")

    sim = GPUSimulation(cfg)

    # Warm-up
    sim.step_one_day()
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    t0 = time.time()
    for day in range(2, n_days + 1):
        sim.step_one_day()
        if day % 15 == 0 or day == n_days:
            pop = sim.get_population_size()
            inf = sim.get_infection_rate()
            eggs = sim.pop.n_eggs
            total = sim.pop.n
            print(f"  Day {day:4d} | Pop {pop:6,d} | Inf {inf:.3f} | "
                  f"Eggs {eggs:7,d} | Total {total:7,d}")

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - t0
    per_day = elapsed / (n_days - 1)

    sex = sim.get_sex_ratio()
    print(f"\n  Final state:")
    print(f"    F_U={sex['F_U']:,}  F_I={sex['F_I']:,}  "
          f"M_U={sex['M_U']:,}  M_I={sex['M_I']:,}")
    print(f"  Timing:")
    print(f"    Total: {elapsed:.1f}s  ({per_day*1000:.0f} ms/day)")
    print(f"    Projected 365 days: {per_day * 365:.0f}s")

    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"    Peak VRAM: {mem:.2f} GB")
        torch.cuda.reset_peak_memory_stats()

    return per_day


if __name__ == '__main__':
    print(f"PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No CUDA GPU — running on CPU")

    # Test each mortality mode
    results = {}
    for mode in ['none', 'logistic', 'cannibalism', 'contest']:
        try:
            per_day = benchmark(mode, n_days=120)
            results[mode] = per_day
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"  OOM with {mode}!")
                torch.cuda.empty_cache()
            else:
                raise

    # Summary
    print(f"\n{'='*65}")
    print(f"  Summary (ms per simulated day, 120-day run)")
    print(f"{'='*65}")
    for mode, pd in results.items():
        proj_365 = pd * 365
        print(f"  {mode:15s}: {pd*1000:7.0f} ms/day  →  {proj_365:.0f}s for 365 days")
    print(f"\n  With 4 GPUs, 160 runs would take ~{max(results.values()) * 365 * 160 / 4 / 60:.0f} min")
