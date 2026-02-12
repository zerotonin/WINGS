"""
Quick benchmark to verify the GPU simulation works and estimate runtime.
Run this interactively on a GPU node before submitting the full array job:

    srun --partition=gpu --gres=gpu:l40s:1 --mem=16G --time=00:10:00 \
         python benchmark_gpu.py
"""

import torch
import time
from gpu_simulation import GPUSimulation, SimConfig

def benchmark(n_pop, backend, n_days=30):
    """Run a short simulation and report timing."""
    cfg = SimConfig(
        initial_population=n_pop,
        max_population=int(n_pop * 1.25),
        max_eggs=int(n_pop * 40),   # large enough for the 23-day egg pipeline
        grid_size=max(200, int(n_pop**0.5) * 5),  # scale grid with population
        mating_backend=backend,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=42,
        wolbachia_effects={
            'cytoplasmic_incompatibility': True,
            'male_killing': True,
            'increased_exploration_rate': True,
            'increased_eggs': True,
            'reduced_eggs': False,
        },
    )

    print(f"\n{'='*60}")
    print(f"  N = {n_pop:,}  |  backend = {backend}  |  device = {cfg.device}")
    print(f"  grid = {cfg.grid_size}×{cfg.grid_size}  |  {n_days} days")
    print(f"{'='*60}")

    sim = GPUSimulation(cfg)

    # Warm-up (first step triggers CUDA kernel compilation)
    sim.step_one_day()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for day in range(2, n_days + 1):
        sim.step_one_day()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - t0
    per_day = elapsed / (n_days - 1)

    print(f"\n  Results after {n_days} days:")
    print(f"    Population : {sim.get_population_size():,}")
    print(f"    Infection  : {sim.get_infection_rate():.3f}")
    print(f"    Eggs       : {sim.pop.n_eggs:,}")
    print(f"    Total entities (adults+eggs): {sim.pop.n:,}")
    sex = sim.get_sex_ratio()
    print(f"    Sex ratio  : F_U={sex['F_U']:,} F_I={sex['F_I']:,} "
          f"M_U={sex['M_U']:,} M_I={sex['M_I']:,}")
    print(f"\n  Timing:")
    print(f"    Total      : {elapsed:.2f}s  ({per_day*1000:.1f} ms/day)")
    print(f"    Projected 365 days: {per_day * 365:.1f}s")

    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"    Peak VRAM  : {mem:.2f} GB")

    return per_day

if __name__ == '__main__':
    print(f"PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No CUDA GPU detected — running on CPU (will be slower)")

    # Run benchmarks at increasing scale
    for n in [1_000, 5_000, 20_000]:
        for backend in ['brute', 'cell_list']:
            try:
                benchmark(n, backend, n_days=30)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"  OOM at N={n:,} with {backend} backend!")
                    torch.cuda.empty_cache()
                else:
                    raise
