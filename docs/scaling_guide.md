# Scaling Guide: HPC Deployment & Population Dynamics

This guide covers how to run W.I.N.G.S. simulations efficiently on
SLURM-managed HPC clusters, and explains the biological reasoning
behind the density-dependent mortality models available in the GPU ABM.

---

## SLURM Scheduling Strategies

### Overview

A full factorial experiment (16 phenotype combinations × 200 replicates
= 3,200 simulations) takes approximately 2–4 hours on modern GPU
hardware. Three scheduling strategies are available; the choice depends
on cluster load and the number of GPUs available.

### Option A: Array Job

Each simulation gets its own SLURM task via `--array`.

```bash
#SBATCH --array=0-3199
#SBATCH --gpus-per-task=1
```

Each task independently initialises conda, claims a GPU, and runs one
simulation.

| Advantage | Disadvantage |
|-----------|--------------|
| Fault tolerant — one crash doesn't affect others | High scheduling overhead (3,200 separate allocations) |
| Tasks run across many nodes simultaneously | Conda + filesystem init repeated per task (~10 s each) |
| Simple to resume (failed tasks re-queue independently) | Queue wait multiplied across all tasks |

Best suited for clusters with abundant GPU availability and fast
scheduling.

### Option B: Single Job, Sequential Loop

One GPU, one job, a bash loop runs all simulations in sequence.

```bash
#SBATCH --gpus-per-task=1
```

| Advantage | Disadvantage |
|-----------|--------------|
| Zero scheduling overhead between runs | Purely serial: 3,200 × ~3 min ≈ 160 hours |
| One conda init, one filesystem wait | Wastes available GPUs |

Only practical for small test runs or clusters where GPU jobs are
heavily rate-limited.

### Option C: Packed Parallel Batches (Recommended)

One job requests multiple GPUs and runs simulations in parallel
batches using `CUDA_VISIBLE_DEVICES`.

```bash
#SBATCH --gpus-per-task=4
```

A bash loop launches one simulation per GPU as a background process,
waits for all to finish, then launches the next batch:

```bash
CUDA_VISIBLE_DEVICES=0 python -m wings.models.gpu_abm --output run1.csv &
CUDA_VISIBLE_DEVICES=1 python -m wings.models.gpu_abm --output run2.csv &
CUDA_VISIBLE_DEVICES=2 python -m wings.models.gpu_abm --output run3.csv &
CUDA_VISIBLE_DEVICES=3 python -m wings.models.gpu_abm --output run4.csv &
wait  # block until all 4 finish
```

Each process sees a single GPU as `cuda:0`, so no code changes are
needed.

| Advantage | Disadvantage |
|-----------|--------------|
| 1 scheduler allocation, 1 conda init | If the job dies, must resubmit |
| 4× throughput (or more, depending on GPUs requested) | — |
| Auto-skip of completed runs (resume after crash) | — |

This is the default strategy used by all SLURM scripts in `slurm/`.

**Comparison at a glance:**

| Metric | Array (3,200 tasks) | Sequential (1 GPU) | Packed (4 GPUs) |
|--------|--------------------|--------------------|-----------------|
| Scheduler allocations | 3,200 | 1 | 1 |
| Overhead | 3,200 × ~10 s | 10 s | 10 s |
| Estimated wall time | Queue-dependent | ~160 h | ~12 h |
| Crash recovery | Automatic | Manual | Automatic |

### The Smart Submission Script

The `slurm/submit_delta_p.sh` script implements a two-phase approach:
when run interactively, it scans the output directory for existing
result files, writes a manifest of missing runs, calculates the exact
array size needed, and submits only the gap-filling jobs. This avoids
reserving GPU resources only to skip completed simulations.

```bash
# Phase 1: scan and submit (interactive)
bash slurm/submit_delta_p.sh

# Dry run — list missing without submitting
DRY_RUN=1 bash slurm/submit_delta_p.sh
```

---

## Density-Dependent Mortality Models

### Motivation

The form of density-dependent regulation affects *Wolbachia* invasion
dynamics (Hancock et al. 2016, *J. Appl. Ecol.*; Hancock et al. 2012,
*J. R. Soc. Interface*). W.I.N.G.S. provides four mortality modes,
selectable via the `--mortality` CLI flag, to allow researchers to
explore this interaction.

### Available Modes

#### `none` — Hard Capacity Cap

Population grows freely until reaching `max_population`, then excess
adults are randomly removed. Simple but biologically unrealistic —
creates artificial boom–bust cycles at carrying capacity.

Suitable for backward compatibility or when density dependence is
not the focus of the experiment.

#### `logistic` — Density-Dependent Adult Mortality

Per-capita adult death rate increases smoothly with population density:

$$\mu(N) = \mu_0 + \mu_0 \cdot \left(\frac{N}{K}\right)^\beta$$

At low density ($N \ll K$), mortality equals the natural baseline. As
$N$ approaches $K$, the death rate rises. The exponent $\beta$ controls
steepness: $\beta = 1$ is linear; $\beta = 2$ gives a gentler ramp
that steepens near $K$.

This is a general resource-competition model. It affects infected and
uninfected adults equally, slowing overall population growth near $K$
without directly creating selection for or against *Wolbachia*.

#### `cannibalism` — Egg Cannibalism (Default)

Adults consume eggs at a density-dependent rate:

$$P(\text{egg eaten per hour}) = r \cdot N_{\text{adults}} \cdot \left(\frac{N}{K}\right)^\beta$$

where $r$ is the per-adult per-hour cannibalism rate.

**Biological basis.** Egg predation by adults and larvae is the primary
population regulation mechanism in *Tribolium* (Park 1934; Daly & Ryan
1983). Generation mortality ranges from ~42% at low density to ~74%
at high density, with predation on eggs in the first 10 days as the
main driver (Sonleitner & Guthrie 1991). This is well established in
the *Tribolium* literature (reviewed in Pointer et al. 2021).

**Interaction with *Wolbachia*.** Egg cannibalism interacts
non-trivially with CI. When CI is active, uninfected females mated
with infected males produce fewer viable eggs (many are already
destroyed by CI), while infected females' eggs are fully viable. Both
then face equal cannibalism pressure. The net effect is that CI +
cannibalism creates a stronger selective advantage for *Wolbachia*
than CI alone, because the reproductive slots lost to CI are not
refilled. This effect is consistent with density-dependent competition
models of *Wolbachia* invasion (Hancock et al. 2016, *BMC Biology*).

**Parameter tuning.** The default cannibalism rate ($r = 6 \times
10^{-7}$) is calibrated so that at steady state ($N = K = 20{,}000$),
approximately 0.1% of eggs survive the 552-hour incubation, balancing
the natural adult death rate of ~2.3 deaths per hour. Adjust
`--cannibalism-rate` to match observed egg-to-adult survival rates for
other species or culture conditions.

#### `contest` — Contest Competition

When $N > K$, each excess individual has a per-hour death probability
of $(N - K) / N$. Below $K$, no extra mortality. This creates a sharp
threshold — populations hovering above $K$ are quickly pushed back.

This models territorial or interference competition. Less appropriate
for *Tribolium* (which do not hold territories) but useful as a
mathematical comparison case.

### Choosing a Mode

For *Tribolium* simulations, `cannibalism` (the default) is recommended.
It is the most biologically grounded option and produces qualitatively
different — and more realistic — invasion dynamics than a hard cap.

For well-mixed or non-spatial comparisons, `logistic` provides a
smooth, analytically tractable alternative. The `none` and `contest`
modes are primarily useful for sensitivity analysis.

---

## Runtime & Memory Expectations

### Growth Scenario (50 → 20,000 beetles)

A single 365-day simulation starting from 50 beetles has three
computational phases:

| Phase | Population | Eggs in Pipeline | Time per Simulated Day |
|-------|-----------|------------------|----------------------|
| Early growth (day 1–60) | 50 → ~500 | 0 → ~5,000 | ~5 ms |
| Exponential growth (day 60–150) | 500 → ~15,000 | 5,000 → ~500,000 | ~50–200 ms |
| Near carrying capacity (day 150+) | ~20,000 | Cannibalism-limited | ~100–500 ms |

**Single run:** ~1–3 minutes on an NVIDIA L40S, A100, or H100.

**Full experiment** (3,200 runs on 4 GPUs): ~12 hours.

**Quick test** (10 reps × 16 combos, 30-day runs on 4 GPUs): ~5 minutes.

### VRAM Usage

At peak population (~500,000 entities × 32 bytes × 8 attributes):
~130 MB. A single simulation fits comfortably on any modern GPU.
Six simulations packed per GPU (as used in the Δp sweep scripts)
require < 1 GB total.

---

## References

- Daly, P.J. & Ryan, M.F. (1983). Density-related mortality of
  *Tribolium confusum*. *Res. Popul. Ecol.* 25, 210–219.
- Hancock, P.A. & Godfray, H.C.J. (2012). Modelling *Wolbachia* spread
  in spatially heterogeneous environments. *J. R. Soc. Interface* 9,
  3045–3054.
- Hancock, P.A. et al. (2016). Predicting *Wolbachia* invasion dynamics
  using density-dependent demographic traits. *BMC Biology* 14, 96.
- Hancock, P.A. et al. (2016). Density-dependent population dynamics
  slow the spread of wMel *Wolbachia*. *J. Appl. Ecol.* 53, 785–793.
- Park, T. (1934). Studies in population physiology III. *J. Exp. Zool.*
  68, 167–182.
- Pointer, M.D. et al. (2021). *Tribolium* beetles as a model system
  in evolution and ecology. *Heredity* 126, 869–883.
- Sonleitner, F.J. & Guthrie, J. (1991). Factors affecting oviposition
  rate in *Tribolium*. *Popul. Ecol.* 33, 1–11.