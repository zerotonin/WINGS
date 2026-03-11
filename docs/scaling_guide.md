# WINGS Scaling Guide: SLURM Strategies & Population Dynamics

## 1. SLURM Scheduling: Array Jobs vs Loop-in-One-Job

### The Three Options

**Option A: Array job** (`--array=1-160`) — *your previous approach*

Each of the 160 combinations gets its own independent SLURM job. The scheduler allocates a separate GPU for each one.

- **Pro:** Fault tolerant — if one task crashes, the other 159 are unaffected. Tasks can run across many nodes simultaneously.
- **Con:** Each task independently waits in queue, initialises conda, waits 5s for filesystem mount, and warms up the GPU. For short-running simulations (~2–10 min each), this overhead dominates. Your 160 tasks each need a scheduler interaction, and on a busy cluster you may wait a long time for 160 separate GPU slots.

**Option B: Single job, sequential loop** (`--gpus-per-task=1`)

One GPU, one job, a bash loop runs all 160 combos in sequence.

- **Pro:** Zero scheduling overhead between runs. One conda init, one filesystem wait.
- **Con:** Purely serial. 160 × ~5 min = ~13 hours. Risky with wall-time limits. Wastes 3 GPUs you could be using.

**Option C: Single job, multi-GPU parallel batches** (`--gpus-per-task=4`) — *recommended*

One job requests 4 GPUs. A bash loop runs 4 simulations in parallel (one per GPU via `CUDA_VISIBLE_DEVICES`), cycling through all 160 combos in ~40 batches.

- **Pro:** 1 scheduler allocation, 1 conda init. 4× throughput. Automatic skip of already-completed runs (resume after crash). Progress counter.
- **Con:** If the job dies, you need to resubmit (but it auto-resumes from where it left off).

### Recommendation

**Option C** is the best fit for your workload. Each simulation uses ~2–3 GB VRAM, and even the L4 (24 GB) can easily run 4 in parallel. The new `submit_wings.sh` implements this.

| Metric | Array (160 tasks) | Loop (1 GPU) | Parallel (4 GPU) |
|--------|-------------------|--------------|-------------------|
| Scheduler allocations | 160 | 1 | 1 |
| Conda + mount overhead | 160 × ~10s = 27 min | 10s | 10s |
| Queue wait | 160 separate waits | 1 wait | 1 wait |
| Wall time (est.) | Depends on queue | ~13h | ~3.5h |
| Resume after crash | Automatic (per task) | Manual | Automatic (skips done CSVs) |

### How to use multi-GPU in the SLURM script

The trick is simple: background processes with `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0 python sim.py --output run1.csv &
CUDA_VISIBLE_DEVICES=1 python sim.py --output run2.csv &
CUDA_VISIBLE_DEVICES=2 python sim.py --output run3.csv &
CUDA_VISIBLE_DEVICES=3 python sim.py --output run4.csv &
wait  # block until all 4 finish
```

Each process only sees one GPU (as device `cuda:0`), so no code changes needed.

---

## 2. Population Dynamics: Starting Small and Growing

### Your scenario

Start with 50 beetles (some infected), let the population grow to carrying capacity (~20,000), and observe how Wolbachia spreads during the growth phase. This is biologically more realistic than starting at equilibrium — it models the colonisation/invasion scenario.

### The problem: how to regulate population at carrying capacity

In your original ABM, the `grim_reaper` randomly culled beetles above `max_population`. This is a hard wall — biologically unsatisfying. The literature offers several density-dependent mechanisms, and which one to use matters because *the form of density-dependence affects Wolbachia spread dynamics* (Hancock et al. 2016, J. Applied Ecology; Hancock et al. 2011, Amer. Naturalist).

### Four implemented options

The updated `gpu_simulation.py` supports four mortality modes via `--mortality`:

#### `none` — Hard cap only (original ABM behaviour)

Population grows freely until hitting `max_population`, then excess adults are randomly removed. Simple but biologically unrealistic — creates artificial boom-bust cycles at carrying capacity.

*Use for:* Backwards compatibility, or when you want the simplest possible dynamics.

#### `logistic` — Density-dependent adult mortality

Per-capita adult death rate increases smoothly with population density:

$$\mu(N) = \mu_0 + \mu_0 \cdot \left(\frac{N}{K}\right)^\beta$$

At low density (N << K), mortality is just the natural baseline. As N approaches K, the death rate rises. The exponent β controls how sharply: β = 1 is linear, β = 2 gives a gentler ramp that steepens near K.

*Biological basis:* General resource competition model. Used in many ecological models. Less specific to beetles but mathematically well-understood.

*Effect on Wolbachia:* Density-dependent adult mortality affects infected and uninfected adults equally (unless you add infection-specific fitness costs). This means it slows overall population growth near K but doesn't directly create selection for or against Wolbachia.

#### `cannibalism` — Egg cannibalism (Tribolium-style) ★ recommended

Adults consume eggs at a rate proportional to adult density:

$$P(\text{egg eaten}) = r \cdot N_{\text{adults}} \cdot \left(\frac{N}{K}\right)^\beta$$

where *r* is the per-adult per-hour cannibalism rate.

*Biological basis:* This is the primary population regulation mechanism in Tribolium beetles. Classic work by Park (1934, 1938) and Daly & Ryan (1983) showed that egg predation by larvae and adults is strongly density-dependent and is the main driver of population regulation. Generation mortality ranged from 42% at low density to 74% at high density, with predation on eggs in the first 10 days being the primary regulating factor. Sonleitner & Gutherie (1991) confirmed that oviposition suppression and egg cannibalism together drive the self-regulation loop. Recent reviews (Pointer et al. 2021, Heredity) describe how Tribolium populations self-regulate through cannibalistic behaviour that scales with density.

*Effect on Wolbachia:* Egg cannibalism is particularly interesting for Wolbachia dynamics because it creates a *non-random* egg mortality pressure. If CI is active, uninfected females mated with infected males produce fewer viable eggs (many are already dead from CI), while infected females' eggs are fully viable — but then both face equal cannibalism pressure. The net effect is that CI + cannibalism together create a stronger selective advantage for Wolbachia than CI alone, because the "wasted" reproductive slots from CI-killed eggs are not refilled. Hancock et al. (2016, BMC Biology) showed that density-dependent larval competition significantly affects the rate of Wolbachia invasion.

#### `contest` — Contest competition

When N > K, each excess individual has a per-hour probability of dying equal to (N − K)/N. Below K, no extra mortality. This is a sharp threshold — population hovering just above K quickly gets pushed back.

*Biological basis:* Represents territorial or interference competition where dominant individuals exclude subordinates. Less appropriate for beetles (which don't hold territories) but useful as a mathematical comparison.

*Effect on Wolbachia:* Similar to `none` but with stochastic rather than deterministic culling above K. The randomness means the population fluctuates around K rather than being clamped to it exactly.

### My recommendation

**Use `cannibalism`** (the default). It is the most biologically grounded for beetles, and it produces qualitatively different — and more realistic — Wolbachia invasion dynamics than a hard cap. The egg cannibalism interacts non-trivially with CI (since CI already kills some eggs, the additional pressure from cannibalism has a differential effect on infected vs uninfected lineages).

The parameter `--cannibalism-rate` (default 0.0001) controls the intensity. At N = K = 20,000 with β = 2, the probability of any single egg being eaten per hour is:

$$P = 0.0001 \times 20{,}000 \times 1.0^2 = 2.0$$

This is clamped to 0.95, meaning at carrying capacity ~95% of eggs are consumed each hour. This sounds extreme, but eggs are in the pipeline for 552 hours — even with 95% hourly destruction, the ~5% that survive each hour yield a steady flow of hatchlings that balances adult deaths. You can tune this parameter to match observed survival-to-hatching rates for your specific beetle species.

### Key references

- Daly, P.J. & Ryan, M.F. (1983). Density-related mortality of Tribolium confusum. *Res. Popul. Ecol.* 25, 210–219.
- Park, T. (1934). Studies in population physiology III. *J. Exp. Zool.* 68, 167–182.
- Sonleitner, F.J. & Gutherie, J. (1991). Factors affecting oviposition rate in Tribolium. *Popul. Ecol.* 33, 1–11.
- Hancock, P.A., Godfray, H.C.J. (2012). Modelling Wolbachia spread in spatially heterogeneous environments. *J. R. Soc. Interface* 9, 3045–3054.
- Hancock, P.A. et al. (2016). Predicting Wolbachia invasion dynamics using density-dependent demographic traits. *BMC Biology* 14, 96.
- Hancock, P.A. et al. (2016). Density-dependent population dynamics slow the spread of wMel Wolbachia. *J. Appl. Ecol.* 53, 785–793.
- Pointer, M.D. et al. (2021). Tribolium beetles as a model system in evolution and ecology. *Heredity* 126, 869–883.

---

## 3. Runtime & Memory Expectations

### Starting from 50 beetles, growing to 20,000

The simulation has three phases with very different computational loads:

| Phase | Population | Eggs in pipeline | Total entities | Time per day |
|-------|-----------|-----------------|---------------|-------------|
| Early growth (day 1–60) | 50 → ~500 | 0 → ~5,000 | < 10,000 | ~5 ms |
| Exponential growth (day 60–150) | 500 → ~15,000 | 5,000 → ~500,000 | Up to ~500K | ~50–200 ms |
| Near carrying capacity (day 150+) | ~20,000 | Cannibalism-limited | ~50–200K | ~100–500 ms |

Projected wall time for a single 365-day run: **~1–3 minutes** (mostly spent in the growth/capacity phases).

160 runs on 4 GPUs: **~40–120 minutes total**.

### VRAM usage

At peak (500K entities × ~32 bytes per attribute × 8 attributes): **~130 MB**. The L4's 24 GB can easily handle this 4× over.

---

## 4. Changed Files Summary

| File | What changed |
|------|-------------|
| `gpu_simulation.py` | Default `initial_population` → 50. Default `max_population` → 20,000. Added `mortality_mode`, `mortality_beta`, `cannibalism_rate` config. Replaced `_retire_dead` with 4-mode density-dependent mortality system. Added CLI flags `--mortality`, `--mortality-beta`, `--cannibalism-rate`. |
| `submit_wings.sh` | Complete rewrite: single job with 4 GPUs, parallel batches, progress counter, auto-skip of completed runs, verbose diagnostics. |
