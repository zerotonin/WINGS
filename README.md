# W.I.N.G.S. — Wolbachia Infection Numerical Growth Simulation


[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://zerotonin.github.io/WINGS)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
<!-- [![PyPI](https://img.shields.io/pypi/v/wings-sim)](https://pypi.org/project/wings-sim/) -->

A spatially explicit agent-based model (ABM) of *Wolbachia* endosymbiont
spread in *Tribolium* beetle populations, with GPU acceleration via PyTorch.

## Overview

W.I.N.G.S. simulates the population dynamics of beetles infected with
*Wolbachia*, modelling four distinct phenotypic effects and their
combinations:

| Effect | Abbrev. | Mechanism |
|--------|---------|-----------|
| Cytoplasmic Incompatibility | **CI** | Infected ♂ × uninfected ♀ → embryo death |
| Male Killing | **MK** | Infected male offspring die during development |
| Increased Exploration Rate | **ER** | Infected beetles move further, increasing mate encounters |
| Increased Egg Laying | **IE** | Infected females produce ~1.2× more eggs per clutch |

The model explores all 16 combinations of these binary effects across
200 replicate simulations each, revealing how CI and ER act as
**complementary frequency-dependent mechanisms** — ER bootstraps infection
from low frequency where CI cannot operate, then CI drives to fixation
at high frequency where ER saturates.

### Models

- **GPU ABM** (`wings.models.gpu_abm`) — Spatially explicit, individual-based,
  with Lévy-flight movement, density-dependent cannibalism, and a 23-day egg
  development pipeline. Runs on CUDA GPUs via PyTorch. Supports populations
  up to 20,000+ adults.

- **Wright-Fisher Model** (`wings.models.wfm`) — Fixed-population, discrete-generation
  model for clean frequency-dynamics analysis without population-size confounds.

- **CPU ABM** (`wings.models.cpu`) — Original Python-loop implementation for
  small populations and development/debugging.

## Installation

```bash
# From source (recommended)
git clone https://github.com/zerotonin/WINGS.git
cd WINGS
pip install -e .

# With GPU support
pip install -e ".[gpu]"

# With documentation build tools
pip install -e ".[docs]"
```

### Conda (HPC clusters)

```bash
# GPU environment (CUDA 12.1)
conda env create -f envs/wings_gpu.yml
conda activate wings-gpu

# CPU-only environment
conda env create -f envs/wings_cpu.yml
conda activate wings
```

## Quick Start

### Single simulation

```bash
# GPU ABM: 50 beetles, 365 days, CI + ER enabled, save to CSV
wings-abm --population 50 --days 365 --ci --er --output result.csv

# Wright-Fisher: all 16 combinations × 200 replicates
wings-wfm --run-all --nreps 200
```

### Batch runs (SLURM)

```bash
# ABM: 16 combos × 200 reps, 10% initial infection
sbatch slurm/submit_abm.sh

# Δp frequency sweep (scans for missing, submits only what's needed)
bash slurm/submit_delta_p.sh
```

### Analysis pipeline

```bash
# 1. Combine raw CSVs into a single dataset
wings-ingest --input-dir /path/to/results --output data/combined.csv

# 2. Generate publication figures (PNG + SVG)
wings-plot --model abm --input data/combined.csv

# 3. Δp analysis (complementary CI/ER frequency dependence)
wings-ingest-dp --input-dir /path/to/delta_p --output data/combined_dp.csv
wings-plot-dp --input data/combined_dp.csv --dt 24 --mode compare
```

## Repository Structure

```
WINGS/
├── wings/                  Python package
│   ├── models/
│   │   ├── cpu/            Original CPU ABM (Beetle, Environment, Reproduction)
│   │   ├── gpu_abm.py      GPU-accelerated ABM (main simulation engine)
│   │   └── wfm.py          Wright-Fisher fixed-generation model
│   └── analysis/
│       ├── ingest.py        Data ingestion (ABM + WFM results → combined CSV)
│       ├── ingest_delta_p.py  Δp sweep ingestion
│       ├── plot_wings.py    Publication figures (time series, heatmaps, strip plots)
│       ├── plot_delta_p.py  Δp vs frequency analysis with analytical overlays
│       └── stats.py         Statistical tests (bootstrap CI, pairwise comparisons)
├── slurm/                  SLURM submission scripts for HPC clusters
├── scripts/                Entry points, demos, GPU benchmarks
├── docs/                   Sphinx documentation source
├── envs/                   Conda environment definitions
├── images/                 Project logo
├── pyproject.toml          Package metadata and dependencies
├── CITATION.cff            Machine-readable citation metadata
├── LICENSE.md              MIT License
└── README.md               This file
```

## Documentation

Full API documentation is built with Sphinx and deployed automatically
to GitHub Pages on each push to `main`:

**[https://zerotonin.github.io/WINGS](https://zerotonin.github.io/WINGS)**

To build locally:

```bash
pip install -e ".[docs]"
cd docs
sphinx-build -b html . _build/html
open _build/html/index.html
```

## Citing W.I.N.G.S.

If you use W.I.N.G.S. in your research, please cite:

> Geurten, B., Bleidorn, C., & Zare, Y. (2025). W.I.N.G.S.: Wolbachia
> Infection Numerical Growth Simulation. *In preparation.*
> https://github.com/zerotonin/WINGS

<!-- TODO: Update with DOI and journal reference when published -->

A `CITATION.cff` file is included for automated citation tools (GitHub
"Cite this repository", Zenodo, Zotero).

## License

This project is licensed under the **MIT License** — see [LICENSE.md](LICENSE.md)
for details. MIT was chosen to maximise adoption and reuse in the academic
community. If you build on this work, we appreciate (but do not require)
a citation.

## Authors

- **Bart Geurten** — University of Otago, New Zealand
- **Christoph Bleidorn** — Georg-August-Universität Göttingen, Germany
- **Yeganeh Zare**
