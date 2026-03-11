#!/bin/bash
# ============================================================
# W.I.N.G.S. Repository Migration Script
# ============================================================
#
# Restructures the flat repository into a proper Python package.
# Run from the repository root:
#   cd /path/to/WINGS
#   bash migrate.sh
#
# Safe to run: uses git mv (preserves history), won't overwrite
# existing files, and prints what it's doing.
#
# After running:
#   git add -A
#   git commit -m "refactor: restructure into wings/ package"
# ============================================================

set -euo pipefail

# Check we're in the right place
if [ ! -f "Beetle.py" ] || [ ! -d "gpu" ]; then
    echo "ERROR: Run this from the WINGS repository root (where Beetle.py lives)"
    exit 1
fi

# Detect whether to use 'git mv' or plain 'mv'
if git rev-parse --git-dir > /dev/null 2>&1; then
    MV="git mv"
    echo "  Using 'git mv' (preserves history)"
else
    MV="mv"
    echo "  Using 'mv' (not a git repo)"
fi

echo ""
echo "============================================"
echo "  W.I.N.G.S. Repository Migration"
echo "============================================"
echo ""

# ── Step 1: Create directory structure ──────────────────────
echo "  [1/8] Creating directory structure..."

mkdir -p wings/models/cpu
mkdir -p wings/analysis
mkdir -p slurm
mkdir -p scripts
mkdir -p docs
mkdir -p envs

echo "    ✓ wings/models/cpu/"
echo "    ✓ wings/analysis/"
echo "    ✓ slurm/"
echo "    ✓ scripts/"
echo "    ✓ docs/"
echo "    ✓ envs/"

# ── Step 2: Move core model files ───────────────────────────
echo ""
echo "  [2/8] Moving core models..."

$MV Beetle.py         wings/models/cpu/beetle.py
$MV Environment.py    wings/models/cpu/environment.py
$MV Reproduction.py   wings/models/cpu/reproduction.py
$MV ParameterSet.py   wings/models/cpu/parameters.py
echo "    ✓ CPU model → wings/models/cpu/"

$MV gpu/gpu_simulation.py      wings/models/gpu_abm.py
$MV gpu/fixed_generation_sim.py wings/models/wfm.py
echo "    ✓ GPU ABM → wings/models/gpu_abm.py"
echo "    ✓ WFM    → wings/models/wfm.py"

# ── Step 3: Move analysis files ─────────────────────────────
echo ""
echo "  [3/8] Moving analysis & plotting..."

$MV ingest_data.py          wings/analysis/ingest.py
$MV gpu/ingest_delta_p.py   wings/analysis/ingest_delta_p.py
$MV plot_wings.py            wings/analysis/plot_wings.py
$MV gpu/plot_delta_p.py      wings/analysis/plot_delta_p.py

# Keep the better version of run_analysis (scripts/ has scipy.stats)
if [ -f "scripts/run_analysis.py" ]; then
    $MV scripts/run_analysis.py  wings/analysis/stats.py
fi
echo "    ✓ Analysis scripts → wings/analysis/"

# ── Step 4: Move SLURM scripts ──────────────────────────────
echo ""
echo "  [4/8] Moving SLURM scripts..."

$MV gpu/submit_wings.sh     slurm/submit_abm.sh
$MV gpu/run_abm_05.sh       slurm/submit_abm_05.sh
$MV gpu/submit_fixed.sh     slurm/submit_wfm.sh
$MV gpu/submit_delta_p.sh   slurm/submit_delta_p.sh
$MV gpu/rerun_missing.sh    slurm/rerun_missing.sh
echo "    ✓ SLURM scripts → slurm/"

# ── Step 5: Move scripts & utilities ────────────────────────
echo ""
echo "  [5/8] Moving scripts & utilities..."

# Keep the scripts/ versions (they have better defaults)
if [ -f "scripts/run_simulation.py" ]; then
    $MV scripts/run_simulation.py scripts/run_simulation.py 2>/dev/null || true
fi
if [ -f "scripts/run_multiple_simulations.py" ]; then
    $MV scripts/run_multiple_simulations.py scripts/run_batch.py
fi
$MV gpu/benchmark_gpu.py scripts/benchmark_gpu.py
echo "    ✓ Scripts → scripts/"

# Move documentation
$MV gpu/WINGS_scaling_guide.md docs/scaling_guide.md
echo "    ✓ Scaling guide → docs/"

# Move conda envs
$MV wings.yml     envs/wings_cpu.yml
$MV wings_gpu.yml envs/wings_gpu.yml
echo "    ✓ Conda envs → envs/"

# ── Step 6: Create __init__.py files ────────────────────────
echo ""
echo "  [6/8] Creating __init__.py files..."

cat > wings/__init__.py << 'PYEOF'
"""
W.I.N.G.S. — Wolbachia Infection Numerical Growth Simulation.

A spatially explicit agent-based model of Wolbachia spread in
Tribolium beetle populations, with GPU acceleration.
"""
__version__ = "0.2.0"
PYEOF

cat > wings/models/__init__.py << 'PYEOF'
"""Simulation engines: CPU ABM, GPU ABM, and Wright-Fisher model."""
PYEOF

cat > wings/models/cpu/__init__.py << 'PYEOF'
"""Original CPU-based agent-based model (per-beetle Python loops)."""
from .environment import Environment
from .beetle import Beetle
from .reproduction import Reproduction
from .parameters import ParameterSet
PYEOF

cat > wings/analysis/__init__.py << 'PYEOF'
"""Post-simulation data ingestion, analysis, and figure generation."""
PYEOF

echo "    ✓ __init__.py files created"

# ── Step 7: Fix internal imports in CPU model ───────────────
echo ""
echo "  [7/8] Fixing CPU model imports..."

# environment.py: from Reproduction import → from .reproduction import
sed -i 's/^from Reproduction import Reproduction/from .reproduction import Reproduction/' \
    wings/models/cpu/environment.py
sed -i 's/^from Beetle import Beetle/from .beetle import Beetle/' \
    wings/models/cpu/environment.py

# reproduction.py: from Beetle import → from .beetle import
sed -i 's/^from Beetle import Beetle/from .beetle import Beetle/' \
    wings/models/cpu/reproduction.py

echo "    ✓ Imports updated to relative"

# ── Step 8: Create pyproject.toml ───────────────────────────
echo ""
echo "  [8/8] Creating pyproject.toml..."

cat > pyproject.toml << 'TOMLEOF'
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "wings"
version = "0.2.0"
description = "Wolbachia Infection Numerical Growth Simulation"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.9"
authors = [
    {name = "Bart Geurten"},
    {name = "Christoph Bleidorn"},
    {name = "Yeganeh Zare"},
]
dependencies = [
    "numpy",
    "pandas",
    "matplotlib",
    "tqdm",
    "scipy",
]

[project.optional-dependencies]
gpu = ["torch>=2.0"]
dev = ["seaborn", "pytest"]

[tool.setuptools.packages.find]
include = ["wings*"]
TOMLEOF

echo "    ✓ pyproject.toml created"

# ── Update .gitignore ───────────────────────────────────────
cat > .gitignore << 'GIEOF'
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/

# Data (large CSVs not in repo)
data/*.csv
*.csv

# Figures (regenerated from data)
figures*/
*.log

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
GIEOF

echo ""
echo "    ✓ .gitignore updated"

# ── Fix SLURM script paths ──────────────────────────────────
echo ""
echo "  [9/9] Fixing SLURM script paths..."

# gpu/gpu_simulation.py → wings/models/gpu_abm.py
sed -i 's|gpu/gpu_simulation\.py|wings/models/gpu_abm.py|g' slurm/*.sh
# gpu/fixed_generation_sim.py → wings/models/wfm.py
sed -i 's|gpu/fixed_generation_sim\.py|wings/models/wfm.py|g' slurm/*.sh

echo "    ✓ SLURM scripts now point to wings/models/"

# ── Report superfluous files to delete ──────────────────────
echo ""
echo "============================================"
echo "  Migration complete!"
echo "============================================"
echo ""
echo "  The following SUPERFLUOUS files should be deleted:"
echo "  (review, then delete manually or run the commands below)"
echo ""

SUPERFLUOUS=(
    "ingest_data_old.py"
    "read_results.py"
    "plot_results.py"
    "run_multiple_simulations.py"
    "run_analysis.py"
    "scripts/run_plots.py"
    "scripts/run_ingestion.py"
    "scripts/run_ana_plots.sh"
    "data/diff_names.sh"
)

for f in "${SUPERFLUOUS[@]}"; do
    if [ -e "$f" ]; then
        echo "    $MV $f → DELETE"
    fi
done

echo ""
echo "  To delete them:"
echo "    git rm ${SUPERFLUOUS[*]}"
echo ""
echo "  Clean up empty gpu/ directory:"
echo "    rm -rf gpu/__pycache__"
echo "    rmdir gpu 2>/dev/null || echo '  gpu/ not empty, check manually'"
echo ""
echo "  Then commit:"
echo "    git add -A"
echo "    git commit -m 'refactor: restructure into wings/ Python package'"
echo ""
echo "  New structure:"
echo "    wings/models/cpu/     — CPU ABM (beetle, environment, reproduction, parameters)"
echo "    wings/models/         — GPU ABM (gpu_abm.py) + WFM (wfm.py)"
echo "    wings/analysis/       — ingestion, plotting, statistics"
echo "    slurm/                — all SLURM submission scripts"
echo "    scripts/              — entry points, demos, benchmarks"
echo "    docs/                 — documentation"
echo "    envs/                 — conda environment definitions"
echo "============================================"
