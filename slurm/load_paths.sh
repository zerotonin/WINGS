#!/usr/bin/env bash
# ============================================================
#  W.I.N.G.S. — load machine-specific paths into the shell
# ============================================================
#  Source this (do not execute it) before submitting jobs:
#
#     source slurm/load_paths.sh          # uses the 'hpc' profile
#     WINGS_PROFILE=local source slurm/load_paths.sh
#
#  It reads local_paths.json (git-ignored) via wings/config.py and
#  exports:
#     WINGS_DATA_ROOT  WINGS_CODE_ROOT
#     WINGS_CONDA_SETUP  WINGS_CONDA_ENV
#     SBATCH_ACCOUNT  SBATCH_PARTITION   (read by sbatch directly)
#
#  The SLURM job scripts then inherit these (sbatch --export=ALL is the
#  default), so no absolute path or account/partition is hard-coded.
# ============================================================

# Default to the HPC profile when sourced for SLURM work.
: "${WINGS_PROFILE:=hpc}"
export WINGS_PROFILE

# Locate the repo: prefer this file's own directory, fall back to an
# already-exported WINGS_CODE_ROOT (e.g. on a compute node where SLURM
# ran a spooled copy of the job script).
_lp_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
if [ -f "${_lp_dir}/../wings/config.py" ]; then
    _wings_repo="$(cd "${_lp_dir}/.." && pwd)"
elif [ -n "${WINGS_CODE_ROOT:-}" ] && [ -f "${WINGS_CODE_ROOT}/wings/config.py" ]; then
    _wings_repo="${WINGS_CODE_ROOT}"
else
    echo "load_paths.sh: cannot locate the WINGS repo (wings/config.py)." >&2
    return 1 2>/dev/null || exit 1
fi

# Find a Python to parse the JSON.  config.py is pure stdlib, so the
# system interpreter is fine even before conda is activated.
for _py in python3 python /usr/bin/python3; do
    if command -v "$_py" >/dev/null 2>&1; then _wings_py="$_py"; break; fi
done
if [ -z "${_wings_py:-}" ]; then
    echo "load_paths.sh: no python interpreter found to read local_paths.json." >&2
    return 1 2>/dev/null || exit 1
fi

_exports="$("$_wings_py" "${_wings_repo}/wings/config.py" --export --profile "${WINGS_PROFILE}")"
if [ -z "${_exports}" ]; then
    echo "load_paths.sh: no paths resolved for profile '${WINGS_PROFILE}'." >&2
    echo "  Did you copy local_paths.template.json to local_paths.json?" >&2
    return 1 2>/dev/null || exit 1
fi
eval "${_exports}"

unset _lp_dir _wings_repo _wings_py _exports _py
