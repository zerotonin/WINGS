"""
W.I.N.G.S. — machine-specific path resolution.

No absolute paths live in the source tree.  Instead each machine keeps a
``local_paths.json`` (git-ignored) next to the repo root, copied from the
committed ``local_paths.template.json`` and filled in.  The file defines
named *profiles* — typically ``local`` (a workstation) and ``hpc`` (an
HPC cluster such as Aoraki) — each holding a ``data_root``, ``code_root``,
conda settings, and (for HPC) the SLURM account and partition.

Resolution order for any key (highest priority first):

    1. Environment variable  WINGS_<KEY>   (e.g. WINGS_DATA_ROOT)
    2. local_paths.json  →  profiles[<profile>][<key>]
    3. The supplied default

The active profile is taken from ``WINGS_PROFILE`` if set, otherwise the
``active_profile`` field in local_paths.json, otherwise ``"local"``.

This module imports nothing outside the standard library, so it can be
run as a script (``python wings/config.py --export``) on a login node
before the package or conda environment is available — that is how the
bash SLURM helpers read their paths.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

# Keys that may be overridden per machine.  Each maps to the WINGS_<KEY>
# environment variable (upper-cased) and to a field in local_paths.json.
PATH_KEYS: tuple[str, ...] = (
    "data_root",
    "code_root",
    "conda_setup",
    "conda_env",
    "slurm_account",
    "slurm_partition",
)

TEMPLATE_NAME = "local_paths.template.json"
LOCAL_NAME = "local_paths.json"


# ======================================================================
#  Locating the repo and the per-machine file
# ======================================================================

def repo_root() -> Path:
    """Return the repository root (the directory that holds ``wings/``)."""
    return Path(__file__).resolve().parent.parent


def local_paths_file() -> Path:
    """Path to the (git-ignored) per-machine ``local_paths.json``."""
    return repo_root() / LOCAL_NAME


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


_warned_missing = False


def is_configured() -> bool:
    """True if a per-machine ``local_paths.json`` exists."""
    return local_paths_file().is_file()


def warn_if_missing() -> None:
    """Print a one-time notice to stderr when local_paths.json is absent.

    Top-level entry points fall back to repo-relative defaults when the
    file is missing; this makes that situation visible instead of silent.
    """
    global _warned_missing
    if not _warned_missing and not is_configured():
        _warned_missing = True
        print(
            f"WINGS: {LOCAL_NAME} not found — falling back to defaults. "
            f"Copy {TEMPLATE_NAME} to {LOCAL_NAME} and fill it in "
            f"(see the Configuration section of the README).",
            file=sys.stderr,
        )


def load_config() -> dict:
    """Load local_paths.json, or {} if it has not been created yet."""
    return _load_json(local_paths_file())


# ======================================================================
#  Profile selection and key resolution
# ======================================================================

def active_profile(cfg: dict | None = None) -> str:
    """Resolve the active profile name (env > file > ``"local"``)."""
    if "WINGS_PROFILE" in os.environ:
        return os.environ["WINGS_PROFILE"]
    cfg = load_config() if cfg is None else cfg
    return cfg.get("active_profile", "local")


def _resolve_path_value(value: str) -> str:
    """Expand ``~`` and make repo-relative paths absolute."""
    expanded = os.path.expanduser(value)
    p = Path(expanded)
    if not p.is_absolute():
        p = repo_root() / p
    return str(p)


def get(key: str, profile: str | None = None, default: str | None = None) -> str | None:
    """Resolve a single configuration key.

    Args:
        key:     One of :data:`PATH_KEYS` (e.g. ``"data_root"``).
        profile: Profile to read; defaults to :func:`active_profile`.
        default: Returned when neither env nor file provides the key.
    """
    env_name = f"WINGS_{key.upper()}"
    if env_name in os.environ:
        value = os.environ[env_name]
    else:
        warn_if_missing()
        cfg = load_config()
        profile = profile or active_profile(cfg)
        value = cfg.get("profiles", {}).get(profile, {}).get(key, default)

    if value is None:
        return None
    if key.endswith("_root") or key.endswith("_setup"):
        return _resolve_path_value(value)
    return value


def data_root(profile: str | None = None) -> Path:
    """Resolve the results/data root, falling back to ``<repo>/data``."""
    return Path(get("data_root", profile=profile, default="data"))


def code_root(profile: str | None = None) -> Path:
    """Resolve the WINGS checkout root, falling back to the repo root."""
    value = get("code_root", profile=profile)
    return Path(value) if value else repo_root()


def output_dir(name: str, profile: str | None = None) -> Path:
    """Return ``<data_root>/<name>`` for a named results sub-directory."""
    return data_root(profile=profile) / name


# ======================================================================
#  Shell interface (consumed by slurm/load_paths.sh)
# ======================================================================

def _export_lines(profile: str) -> list[str]:
    lines = [f"export WINGS_PROFILE={shlex.quote(profile)}"]
    for key in PATH_KEYS:
        value = get(key, profile=profile)
        if value is None:
            continue
        lines.append(f"export WINGS_{key.upper()}={shlex.quote(value)}")
    # Mirror the SLURM settings onto the names sbatch reads directly.
    account = get("slurm_account", profile=profile)
    partition = get("slurm_partition", profile=profile)
    if account:
        lines.append(f"export SBATCH_ACCOUNT={shlex.quote(account)}")
    if partition:
        lines.append(f"export SBATCH_PARTITION={shlex.quote(partition)}")
    return lines


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WINGS path resolver")
    parser.add_argument("--profile", default=None,
                        help="Profile name (default: active profile)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--export", action="store_true",
                       help="Emit 'export VAR=...' lines for the shell")
    group.add_argument("--get", metavar="KEY",
                       help="Print a single resolved key")
    args = parser.parse_args(argv)

    profile = args.profile or active_profile()
    warn_if_missing()

    if args.get:
        value = get(args.get, profile=profile)
        if value is None:
            return 1
        print(value)
        return 0

    if args.export:
        print("\n".join(_export_lines(profile)))
        return 0

    # Default: human-readable dump of the active profile.
    print(f"profile: {profile}")
    for key in PATH_KEYS:
        print(f"  {key:16s} {get(key, profile=profile)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
