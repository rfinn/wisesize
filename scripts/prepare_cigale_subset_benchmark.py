#!/usr/bin/env python
"""Prepare a smaller CIGALE benchmark on local Draco storage."""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path("/mnt/astro/SGA-CIGALE")
DEFAULT_DESTINATION_ROOT = Path("/data-pool/rfinn/SGA-CIGALE")
DEFAULT_SOURCE_RUN = "wisesize_sga2025_ap03_z0002_0025_w3snr10_chunk01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a CIGALE subset benchmark using an existing input table and "
            "run configuration."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"Existing SGA-CIGALE root (default: {DEFAULT_SOURCE_ROOT}).",
    )
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=DEFAULT_DESTINATION_ROOT,
        help=f"Benchmark root (default: {DEFAULT_DESTINATION_ROOT}).",
    )
    parser.add_argument(
        "--source-run",
        "--config-run",
        dest="source_run",
        default=DEFAULT_SOURCE_RUN,
        help=(
            "Existing run directory from which to copy the configuration "
            f"(default: {DEFAULT_SOURCE_RUN})."
        ),
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help=(
            "CIGALE input table to subset. By default, use the data_file from "
            "the selected configuration."
        ),
    )
    parser.add_argument(
        "--n-objects",
        type=int,
        default=500,
        help="Number of input objects to retain (default: 500).",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=8,
        help="Number of CIGALE worker processes to configure (default: 8).",
    )
    parser.add_argument(
        "--name",
        help="Benchmark name. By default, append _firstN_localdisk to SOURCE_RUN.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a prepared benchmark if no CIGALE out directory exists.",
    )
    return parser.parse_args()


def config_value(config_text: str, key: str) -> str:
    match = re.search(
        rf"^{re.escape(key)}\s*=\s*(.+?)\s*$", config_text, re.MULTILINE
    )
    if match is None:
        raise KeyError(f"Configuration has no {key!r} setting")
    return match.group(1)


def replace_config_value(config_text: str, key: str, value: str) -> str:
    pattern = rf"^({re.escape(key)}\s*=\s*).*$"
    updated, count = re.subn(
        pattern, rf"\g<1>{value}", config_text, flags=re.MULTILINE
    )
    if count != 1:
        raise ValueError(f"Expected one {key!r} setting, found {count}")
    return updated


def copy_input_subset(source: Path, destination: Path, n_objects: int) -> None:
    if n_objects < 1:
        raise ValueError("--n-objects must be positive")

    with source.open("r", encoding="utf-8") as source_handle:
        header = source_handle.readline()
        if not header.startswith("# id "):
            raise ValueError(f"Unexpected CIGALE input header in {source}")
        rows = []
        for _ in range(n_objects):
            row = source_handle.readline()
            if not row:
                break
            rows.append(row)

    if len(rows) != n_objects:
        raise ValueError(
            f"Requested {n_objects} objects, but {source} contains only {len(rows)}"
        )
    with destination.open("w", encoding="utf-8") as destination_handle:
        destination_handle.write(header)
        destination_handle.writelines(rows)


def write_launcher(path: Path, root: Path, run_name: str, cores: int) -> None:
    run_dir = root / "cigale_runs" / run_name
    cache_dir = root / "cache"
    log_name = f"run-cores{cores}-{run_name}.log"
    launcher = f"""#!/usr/bin/env bash
set -euo pipefail

export XDG_CACHE_HOME={cache_dir}
export MPLCONFIGDIR={cache_dir / 'matplotlib'}
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

cd {run_dir}
if [[ -e out ]]; then
    echo "Refusing to overwrite existing CIGALE output: $PWD/out" >&2
    exit 1
fi

PCIGALE=${{PCIGALE:-pcigale}}
"$PCIGALE" check
/usr/bin/time -v "$PCIGALE" run 2>&1 | tee {log_name}
"""
    path.write_text(launcher, encoding="utf-8")
    path.chmod(0o755)


def main() -> None:
    args = parse_args()
    if args.cores < 1:
        raise ValueError("--cores must be positive")
    run_name = args.name or f"{args.source_run}_first{args.n_objects}_localdisk"
    source_run_dir = args.source_root / "cigale_runs" / args.source_run
    source_config = source_run_dir / "pcigale.ini"
    source_spec = source_run_dir / "pcigale.ini.spec"

    if not source_config.is_file() or not source_spec.is_file():
        raise FileNotFoundError(f"Missing CIGALE configuration under {source_run_dir}")

    config_text = source_config.read_text(encoding="utf-8")
    if args.input_file is None:
        source_data_setting = config_value(config_text, "data_file")
        source_data = (source_run_dir / source_data_setting).resolve()
    else:
        source_data = args.input_file.expanduser().resolve()
    if not source_data.is_file():
        raise FileNotFoundError(f"Missing source input table: {source_data}")

    destination_input_dir = args.destination_root / "inputs"
    destination_run_dir = args.destination_root / "cigale_runs" / run_name
    destination_input = destination_input_dir / f"{run_name}.dat"
    destination_config = destination_run_dir / "pcigale.ini"
    destination_spec = destination_run_dir / "pcigale.ini.spec"
    launcher = destination_run_dir / "run-benchmark.sh"

    if destination_run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Benchmark already exists: {destination_run_dir}; use --overwrite"
            )
        if (destination_run_dir / "out").exists():
            raise FileExistsError(
                f"Refusing to overwrite completed output: {destination_run_dir / 'out'}"
            )
        shutil.rmtree(destination_run_dir)

    destination_input_dir.mkdir(parents=True, exist_ok=True)
    destination_run_dir.mkdir(parents=True)
    copy_input_subset(source_data, destination_input, args.n_objects)

    relative_input = os.path.relpath(destination_input, destination_run_dir)
    updated_config = replace_config_value(config_text, "data_file", relative_input)
    updated_config = replace_config_value(updated_config, "cores", str(args.cores))
    destination_config.write_text(updated_config, encoding="utf-8")
    shutil.copy2(source_spec, destination_spec)
    write_launcher(launcher, args.destination_root, run_name, args.cores)

    print(f"Prepared {args.n_objects:,} objects at {destination_run_dir}")
    print(f"Input: {destination_input}")
    print(f"Run with: {launcher}")


if __name__ == "__main__":
    main()
