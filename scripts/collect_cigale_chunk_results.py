#!/usr/bin/env python
"""Validate and combine CIGALE chunk results into one SGA-keyed catalog."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from astropy.table import Column, Table, vstack


DEFAULT_ROOT = Path("/data-pool/rfinn/SGA-CIGALE")
DEFAULT_STEM = "wisesize_sga2025_ap03_z0002_0025_w3snr10_errfloor0p10mag"
CHUNK_RE = re.compile(r"_chunk(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine CIGALE results.fits files, expose the input ID as SGAID, "
            "and verify every result against its chunk input."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"SGA-CIGALE data root (default: {DEFAULT_ROOT}).",
    )
    parser.add_argument(
        "--stem",
        default=DEFAULT_STEM,
        help=f"Common sample/run name (default: {DEFAULT_STEM}).",
    )
    parser.add_argument(
        "--chunks",
        nargs="+",
        type=int,
        help="Specific chunk numbers to collect. By default, collect all run directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output FITS path. A path under ROOT/results is used by default.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output catalog.",
    )
    return parser.parse_args()


def chunk_number(path: Path, stem: str) -> int:
    match = CHUNK_RE.search(path.name)
    if match is None or not path.name.startswith(f"{stem}_chunk"):
        raise ValueError(f"Cannot determine chunk number from {path}")
    return int(match.group(1))


def find_run_directories(
    root: Path, stem: str, requested_chunks: list[int] | None
) -> list[Path]:
    run_root = root / "cigale_runs"
    if requested_chunks:
        paths = [run_root / f"{stem}_chunk{number:02d}" for number in requested_chunks]
    else:
        paths = sorted(
            path
            for path in run_root.glob(f"{stem}_chunk*")
            if path.is_dir() and CHUNK_RE.search(path.name)
        )

    if not paths:
        raise FileNotFoundError(f"No run directories found for {stem} under {run_root}")

    missing = [path for path in paths if not path.is_dir()]
    if missing:
        joined = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing run directories:\n  {joined}")

    numbers = [chunk_number(path, stem) for path in paths]
    if len(numbers) != len(set(numbers)):
        raise ValueError(f"Duplicate chunk numbers requested: {numbers}")
    return [path for _, path in sorted(zip(numbers, paths, strict=True))]


def as_integer_ids(values: object, label: str) -> np.ndarray:
    array = np.ma.asarray(values)
    if np.ma.isMaskedArray(array) and np.any(np.ma.getmaskarray(array)):
        raise ValueError(f"{label} contains masked IDs")

    try:
        numeric = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} contains non-numeric IDs") from error

    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{label} contains non-finite IDs")
    rounded = np.rint(numeric)
    if not np.all(numeric == rounded):
        raise ValueError(f"{label} contains non-integer IDs")

    ids = rounded.astype(np.int64)
    unique, counts = np.unique(ids, return_counts=True)
    duplicates = unique[counts > 1]
    if len(duplicates):
        preview = ", ".join(str(value) for value in duplicates[:10])
        raise ValueError(f"{label} contains duplicate IDs: {preview}")
    return ids


def read_input_ids(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CIGALE input table: {path}")
    table = Table.read(path, format="ascii.commented_header", guess=False)
    if "id" not in table.colnames:
        raise KeyError(f"Input table has no 'id' column: {path}")
    return as_integer_ids(table["id"], f"input {path}")


def read_result_chunk(root: Path, run_dir: Path, stem: str) -> Table:
    number = chunk_number(run_dir, stem)
    result_path = run_dir / "out" / "results.fits"
    input_path = root / "inputs" / f"{run_dir.name}.dat"
    if not result_path.is_file():
        raise FileNotFoundError(
            f"Chunk {number:02d} has no completed result table: {result_path}"
        )

    input_ids = read_input_ids(input_path)
    result = Table.read(result_path)
    if "id" not in result.colnames:
        raise KeyError(f"CIGALE result has no 'id' column: {result_path}")
    result_ids = as_integer_ids(result["id"], f"result {result_path}")

    missing = np.setdiff1d(input_ids, result_ids)
    unexpected = np.setdiff1d(result_ids, input_ids)
    if len(missing) or len(unexpected):
        raise ValueError(
            f"Chunk {number:02d} ID mismatch: {len(missing)} missing results, "
            f"{len(unexpected)} unexpected results"
        )
    if len(result) != len(input_ids):
        raise ValueError(
            f"Chunk {number:02d} row-count mismatch: {len(input_ids)} inputs, "
            f"{len(result)} results"
        )

    result = result.copy(copy_data=True)
    result.add_column(Column(result_ids, name="SGAID"), index=0)
    result.add_column(
        Column(np.full(len(result), number, dtype=np.int16), name="CIGALE_CHUNK"),
        index=1,
    )
    print(f"Chunk {number:02d}: validated {len(result):,} rows")
    return result


def default_output_path(
    root: Path, stem: str, requested_chunks: list[int] | None
) -> Path:
    if requested_chunks:
        chunk_label = "_".join(f"{number:02d}" for number in sorted(requested_chunks))
        name = f"{stem}_chunks_{chunk_label}_results.fits"
    else:
        name = f"{stem}_results.fits"
    return root / "results" / name


def main() -> None:
    args = parse_args()
    run_directories = find_run_directories(args.root, args.stem, args.chunks)
    tables = [
        read_result_chunk(args.root, run_dir, args.stem)
        for run_dir in run_directories
    ]
    combined = vstack(tables, join_type="exact", metadata_conflicts="silent")
    combined_ids = as_integer_ids(combined["SGAID"], "combined result")
    combined["SGAID"] = combined_ids
    combined.sort("SGAID")

    output = args.output or default_output_path(args.root, args.stem, args.chunks)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.write(output, overwrite=args.overwrite)
    print(f"Wrote {len(combined):,} rows to {output}")


if __name__ == "__main__":
    main()
