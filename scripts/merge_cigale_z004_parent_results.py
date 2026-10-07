#!/usr/bin/env python3
"""Merge the low-z and redshift-shell CIGALE catalogs into Z < 0.04."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack


DEFAULT_SGA_FITS = Path("/Users/rfinn/research/SGA2025/SGA2025-v1.0.fits")
DEFAULT_RESULTS_ROOT = Path("/Users/rfinn/research/SGA-CIGALE/results")
DEFAULT_LOWZ_RESULTS = DEFAULT_RESULTS_ROOT / (
    "sga2025_ap04_zlt0025_parent_errfloor0p10mag_"
    "agnfrac0to0p5_results.fits"
)
DEFAULT_SHELL_RESULTS = DEFAULT_RESULTS_ROOT / (
    "sga2025_ap04_z0025_0040_shell_errfloor0p10mag_agnfrac0to0p5_"
    "chunks_01_02_03_04_05_06_results.fits"
)
DEFAULT_OUTPUT = DEFAULT_RESULTS_ROOT / (
    "sga2025_ap04_zlt0040_parent_errfloor0p10mag_"
    "agnfrac0to0p5_results.fits"
)
DEFAULT_SPLIT_Z = 0.025
DEFAULT_Z_MAX = 0.04
SGA_COLUMNS = (
    "SGANAME",
    "GALAXY",
    "OBJNAME",
    "RA",
    "DEC",
    "Z",
    "Z_COSMO",
    "DIST",
    "DIST_REF",
    "DIST_METHOD",
    "Z_REF",
    "Z_FLAG",
    "SAMPLE",
    "D26",
    "GROUP_NAME",
    "GROUP_PRIMARY",
    "GROUP_RA",
    "GROUP_DEC",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge the enriched Z < split catalog with the raw split <= Z < "
            "z_max shell, validate membership, and attach SGA metadata."
        )
    )
    parser.add_argument("--sga-fits", type=Path, default=DEFAULT_SGA_FITS)
    parser.add_argument("--lowz-results", type=Path, default=DEFAULT_LOWZ_RESULTS)
    parser.add_argument(
        "--shell-results", type=Path, default=DEFAULT_SHELL_RESULTS
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-z", type=float, default=DEFAULT_SPLIT_Z)
    parser.add_argument("--z-max", type=float, default=DEFAULT_Z_MAX)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def source_rows_for_ids(sga_ids: np.ndarray, result_ids: np.ndarray) -> np.ndarray:
    sorter = np.argsort(sga_ids)
    positions = np.searchsorted(sga_ids, result_ids, sorter=sorter)
    if np.any(positions == len(sga_ids)):
        raise ValueError("At least one result SGAID is absent from SGA.")
    rows = sorter[positions]
    if not np.array_equal(sga_ids[rows], result_ids):
        raise ValueError("At least one result SGAID is absent from SGA.")
    return rows


def enrich_shell(shell: Table, sga: fits.FITS_rec, sga_ids: np.ndarray) -> Table:
    output = shell.copy(copy_data=True)
    rows = source_rows_for_ids(sga_ids, np.asarray(output["SGAID"], np.int64))
    output.add_column(
        np.full(len(output), "shell", dtype="U10"),
        index=1,
        name="CIGALE_SAMPLE",
    )
    output.add_column(
        np.zeros(len(output), dtype=bool),
        index=2,
        name="WISESIZE_SELECTED",
    )
    for column in reversed(SGA_COLUMNS):
        output.add_column(np.array(sga[column][rows]), index=1, name=column)
    return output


def main() -> None:
    args = parse_args()
    for path in (args.sga_fits, args.lowz_results, args.shell_results):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not 0.0 < args.split_z < args.z_max:
        raise ValueError("Require 0 < --split-z < --z-max.")

    lowz = Table.read(args.lowz_results)
    shell_raw = Table.read(args.shell_results)
    for name, table in (("low-z", lowz), ("shell", shell_raw)):
        if "SGAID" not in table.colnames:
            raise ValueError(f"{name} results do not contain SGAID.")
        if len(np.unique(table["SGAID"])) != len(table):
            raise ValueError(f"{name} results contain duplicate SGAID values.")

    overlap = np.intersect1d(lowz["SGAID"], shell_raw["SGAID"])
    if len(overlap):
        raise ValueError(f"Input catalogs overlap by {len(overlap)} SGAID values.")

    with fits.open(args.sga_fits, memmap=True) as hdul:
        sga = hdul[1].data
        sga_ids = np.asarray(sga["SGAID"], dtype=np.int64)
        if len(np.unique(sga_ids)) != len(sga_ids):
            raise ValueError("SGA catalog contains duplicate SGAID values.")
        shell = enrich_shell(shell_raw, sga, sga_ids)

        if lowz.colnames != shell.colnames:
            raise ValueError("Enriched low-z and shell result schemas differ.")
        if np.any(np.asarray(lowz["Z"], float) >= args.split_z):
            raise ValueError("Low-z catalog contains rows at or above split-z.")
        shell_z = np.asarray(shell["Z"], float)
        if np.any((shell_z < args.split_z) | (shell_z >= args.z_max)):
            raise ValueError("Shell catalog contains rows outside its interval.")

        combined = vstack(
            [lowz, shell], join_type="exact", metadata_conflicts="silent"
        )
        redshift = np.asarray(sga["Z"], dtype=np.float64)
        expected = np.isfinite(redshift) & (redshift != 0.0) & (redshift < args.z_max)
        expected_ids = sga_ids[expected]
        missing = np.setdiff1d(expected_ids, combined["SGAID"])
        extra = np.setdiff1d(combined["SGAID"], expected_ids)
        if len(missing) or len(extra):
            raise ValueError(
                "Merged results do not match the requested SGA parent: "
                f"{len(missing)} missing and {len(extra)} extra."
            )

    combined.sort("Z")
    combined.meta.clear()
    combined.meta["ZMAX"] = (args.z_max, "Strict upper bound on measured SGA Z")
    combined.meta["ZNONZERO"] = (True, "SGA Z=0 missing-redshift rows excluded")
    combined.meta["ZSPLIT"] = (args.split_z, "Boundary between CIGALE runs")
    combined.meta["NLOWZ"] = (len(lowz), "Rows in lower-redshift tier")
    combined.meta["NSHELL"] = (len(shell), "Rows in upper-redshift shell")
    combined.meta["SGAFILE"] = args.sga_fits.name

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.write(args.output, overwrite=args.overwrite)
    print(f"Low-z rows: {len(lowz):,}")
    print(f"Shell rows: {len(shell):,}")
    print(f"Merged parent rows: {len(combined):,}")
    print(f"Measured Z range: {np.min(combined['Z']):.8f} to {np.max(combined['Z']):.8f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
