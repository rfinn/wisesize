#!/usr/bin/env python3
"""Merge WISEsize and fill-in CIGALE results into the low-z SGA parent."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack


DEFAULT_SGA_FITS = Path("/Users/rfinn/research/SGA2025/SGA2025-v1.0.fits")
DEFAULT_RESULTS_ROOT = Path("/Users/rfinn/research/SGA-CIGALE/results")
DEFAULT_WISESIZE_RESULTS = DEFAULT_RESULTS_ROOT / (
    "wisesize_sga2025_ap04_z0002_0025_w3snr10_errfloor0p10mag_"
    "agnfrac0to0p5_results.fits"
)
DEFAULT_COMPLEMENT_RESULTS = DEFAULT_RESULTS_ROOT / (
    "sga2025_ap04_z0002_0025_w3snr10_complement_errfloor0p10mag_"
    "agnfrac0to0p5_chunks_01_02_results.fits"
)
DEFAULT_OUTPUT = DEFAULT_RESULTS_ROOT / (
    "sga2025_ap04_zlt0025_parent_errfloor0p10mag_"
    "agnfrac0to0p5_results.fits"
)
DEFAULT_Z_MAX = 0.025
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
            "Merge the completed WISEsize and complement CIGALE catalogs, "
            "validate the finite nonzero-Z parent, and attach SGA metadata."
        )
    )
    parser.add_argument("--sga-fits", type=Path, default=DEFAULT_SGA_FITS)
    parser.add_argument(
        "--wisesize-results", type=Path, default=DEFAULT_WISESIZE_RESULTS
    )
    parser.add_argument(
        "--complement-results", type=Path, default=DEFAULT_COMPLEMENT_RESULTS
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--z-max", type=float, default=DEFAULT_Z_MAX)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def add_sample_columns(table: Table, label: str, selected: bool) -> Table:
    output = table.copy(copy_data=True)
    output.add_column(
        np.full(len(output), label, dtype="U10"),
        index=1,
        name="CIGALE_SAMPLE",
    )
    output.add_column(
        np.full(len(output), selected, dtype=bool),
        index=2,
        name="WISESIZE_SELECTED",
    )
    return output


def main() -> None:
    args = parse_args()
    paths = (args.sga_fits, args.wisesize_results, args.complement_results)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.z_max <= 0.0:
        raise ValueError("--z-max must be positive.")

    wisesize = Table.read(args.wisesize_results)
    complement = Table.read(args.complement_results)
    if wisesize.colnames != complement.colnames:
        raise ValueError("WISEsize and complement result schemas differ.")
    for name, table in (("WISEsize", wisesize), ("complement", complement)):
        if "SGAID" not in table.colnames:
            raise ValueError(f"{name} results do not contain SGAID.")
        if len(np.unique(table["SGAID"])) != len(table):
            raise ValueError(f"{name} results contain duplicate SGAID values.")

    overlap = np.intersect1d(wisesize["SGAID"], complement["SGAID"])
    if len(overlap):
        raise ValueError(f"Input catalogs overlap by {len(overlap)} SGAID values.")

    combined = vstack(
        [
            add_sample_columns(wisesize, "wisesize", True),
            add_sample_columns(complement, "complement", False),
        ],
        join_type="exact",
        metadata_conflicts="silent",
    )

    with fits.open(args.sga_fits, memmap=True) as hdul:
        sga = hdul[1].data
        sga_ids = np.asarray(sga["SGAID"], dtype=np.int64)
        if len(np.unique(sga_ids)) != len(sga_ids):
            raise ValueError("SGA catalog contains duplicate SGAID values.")
        sorter = np.argsort(sga_ids)
        positions = np.searchsorted(sga_ids, combined["SGAID"], sorter=sorter)
        if np.any(positions == len(sga_ids)):
            raise ValueError("At least one result SGAID is absent from SGA.")
        source_rows = sorter[positions]
        if not np.array_equal(sga_ids[source_rows], combined["SGAID"]):
            raise ValueError("At least one result SGAID is absent from SGA.")

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

        for column in reversed(SGA_COLUMNS):
            combined.add_column(np.array(sga[column][source_rows]), index=1, name=column)

    combined.sort("Z")
    combined.meta["ZMAX"] = (args.z_max, "Strict upper bound on measured SGA Z")
    combined.meta["ZNONZERO"] = (True, "SGA Z=0 missing-redshift rows excluded")
    combined.meta["NWISE"] = (len(wisesize), "Rows from WISEsize selection")
    combined.meta["NCOMP"] = (len(complement), "Rows from fill-in complement")
    combined.meta["SGAFILE"] = args.sga_fits.name

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.write(args.output, overwrite=args.overwrite)
    print(f"WISEsize rows: {len(wisesize):,}")
    print(f"Complement rows: {len(complement):,}")
    print(f"Merged parent rows: {len(combined):,}")
    print(f"Measured Z range: {np.min(combined['Z']):.8f} to {np.max(combined['Z']):.8f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
