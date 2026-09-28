#!/usr/bin/env python
"""Summarize which CIGALE model-grid values are occupied by best-fit results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from astropy.table import Table


DEFAULT_RESULTS = Path("/Users/rfinn/research/Virgo/cigale/cigale_vf_metallicity.fits")
DEFAULT_CSV = Path("/Users/rfinn/research/SGA-CIGALE/reports/conger_full_grid_parameter_occupancy.csv")
DEFAULT_MD = Path("/Users/rfinn/research/SGA-CIGALE/reports/conger_full_grid_parameter_occupancy.md")

GRID = {
    "best.sfh.tau_main": [300, 500, 1000, 3000, 6000, 1e5],
    "best.sfh.tau_burst": [100, 200, 400],
    "best.sfh.f_burst": [0, 0.001, 0.005, 0.01, 0.05, 0.1],
    "best.sfh.age": [1e3, 3e3, 5e3, 7e3, 1e4, 13000],
    "best.sfh.burst_age": [20, 80, 200, 400, 800, 1e3],
    "best.stellar.metallicity": [0.004, 0.02, 0.05],
    "best.attenuation.Av_ISM": [
        0.0,
        0.01,
        0.025,
        0.03,
        0.035,
        0.04,
        0.05,
        0.06,
        0.12,
        0.15,
        1.0,
        1.3,
        1.5,
        1.8,
        2.1,
        2.4,
        2.7,
        3.0,
        3.3,
    ],
    "best.dust.umin": [1.0, 5.0, 10.0],
    "best.dust.alpha": [1.0, 2.0, 2.8],
    "best.dust.gamma": [0.02, 0.1],
    "best.agn.fracAGN": [0.0, 0.05, 0.1, 0.5],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count best-fit occupancy of CIGALE model-grid parameter values."
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    parser.add_argument(
        "--good-chi2-max",
        type=float,
        default=None,
        help="Optional maximum best.reduced_chi_square to include.",
    )
    return parser.parse_args()


def summarize_parameter(table: Table, column: str, grid_values: list[float], mask: np.ndarray) -> list[dict[str, object]]:
    values = np.asarray(table[column], dtype=float)
    values = values[mask & np.isfinite(values)]
    grid = np.asarray(grid_values, dtype=float)
    total = len(values)
    matched_any = np.zeros(total, dtype=bool)
    rows: list[dict[str, object]] = []
    for idx, grid_value in enumerate(grid):
        matched = np.isclose(values, grid_value, rtol=1e-6, atol=1e-10)
        matched_any |= matched
        count = int(np.sum(matched))
        rows.append(
            {
                "parameter": column.replace("best.", ""),
                "column": column,
                "grid_value": f"{grid_value:g}",
                "count": count,
                "fraction": count / total if total else np.nan,
                "total_valid": total,
            }
        )
    off_grid_count = int(np.sum(~matched_any))
    if off_grid_count:
        off_grid_values, off_grid_counts = np.unique(values[~matched_any], return_counts=True)
        rows.append(
            {
                "parameter": column.replace("best.", ""),
                "column": column,
                "grid_value": "OFF_GRID:" + ";".join(
                    f"{value:g}={count}" for value, count in zip(off_grid_values, off_grid_counts)
                ),
                "count": off_grid_count,
                "fraction": off_grid_count / total if total else np.nan,
                "total_valid": total,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["parameter", "column", "grid_value", "count", "fraction", "total_valid"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]], source: Path, mask_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_param: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_param.setdefault(str(row["parameter"]), []).append(row)

    lines = [
        "# Conger Full-Grid Best-Fit Parameter Occupancy",
        "",
        f"Source: `{source}`",
        f"Objects included: {mask_count}",
        "",
        "Counts are based on the discrete `best.*` grid values in Kim Conger's full Virgo Filament Survey CIGALE output.",
        "",
    ]
    for parameter, param_rows in by_param.items():
        grid_rows = [row for row in param_rows if not str(row["grid_value"]).startswith("OFF_GRID:")]
        off_grid = [row for row in param_rows if str(row["grid_value"]).startswith("OFF_GRID:")]
        unused = [row["grid_value"] for row in grid_rows if int(row["count"]) == 0]
        low = [row["grid_value"] for row in grid_rows if 0 < float(row["fraction"]) < 0.01]
        lines.append(f"## {parameter}")
        if unused:
            lines.append(f"- Unused grid values: {', '.join(map(str, unused))}")
        else:
            lines.append("- Unused grid values: none")
        if low:
            lines.append(f"- Occupied by <1% of objects: {', '.join(map(str, low))}")
        for row in off_grid:
            lines.append(f"- Off-grid values found: {str(row['grid_value']).removeprefix('OFF_GRID:')}")
        lines.append("")
        lines.append("| Grid value | Count | Fraction |")
        lines.append("| --- | ---: | ---: |")
        for row in param_rows:
            lines.append(f"| {row['grid_value']} | {row['count']} | {float(row['fraction']):.4f} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    table = Table.read(args.results)
    mask = np.ones(len(table), dtype=bool)
    if args.good_chi2_max is not None:
        chi2 = np.asarray(table["best.reduced_chi_square"], dtype=float)
        mask &= np.isfinite(chi2) & (chi2 <= args.good_chi2_max)

    rows: list[dict[str, object]] = []
    for column, grid_values in GRID.items():
        if column not in table.colnames:
            continue
        rows.extend(summarize_parameter(table, column, grid_values, mask))

    write_csv(args.csv, rows)
    write_markdown(args.markdown, rows, args.results, int(np.sum(mask)))
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
