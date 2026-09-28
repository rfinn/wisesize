#!/usr/bin/env python
"""Compare two CIGALE results tables object-by-object."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from astropy.table import Table


DEFAULT_COLUMNS = [
    "bayes.stellar.m_star",
    "bayes.sfh.sfr",
    "bayes.dust.mass",
    "bayes.attenuation.Av_ISM",
    "bayes.agn.fracAGN",
    "best.reduced_chi_square",
]
LOG_COLUMNS = {
    "bayes.stellar.m_star",
    "bayes.sfh.sfr",
    "bayes.dust.mass",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two CIGALE result FITS tables.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--label-reference", default="reference")
    parser.add_argument("--label-test", default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--columns", nargs="*", default=DEFAULT_COLUMNS)
    parser.add_argument(
        "--strip-prefix",
        default="",
        help="Optional prefix to strip from both ID columns before matching.",
    )
    return parser.parse_args()


def normalize_id(value: object, strip_prefix: str = "") -> str:
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    elif isinstance(value, (np.integer, int)):
        text = str(int(value))
    elif isinstance(value, (np.floating, float)) and np.isfinite(value):
        rounded = round(float(value))
        text = str(int(rounded)) if np.isclose(value, rounded) else f"{value:g}"
    else:
        text = str(value)
    text = text.strip()
    if strip_prefix and text.startswith(strip_prefix):
        text = text[len(strip_prefix) :]
    return text


def indexed_table(path: Path, strip_prefix: str) -> tuple[Table, dict[str, int]]:
    table = Table.read(path)
    index = {
        normalize_id(value, strip_prefix): idx
        for idx, value in enumerate(table["id"])
    }
    return table, index


def finite_pair(ref: np.ndarray, test: np.ndarray, log: bool) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(ref, dtype=float)
    test = np.asarray(test, dtype=float)
    ok = np.isfinite(ref) & np.isfinite(test)
    if log:
        ok &= (ref > 0) & (test > 0)
        ref = np.log10(ref[ok])
        test = np.log10(test[ok])
    else:
        ref = ref[ok]
        test = test[ok]
    return ref, test


def summarize_column(ref: np.ndarray, test: np.ndarray, log: bool) -> dict[str, float]:
    refv, testv = finite_pair(ref, test, log)
    delta = testv - refv
    if len(delta) == 0:
        return {"n": 0, "median_delta": np.nan, "mad_delta": np.nan, "p16_delta": np.nan, "p84_delta": np.nan}
    return {
        "n": len(delta),
        "median_delta": float(np.nanmedian(delta)),
        "mad_delta": float(np.nanmedian(np.abs(delta - np.nanmedian(delta)))),
        "p16_delta": float(np.nanpercentile(delta, 16)),
        "p84_delta": float(np.nanpercentile(delta, 84)),
    }


def plot_column(
    output_dir: Path,
    column: str,
    ref: np.ndarray,
    test: np.ndarray,
    log: bool,
    label_reference: str,
    label_test: str,
) -> None:
    refv, testv = finite_pair(ref, test, log)
    if len(refv) == 0:
        return
    delta = testv - refv
    stem = column.replace(".", "_")
    xlabel = f"log10 {label_reference}" if log else label_reference
    ylabel = f"log10 {label_test}" if log else label_test
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].scatter(refv, testv, s=14, alpha=0.75)
    lo = min(np.nanmin(refv), np.nanmin(testv))
    hi = max(np.nanmax(refv), np.nanmax(testv))
    axes[0].plot([lo, hi], [lo, hi], color="0.3", lw=1)
    axes[0].set_xlabel(f"{column}\n{xlabel}")
    axes[0].set_ylabel(ylabel)
    axes[0].set_title("Object-by-object")

    axes[1].axhline(0, color="0.3", lw=1)
    axes[1].scatter(refv, delta, s=14, alpha=0.75)
    axes[1].set_xlabel(f"{column}\n{xlabel}")
    axes[1].set_ylabel(f"{label_test} - {label_reference}")
    axes[1].set_title("Residual")
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_comparison.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ref_table, ref_index = indexed_table(args.reference, args.strip_prefix)
    test_table, test_index = indexed_table(args.test, args.strip_prefix)
    common_ids = sorted(set(ref_index) & set(test_index))
    if not common_ids:
        raise RuntimeError("No common IDs found between the two result tables.")

    summary_rows = []
    for column in args.columns:
        if column not in ref_table.colnames or column not in test_table.colnames:
            continue
        ref = np.asarray([ref_table[column][ref_index[obj_id]] for obj_id in common_ids], dtype=float)
        test = np.asarray([test_table[column][test_index[obj_id]] for obj_id in common_ids], dtype=float)
        log = column in LOG_COLUMNS
        stats = summarize_column(ref, test, log)
        stats.update({"column": column, "log10": log})
        summary_rows.append(stats)
        plot_column(
            args.output_dir,
            column,
            ref,
            test,
            log,
            args.label_reference,
            args.label_test,
        )

    with (args.output_dir / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["column", "log10", "n", "median_delta", "mad_delta", "p16_delta", "p84_delta"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# CIGALE Results Comparison",
        "",
        f"Reference: `{args.reference}`",
        f"Test: `{args.test}`",
        f"Matched objects: {len(common_ids)}",
        "",
        "| Column | log10? | N | median delta | MAD | p16 | p84 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['column']} | {row['log10']} | {row['n']} | "
            f"{row['median_delta']:.4g} | {row['mad_delta']:.4g} | "
            f"{row['p16_delta']:.4g} | {row['p84_delta']:.4g} |"
        )
    (args.output_dir / "comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Matched {len(common_ids)} objects")
    print(f"Wrote {args.output_dir / 'comparison_summary.md'}")


if __name__ == "__main__":
    main()
