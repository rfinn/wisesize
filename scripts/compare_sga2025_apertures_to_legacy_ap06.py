#!/usr/bin/env python3
"""Find the SGA2025 aperture that best reproduces Kim's legacy AP06 fluxes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.table import Table


DEFAULT_SGA = Path("/Users/rfinn/research/SGA2025/SGA2025-v1.0.fits")
DEFAULT_LEGACY = Path(
    "/Users/rfinn/research/Virgo/tables-north/v2/vf_v2_legacy_ephot.fits"
)
DEFAULT_OVERLAP = Path(
    "/Users/rfinn/research/SGA-CIGALE/comparisons/"
    "wisesize_vf_cigale_metallicity_20260305/"
    "wisesize_vf_cigale_overlap.fits"
)
DEFAULT_OUTPUT_DIR = Path(
    "/Users/rfinn/research/SGA-CIGALE/reports/"
    "sga2025_aperture_match_legacy_ap06"
)
APERTURES = [f"AP{index:02d}" for index in range(5)]
BANDS = ["FUV", "NUV", "G", "R", "Z", "W1", "W2", "W3", "W4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sga", type=Path, default=DEFAULT_SGA)
    parser.add_argument("--legacy", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--overlap", type=Path, default=DEFAULT_OVERLAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def aligned_indices(values: np.ndarray, requested: np.ndarray, label: str) -> np.ndarray:
    order = np.argsort(values)
    positions = np.searchsorted(values[order], requested)
    if np.any(positions == len(values)):
        raise RuntimeError(f"Could not align every {label} identifier.")
    indices = order[positions]
    if not np.array_equal(values[indices], requested):
        raise RuntimeError(f"Could not align every {label} identifier.")
    return indices


def ratio_stats(
    aperture: str,
    quantity: str,
    new: np.ndarray,
    legacy: np.ndarray,
) -> dict[str, object]:
    good = np.isfinite(new) & (new > 0) & np.isfinite(legacy) & (legacy > 0)
    log_ratio = np.log10(new[good] / legacy[good])
    median = float(np.median(log_ratio))
    return {
        "sga_aperture": aperture,
        "quantity": quantity,
        "n": int(len(log_ratio)),
        "median_log_ratio": median,
        "median_ratio": float(10**median),
        "mad_log_ratio": float(np.median(np.abs(log_ratio - median))),
        "p16_log_ratio": float(np.percentile(log_ratio, 16)),
        "p84_log_ratio": float(np.percentile(log_ratio, 84)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_band_ratios(path: Path, rows: list[dict[str, object]]) -> None:
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    positions = np.arange(len(BANDS))
    for aperture in APERTURES:
        selected = {
            str(row["quantity"]): row
            for row in rows
            if row["sga_aperture"] == aperture and row["quantity"] in BANDS
        }
        medians = np.array([selected[band]["median_log_ratio"] for band in BANDS])
        axis.plot(positions, medians, marker="o", label=aperture)
    axis.axhline(0, color="black", lw=1)
    axis.set_xticks(positions, BANDS)
    axis.set_ylabel("Median log10(SGA2025 aperture / legacy AP06)")
    axis.set_xlabel("Band")
    axis.legend(title="SGA2025", ncol=3, frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlap = Table.read(args.overlap)
    legacy_table = Table.read(args.legacy)

    with fits.open(args.sga, memmap=True) as hdul:
        sga_phot = hdul[2].data
        sga_indices = aligned_indices(
            np.asarray(sga_phot["SGAID"], dtype=np.int64),
            np.asarray(overlap["SGAID"], dtype=np.int64),
            "SGAID",
        )
        legacy_indices = aligned_indices(
            np.asarray(legacy_table["VFID"]).astype(str),
            np.asarray(overlap["VFID"]).astype(str),
            "VFID",
        )

        rows: list[dict[str, object]] = []
        for aperture in APERTURES:
            rows.append(
                ratio_stats(
                    aperture,
                    "SMA",
                    np.asarray(sga_phot[f"SMA_{aperture}"][sga_indices], dtype=float),
                    np.asarray(legacy_table["SMA_AP06"][legacy_indices], dtype=float),
                )
            )
            for band in BANDS:
                rows.append(
                    ratio_stats(
                        aperture,
                        band,
                        np.asarray(
                            sga_phot[f"FLUX_{aperture}_{band}"][sga_indices],
                            dtype=float,
                        ),
                        np.asarray(
                            legacy_table[f"FLUX_AP06_{band}"][legacy_indices],
                            dtype=float,
                        ),
                    )
                )

    summary = []
    for aperture in APERTURES:
        selected = [
            row
            for row in rows
            if row["sga_aperture"] == aperture and row["quantity"] in BANDS
        ]
        sma = next(
            row
            for row in rows
            if row["sga_aperture"] == aperture and row["quantity"] == "SMA"
        )
        summary.append(
            {
                "sga_aperture": aperture,
                "median_sma_ratio": sma["median_ratio"],
                "median_band_ratio": float(
                    10 ** np.median([row["median_log_ratio"] for row in selected])
                ),
                "median_abs_band_log_offset": float(
                    np.median([abs(row["median_log_ratio"]) for row in selected])
                ),
                "median_band_log_mad": float(
                    np.median([row["mad_log_ratio"] for row in selected])
                ),
            }
        )
    best = min(summary, key=lambda row: row["median_abs_band_log_offset"])

    write_csv(args.output_dir / "aperture_band_comparison.csv", rows)
    write_csv(args.output_dir / "aperture_summary.csv", summary)
    plot_band_ratios(args.output_dir / "aperture_band_comparison.png", rows)

    lines = [
        "# SGA2025 aperture match to Kim legacy AP06",
        "",
        f"Matched objects: {len(overlap)}",
        f"Best SGA2025 aperture: **{best['sga_aperture']}**",
        "",
        "The best aperture minimizes the median absolute band-by-band log-flux "
        "offset relative to legacy AP06.",
        "",
        "| SGA aperture | Median SMA ratio | Median band flux ratio | "
        "Median absolute log offset | Median band log MAD |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['sga_aperture']} | {row['median_sma_ratio']:.4f} | "
            f"{row['median_band_ratio']:.4f} | "
            f"{row['median_abs_band_log_offset']:.4f} | "
            f"{row['median_band_log_mad']:.4f} |"
        )
    (args.output_dir / "aperture_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Best aperture: {best['sga_aperture']}")
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
