#!/usr/bin/env python
"""Compare Kim Conger's VFS photometry with SGA2025/AP03 photometry.

The comparison is performed in two forms for the same VFID-matched objects:

1. The exact mJy values supplied to CIGALE in Kim's and the SGA2025 inputs.
2. The underlying legacy AP04 and SGA2025 AP03 measurements in nanomaggies.

Keeping both views separates changes in the aperture photometry from changes in
Milky Way extinction corrections, uncertainty handling, and output rounding.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u


DEFAULT_NEW_INPUT = Path(
    "/Users/rfinn/research/SGA-CIGALE/inputs/"
    "conger_overlap_sga2025_ap03_cigale100_v2match.dat"
)
DEFAULT_KIM_INPUT = Path("/Users/rfinn/research/SGA-CIGALE/conger_test/vf_data.txt")
DEFAULT_OLD_EPHOT = Path(
    "/Users/rfinn/research/Virgo/tables-north/v2/vf_v2_legacy_ephot.fits"
)
DEFAULT_INTERMEDIATE = Path(
    "/Users/rfinn/research/SGA-CIGALE/intermediate/"
    "sga2025_ap03_cigale_intermediate.fits"
)
DEFAULT_MATCHES = Path(
    "/Users/rfinn/research/SGA-CIGALE/reports/"
    "conger_overlap_sga2025_matches_100_v2match.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/Users/rfinn/research/SGA-CIGALE/reports/"
    "compare_kim_vs_sga2025_input_photometry100_v2match"
)
DEFAULT_DEC_CUT = 32.375
NANOMAGGY_TO_MJY = 0.003631


BANDS = {
    "FUV": {
        "kim": ("FUV",),
        "new": ("galex.FUV",),
        "raw": "FUV",
    },
    "NUV": {
        "kim": ("NUV",),
        "new": ("galex.NUV",),
        "raw": "NUV",
    },
    "g": {
        "kim": ("BASS-g", "decamDR1-g"),
        "new": ("BASS-g", "decamDR1-g"),
        "raw": "G",
    },
    "r": {
        "kim": ("BASS-r", "decamDR1-r"),
        "new": ("BASS-r", "decamDR1-r"),
        "raw": "R",
    },
    "z": {
        "kim": ("decamDR1-z",),
        "new": ("decamDR1-z",),
        "raw": "Z",
    },
    "W1": {
        "kim": ("WISE1",),
        "new": ("wise.W1",),
        "raw": "W1",
    },
    "W2": {
        "kim": ("WISE2",),
        "new": ("wise.W2",),
        "raw": "W2",
    },
    "W3": {
        "kim": ("WISE3",),
        "new": ("wise.W3",),
        "raw": "W3",
    },
    "W4": {
        "kim": ("WISE4",),
        "new": ("wise.W4",),
        "raw": "W4",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Kim's VFS input photometry with SGA2025/AP03 photometry."
    )
    parser.add_argument("--new-input", type=Path, default=DEFAULT_NEW_INPUT)
    parser.add_argument("--kim-input", type=Path, default=DEFAULT_KIM_INPUT)
    parser.add_argument("--old-ephot", type=Path, default=DEFAULT_OLD_EPHOT)
    parser.add_argument("--intermediate", type=Path, default=DEFAULT_INTERMEDIATE)
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dec-cut", type=float, default=DEFAULT_DEC_CUT)
    return parser.parse_args()


def normalize_id(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value).strip()


def read_cigale_ascii(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                names = line[1:].strip().split()
                break
        else:
            raise ValueError(f"No commented header found in {path}")
    data = np.genfromtxt(path, comments="#", dtype=str, encoding="utf-8")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != len(names):
        raise ValueError(f"{path} has {data.shape[1]} columns but {len(names)} names")
    return names, data


def table_index(values: object) -> dict[str, int]:
    return {normalize_id(value): i for i, value in enumerate(values)}


def ascii_value(
    names: list[str], row: np.ndarray, candidates: tuple[str, ...]
) -> tuple[float, float, str]:
    for name in candidates:
        if name not in names or f"{name}_err" not in names:
            continue
        flux = float(row[names.index(name)])
        error = float(row[names.index(f"{name}_err")])
        if np.isfinite(flux) or np.isfinite(error):
            return flux, error, name
    return np.nan, np.nan, ""


def finite_positive(*values: float) -> bool:
    return all(np.isfinite(value) and value > 0.0 for value in values)


def ratio_stats(reference: np.ndarray, test: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(reference) & np.isfinite(test) & (reference > 0.0) & (test > 0.0)
    if not np.any(ok):
        return {
            "n": 0,
            "median_log_ratio": np.nan,
            "mad_log_ratio": np.nan,
            "p16_log_ratio": np.nan,
            "p84_log_ratio": np.nan,
            "median_flux_ratio": np.nan,
            "median_delta_mag": np.nan,
        }
    delta = np.log10(test[ok] / reference[ok])
    median = np.median(delta)
    return {
        "n": int(np.count_nonzero(ok)),
        "median_log_ratio": float(median),
        "mad_log_ratio": float(np.median(np.abs(delta - median))),
        "p16_log_ratio": float(np.percentile(delta, 16)),
        "p84_log_ratio": float(np.percentile(delta, 84)),
        "median_flux_ratio": float(10**median),
        "median_delta_mag": float(-2.5 * median),
    }


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, int]]:
    new_names, new_data = read_cigale_ascii(args.new_input)
    kim_names, kim_data = read_cigale_ascii(args.kim_input)
    old = Table.read(args.old_ephot)
    intermediate = Table.read(args.intermediate, hdu="SGA_AP03_CIGALE_PREP")
    matches = Table.read(args.matches, format="ascii.csv")

    new_index = table_index(new_data[:, new_names.index("id")])
    kim_index = table_index(kim_data[:, kim_names.index("id")])
    old_index = table_index(old["VFID"])
    intermediate_index = table_index(intermediate["SGA_ID"])
    match_index = table_index(matches["vfid"])

    common = [
        vfid
        for vfid in new_index
        if vfid in kim_index and vfid in old_index and vfid in match_index
    ]
    if not common:
        raise RuntimeError("No common VFIDs found among the input tables.")

    rows: list[dict[str, object]] = []
    skipped_intermediate = 0
    for vfid in common:
        new_row = new_data[new_index[vfid]]
        kim_row = kim_data[kim_index[vfid]]
        old_row = old[old_index[vfid]]
        match_row = matches[match_index[vfid]]
        sgaid = normalize_id(match_row["sgaid"])
        if sgaid not in intermediate_index:
            skipped_intermediate += 1
            continue
        new_raw_row = intermediate[intermediate_index[sgaid]]
        dec = float(match_row["sga_dec"])
        region = "south" if dec < args.dec_cut else "north"
        legacy_coord = SkyCoord(
            float(old_row["RA_MOMENT"]) * u.deg,
            float(old_row["DEC_MOMENT"]) * u.deg,
        )
        new_coord = SkyCoord(
            float(match_row["sga_ra"]) * u.deg,
            float(match_row["sga_dec"]) * u.deg,
        )
        legacy_centroid_sep_arcsec = float(legacy_coord.separation(new_coord).arcsec)

        for band, config in BANDS.items():
            kim_flux, kim_error, kim_filter = ascii_value(
                kim_names, kim_row, config["kim"]
            )
            new_flux, new_error, new_filter = ascii_value(
                new_names, new_row, config["new"]
            )
            raw_band = str(config["raw"])
            raw_lower = raw_band.lower()

            old_flux = float(old_row[f"FLUX_AP04_{raw_band}"])
            old_ivar = float(old_row[f"FLUX_IVAR_AP04_{raw_band}"])
            old_error = old_ivar**-0.5 if np.isfinite(old_ivar) and old_ivar > 0 else np.nan

            corrected_flux = float(new_raw_row[f"flux_ap03_{raw_lower}_mjy_corr"])
            corrected_error = float(
                new_raw_row[f"flux_err_ap03_{raw_lower}_mjy_corr"]
            )
            transmission = float(new_raw_row[f"mw_transmission_{raw_lower}"])
            new_raw_flux = corrected_flux * transmission / NANOMAGGY_TO_MJY
            new_raw_error = corrected_error * transmission / NANOMAGGY_TO_MJY
            new_band_used = bool(new_raw_row[f"use_band_{raw_lower}"])

            input_ok = finite_positive(kim_flux, new_flux)
            raw_ok = (
                input_ok
                and new_band_used
                and finite_positive(old_flux, new_raw_flux)
            )
            rows.append(
                {
                    "vfid": vfid,
                    "sgaid": sgaid,
                    "dec": dec,
                    "region": region,
                    "legacy_centroid_sep_arcsec": legacy_centroid_sep_arcsec,
                    "legacy_centroid_match": legacy_centroid_sep_arcsec <= 10.0,
                    "band": band,
                    "kim_filter": kim_filter,
                    "new_filter": new_filter,
                    "kim_input_mjy": kim_flux,
                    "kim_input_error_mjy": kim_error,
                    "new_input_mjy": new_flux,
                    "new_input_error_mjy": new_error,
                    "input_pair_valid": input_ok,
                    "input_log10_ratio": (
                        np.log10(new_flux / kim_flux) if input_ok else np.nan
                    ),
                    "input_delta_mag": (
                        -2.5 * np.log10(new_flux / kim_flux) if input_ok else np.nan
                    ),
                    "old_raw_nanomaggy": old_flux,
                    "old_raw_error_nanomaggy": old_error,
                    "new_raw_nanomaggy": new_raw_flux,
                    "new_raw_error_nanomaggy": new_raw_error,
                    "new_mw_transmission": transmission,
                    "new_band_used": new_band_used,
                    "raw_pair_valid": raw_ok,
                    "raw_log10_ratio": (
                        np.log10(new_raw_flux / old_flux) if raw_ok else np.nan
                    ),
                    "raw_delta_mag": (
                        -2.5 * np.log10(new_raw_flux / old_flux) if raw_ok else np.nan
                    ),
                }
            )

    counts = {
        "new_input_rows": len(new_index),
        "common_vfids": len(common),
        "compared_vfids": len({str(row["vfid"]) for row in rows}),
        "south_vfids": len(
            {str(row["vfid"]) for row in rows if row["region"] == "south"}
        ),
        "north_vfids": len(
            {str(row["vfid"]) for row in rows if row["region"] == "north"}
        ),
        "skipped_intermediate": skipped_intermediate,
        "legacy_centroid_matches": len(
            {
                str(row["vfid"])
                for row in rows
                if bool(row["legacy_centroid_match"])
            }
        ),
    }
    return rows, counts


def make_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for comparison, ref_key, test_key, ref_err_key, test_err_key in (
        (
            "cigale_input",
            "kim_input_mjy",
            "new_input_mjy",
            "kim_input_error_mjy",
            "new_input_error_mjy",
        ),
        (
            "raw_old_ap04_new_ap03",
            "old_raw_nanomaggy",
            "new_raw_nanomaggy",
            "old_raw_error_nanomaggy",
            "new_raw_error_nanomaggy",
        ),
    ):
        valid_key = "input_pair_valid" if comparison == "cigale_input" else "raw_pair_valid"
        for region in ("all", "south", "north"):
            for band in BANDS:
                selected = [
                    row
                    for row in rows
                    if row["band"] == band
                    and bool(row[valid_key])
                    and (region == "all" or row["region"] == region)
                ]
                reference = np.asarray([row[ref_key] for row in selected], dtype=float)
                test = np.asarray([row[test_key] for row in selected], dtype=float)
                reference_error = np.asarray(
                    [row[ref_err_key] for row in selected], dtype=float
                )
                test_error = np.asarray([row[test_err_key] for row in selected], dtype=float)
                stats = ratio_stats(reference, test)
                error_stats = ratio_stats(reference_error, test_error)
                fractional_error_ok = (
                    np.isfinite(reference)
                    & np.isfinite(test)
                    & np.isfinite(reference_error)
                    & np.isfinite(test_error)
                    & (reference > 0.0)
                    & (test > 0.0)
                    & (reference_error > 0.0)
                    & (test_error > 0.0)
                )
                if np.any(fractional_error_ok):
                    median_reference_fractional_error = float(
                        np.median(
                            reference_error[fractional_error_ok]
                            / reference[fractional_error_ok]
                        )
                    )
                    median_test_fractional_error = float(
                        np.median(
                            test_error[fractional_error_ok] / test[fractional_error_ok]
                        )
                    )
                else:
                    median_reference_fractional_error = np.nan
                    median_test_fractional_error = np.nan
                summary.append(
                    {
                        "comparison": comparison,
                        "region": region,
                        "band": band,
                        **stats,
                        "n_error": error_stats["n"],
                        "median_log_error_ratio": error_stats["median_log_ratio"],
                        "median_error_ratio": error_stats["median_flux_ratio"],
                        "median_reference_fractional_error": (
                            median_reference_fractional_error
                        ),
                        "median_test_fractional_error": median_test_fractional_error,
                    }
                )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_flux_grid(
    rows: list[dict[str, object]],
    output_path: Path,
    comparison: str,
) -> None:
    if comparison == "cigale_input":
        ref_key, test_key, valid_key = (
            "kim_input_mjy",
            "new_input_mjy",
            "input_pair_valid",
        )
        ref_label, test_label = "Kim input (mJy)", "SGA2025 input (mJy)"
        title = "CIGALE-ready input photometry"
    else:
        ref_key, test_key, valid_key = (
            "old_raw_nanomaggy",
            "new_raw_nanomaggy",
            "raw_pair_valid",
        )
        ref_label, test_label = "Legacy AP04 (nanomaggy)", "SGA2025 AP03 (nanomaggy)"
        title = "Legacy AP04 versus SGA2025 AP03 photometry"

    colors = {"south": "#0072B2", "north": "#D55E00"}
    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    for ax, band in zip(axes.flat, BANDS):
        selected = [row for row in rows if row["band"] == band and bool(row[valid_key])]
        for region in ("south", "north"):
            region_rows = [row for row in selected if row["region"] == region]
            x = np.asarray([row[ref_key] for row in region_rows], dtype=float)
            y = np.asarray([row[test_key] for row in region_rows], dtype=float)
            ok = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
            ax.scatter(
                np.log10(x[ok]),
                np.log10(y[ok]),
                s=15,
                alpha=0.65,
                color=colors[region],
                label=region,
            )
        if selected:
            x = np.asarray([row[ref_key] for row in selected], dtype=float)
            y = np.asarray([row[test_key] for row in selected], dtype=float)
            ok = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
            if np.any(ok):
                lo = min(np.min(np.log10(x[ok])), np.min(np.log10(y[ok])))
                hi = max(np.max(np.log10(x[ok])), np.max(np.log10(y[ok])))
                ax.plot([lo, hi], [lo, hi], color="0.25", lw=1)
                delta_mag = -2.5 * np.log10(y[ok] / x[ok])
                ax.text(
                    0.04,
                    0.96,
                    f"N={np.count_nonzero(ok)}\nmed Δm={np.median(delta_mag):+.2f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                )
        ax.set_title(band)
        ax.set_xlabel(f"log10 {ref_label}")
        ax.set_ylabel(f"log10 {test_label}")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.002),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_delta_summary(summary: list[dict[str, object]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    colors = {"south": "#0072B2", "north": "#D55E00"}
    offsets = {"south": -0.13, "north": 0.13}
    x = np.arange(len(BANDS), dtype=float)
    for ax, comparison, title in (
        (axes[0], "cigale_input", "CIGALE-ready flux: SGA2025 minus Kim"),
        (
            axes[1],
            "raw_old_ap04_new_ap03",
            "Raw flux: SGA2025 AP03 minus legacy AP04",
        ),
    ):
        for region in ("south", "north"):
            selected = {
                str(row["band"]): row
                for row in summary
                if row["comparison"] == comparison and row["region"] == region
            }
            med = np.asarray(
                [selected[band]["median_delta_mag"] for band in BANDS], dtype=float
            )
            low = np.asarray(
                [-2.5 * selected[band]["p84_log_ratio"] for band in BANDS],
                dtype=float,
            )
            high = np.asarray(
                [-2.5 * selected[band]["p16_log_ratio"] for band in BANDS],
                dtype=float,
            )
            yerr = np.vstack((med - low, high - med))
            ax.errorbar(
                x + offsets[region],
                med,
                yerr=yerr,
                fmt="o",
                ms=5,
                capsize=3,
                color=colors[region],
                label=region,
            )
        ax.axhline(0.0, color="0.25", lw=1)
        ax.set_ylabel("median Δmag\n(16th–84th pct.)")
        ax.set_title(title)
        ax.grid(axis="y", color="0.9", lw=0.8)
    axes[0].legend(frameon=False, ncol=2)
    axes[1].set_xticks(x, list(BANDS))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def fmt(value: object) -> str:
    numeric = float(value)
    return "nan" if not np.isfinite(numeric) else f"{numeric:.3f}"


def write_report(
    path: Path,
    args: argparse.Namespace,
    counts: dict[str, int],
    summary: list[dict[str, object]],
) -> None:
    lines = [
        "# Kim versus SGA2025 input photometry",
        "",
        f"New CIGALE input: `{args.new_input}`",
        f"Kim CIGALE input: `{args.kim_input}`",
        f"Legacy AP04 source: `{args.old_ephot}`",
        f"Declination split: `{args.dec_cut:g}` degrees",
        "",
        f"Matched VFIDs: {counts['compared_vfids']} "
        f"({counts['south_vfids']} south, {counts['north_vfids']} north)",
        f"Legacy aperture centroids within 10 arcsec of SGA2025: "
        f"{counts['legacy_centroid_matches']} / {counts['compared_vfids']}",
        "",
        "The CIGALE-input comparison uses the exact mJy values supplied to each fit. "
        "The raw comparison uses legacy AP04 and SGA2025 AP03 nanomaggies and is "
        "restricted to bands supplied in both CIGALE inputs. Positive finite fluxes "
        "are used for logarithmic ratios. "
        "Delta magnitude is `-2.5 log10(SGA2025 / Kim-or-legacy)`, so a negative value "
        "means the SGA2025 flux is brighter. Kim's input has no i-band, so i is not "
        "included here.",
        "",
    ]
    for comparison, heading in (
        ("cigale_input", "CIGALE-ready mJy fluxes"),
        (
            "raw_old_ap04_new_ap03",
            "Raw legacy AP04 versus SGA2025 AP03 nanomaggy fluxes",
        ),
    ):
        lines.extend(
            [
                f"## {heading}",
                "",
                "| Region | Band | N | Median SGA2025/reference | Median delta mag | "
                "16th-84th log ratio | Median error ratio | Median fractional error "
                "reference -> SGA2025 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for region in ("all", "south", "north"):
            for band in BANDS:
                row = next(
                    item
                    for item in summary
                    if item["comparison"] == comparison
                    and item["region"] == region
                    and item["band"] == band
                )
                lines.append(
                    f"| {region} | {band} | {row['n']} | "
                    f"{fmt(row['median_flux_ratio'])} | {fmt(row['median_delta_mag'])} | "
                    f"{fmt(row['p16_log_ratio'])} to {fmt(row['p84_log_ratio'])} | "
                    f"{fmt(row['median_error_ratio'])} | "
                    f"{fmt(row['median_reference_fractional_error'])} -> "
                    f"{fmt(row['median_test_fractional_error'])} |"
                )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, counts = build_rows(args)
    summary = make_summary(rows)

    write_csv(args.output_dir / "photometry_comparison_long.csv", rows)
    write_csv(args.output_dir / "photometry_summary.csv", summary)
    plot_flux_grid(
        rows,
        args.output_dir / "cigale_input_flux_comparison.png",
        "cigale_input",
    )
    plot_flux_grid(
        rows,
        args.output_dir / "raw_ap04_vs_ap03_flux_comparison.png",
        "raw_old_ap04_new_ap03",
    )
    plot_delta_summary(summary, args.output_dir / "delta_mag_summary.png")
    write_report(
        args.output_dir / "photometry_comparison.md", args, counts, summary
    )

    print(
        f"Compared {counts['compared_vfids']} VFIDs: "
        f"{counts['south_vfids']} south and {counts['north_vfids']} north"
    )
    print(f"Wrote {args.output_dir / 'photometry_comparison.md'}")


if __name__ == "__main__":
    main()
