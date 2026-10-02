#!/usr/bin/env python
"""Cross-check WISEsize CIGALE results against the Virgo Filament CIGALE table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from astropy.constants import c
from astropy.coordinates import SkyCoord, match_coordinates_sky, search_around_sky
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from scipy.stats import spearmanr


DEFAULT_CIGALE = Path(
    "/Users/rfinn/research/SGA-CIGALE/results/"
    "wisesize_sga2025_ap03_z0002_0025_w3snr10_errfloor0p10mag_results.fits"
)
DEFAULT_SGA = Path("/Users/rfinn/research/SGA2025/SGA2025-v1.0.fits")
DEFAULT_VF_MAIN = Path(
    "/Users/rfinn/research/Virgo/tables-north/v2/vf_v2_main.fits"
)
DEFAULT_VF_CIGALE = Path(
    "/Users/rfinn/research/Virgo/tables-north/v2/"
    "vf_v2_cigale_metallicity_20260305.fits"
)
DEFAULT_OUTPUT_DIR = Path(
    "/Users/rfinn/research/SGA-CIGALE/comparisons/"
    "wisesize_vf_cigale_metallicity_20260305"
)

PRIMARY_PARAMETERS = [
    {
        "short": "MSTAR",
        "column": "bayes.stellar.m_star",
        "label": r"Bayesian $M_\star$",
        "log10": True,
    },
    {
        "short": "BEST_MSTAR",
        "column": "best.stellar.m_star",
        "label": r"best-fit $M_\star$",
        "log10": True,
    },
    {
        "short": "SFR",
        "column": "bayes.sfh.sfr",
        "label": "Bayesian SFR",
        "log10": True,
    },
    {
        "short": "BEST_SFR",
        "column": "best.sfh.sfr",
        "label": "best-fit SFR",
        "log10": True,
        "plot_percentiles": (2, 98),
    },
]

SECONDARY_PARAMETERS = [
    {
        "short": "DUST_MASS",
        "column": "bayes.dust.mass",
        "label": r"$M_{dust}$",
        "log10": True,
    },
    {
        "short": "AV_ISM",
        "column": "bayes.attenuation.Av_ISM",
        "label": r"$A_{V,ISM}$",
        "log10": False,
    },
    {
        "short": "FRAC_AGN",
        "column": "bayes.agn.fracAGN",
        "label": "AGN fraction",
        "log10": False,
    },
    {
        "short": "METALLICITY",
        "column": "bayes.stellar.metallicity",
        "label": "Stellar metallicity",
        "log10": False,
    },
]

PARAMETERS = PRIMARY_PARAMETERS + SECONDARY_PARAMETERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-match and compare WISEsize and Virgo Filament CIGALE results."
    )
    parser.add_argument("--cigale", type=Path, default=DEFAULT_CIGALE)
    parser.add_argument("--sga", type=Path, default=DEFAULT_SGA)
    parser.add_argument("--vf-main", type=Path, default=DEFAULT_VF_MAIN)
    parser.add_argument("--vf-cigale", type=Path, default=DEFAULT_VF_CIGALE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-separation-arcsec", type=float, default=30.0)
    parser.add_argument(
        "--redshift-confirm-tolerance",
        type=float,
        default=0.001,
        help="Maximum |SGA z - VF vr/c| for a redshift-confirmed match.",
    )
    parser.add_argument(
        "--redshift-discordant-tolerance",
        type=float,
        default=0.003,
        help="Differences above this value are flagged as discordant.",
    )
    return parser.parse_args()


def align_sga_to_cigale(sga_path: Path, sgaids: np.ndarray) -> dict[str, np.ndarray]:
    with fits.open(sga_path, memmap=True) as hdul:
        sga = hdul[1].data
        catalog_ids = np.asarray(sga["SGAID"], dtype=np.int64)
        order = np.argsort(catalog_ids)
        positions = np.searchsorted(catalog_ids[order], sgaids)
        in_bounds = positions < len(order)
        found = np.zeros(len(sgaids), dtype=bool)
        found[in_bounds] = catalog_ids[order[positions[in_bounds]]] == sgaids[in_bounds]
        if not np.all(found):
            missing = sgaids[~found]
            raise RuntimeError(
                f"{len(missing)} CIGALE SGAIDs are absent from the SGA catalog; "
                f"first missing IDs: {missing[:10].tolist()}"
            )
        indices = order[positions]
        return {
            "RA": np.asarray(sga["RA"][indices], dtype=float),
            "DEC": np.asarray(sga["DEC"][indices], dtype=float),
            "Z": np.asarray(sga["Z"][indices], dtype=float),
            "Z_REF": np.asarray(sga["Z_REF"][indices]).astype(str),
        }


def validate_vf_row_alignment(vf_main: Table, vf_cigale: Table) -> None:
    if len(vf_main) != len(vf_cigale):
        raise RuntimeError(
            f"VF tables are not row aligned: {len(vf_main)} vs {len(vf_cigale)} rows."
        )
    main_ids = np.asarray(vf_main["VFID"]).astype(str)
    cigale_ids = np.asarray(vf_cigale["id"]).astype(str)
    if not np.array_equal(main_ids, cigale_ids):
        bad = np.flatnonzero(main_ids != cigale_ids)
        raise RuntimeError(
            "VFID and CIGALE id are not row aligned; first mismatched rows: "
            f"{bad[:10].tolist()}"
        )


def match_catalogs(
    sga: dict[str, np.ndarray],
    vf_main: Table,
    max_separation_arcsec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sga_coord = SkyCoord(sga["RA"] * u.deg, sga["DEC"] * u.deg)
    vf_coord = SkyCoord(
        np.asarray(vf_main["RA"], dtype=float) * u.deg,
        np.asarray(vf_main["DEC"], dtype=float) * u.deg,
    )
    nearest_index, separation, _ = match_coordinates_sky(sga_coord, vf_coord)
    matched = separation.arcsec <= max_separation_arcsec
    sga_candidate_index, _, _, _ = search_around_sky(
        sga_coord, vf_coord, max_separation_arcsec * u.arcsec
    )
    candidate_count = np.bincount(
        sga_candidate_index, minlength=len(sga_coord)
    ).astype(np.int16)

    duplicate_assignment = np.zeros(len(sga_coord), dtype=bool)
    valid_indices = nearest_index[matched]
    unique, counts = np.unique(valid_indices, return_counts=True)
    duplicated = unique[counts > 1]
    if len(duplicated):
        duplicate_assignment[matched] = np.isin(valid_indices, duplicated)
    return (
        np.asarray(nearest_index, dtype=np.int64),
        separation.arcsec,
        matched,
        candidate_count,
        duplicate_assignment,
    )


def valid_vf_fit(vf_cigale: Table) -> np.ndarray:
    required_positive = [
        "bayes.stellar.m_star",
        "bayes.sfh.sfr",
        "bayes.dust.mass",
        "best.reduced_chi_square",
    ]
    valid = np.ones(len(vf_cigale), dtype=bool)
    for name in required_positive:
        values = np.asarray(vf_cigale[name], dtype=float)
        valid &= np.isfinite(values) & (values > 0)
    return valid


def comparison_stats(
    quantity: str,
    subset: str,
    wisesize: np.ndarray,
    vf: np.ndarray,
    selection: np.ndarray,
    log10: bool,
) -> dict[str, object]:
    good = selection & np.isfinite(wisesize) & np.isfinite(vf)
    if log10:
        good &= (wisesize > 0) & (vf > 0)
        delta = np.log10(wisesize[good]) - np.log10(vf[good])
    else:
        delta = wisesize[good] - vf[good]
    row: dict[str, object] = {
        "quantity": quantity,
        "subset": subset,
        "log10": log10,
        "n": len(delta),
    }
    if len(delta) == 0:
        row.update(
            median_delta=np.nan,
            mad_delta=np.nan,
            p16_delta=np.nan,
            p84_delta=np.nan,
            median_ratio=np.nan,
            spearman_rho=np.nan,
        )
        return row
    median = float(np.median(delta))
    row.update(
        median_delta=median,
        mad_delta=float(np.median(np.abs(delta - median))),
        p16_delta=float(np.percentile(delta, 16)),
        p84_delta=float(np.percentile(delta, 84)),
        median_ratio=float(10**median) if log10 else np.nan,
        spearman_rho=float(spearmanr(wisesize[good], vf[good]).statistic),
    )
    return row


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "quantity",
        "subset",
        "log10",
        "n",
        "median_delta",
        "mad_delta",
        "p16_delta",
        "p84_delta",
        "median_ratio",
        "spearman_rho",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def formatted(value: object) -> str:
    return "nan" if not np.isfinite(float(value)) else f"{float(value):.4g}"


def write_summary_markdown(
    path: Path,
    args: argparse.Namespace,
    overlap: Table,
    rows: list[dict[str, object]],
    bayes_best_rows: list[dict[str, object]],
) -> None:
    confirmed = np.asarray(overlap["REDSHIFT_CONFIRMED"], dtype=bool)
    discordant = np.asarray(overlap["REDSHIFT_DISCORDANT"], dtype=bool)
    valid = np.asarray(overlap["VF_CIGALE_VALID"], dtype=bool)
    separation = np.asarray(overlap["SEPARATION_ARCSEC"], dtype=float)
    candidate_count = np.asarray(overlap["VF_CANDIDATE_COUNT"], dtype=int)
    duplicate = np.asarray(overlap["VF_DUPLICATE_ASSIGNMENT"], dtype=bool)
    lines = [
        "# WISEsize vs Virgo Filament CIGALE Comparison",
        "",
        f"WISEsize CIGALE: `{args.cigale}`",
        f"SGA: `{args.sga}`",
        f"VF main: `{args.vf_main}`",
        f"VF CIGALE: `{args.vf_cigale}`",
        "",
        "## Match diagnostics",
        "",
        f"- Positional matches within {args.max_separation_arcsec:g} arcsec: "
        f"{len(overlap)}",
        f"- Median separation: {np.median(separation):.3f} arcsec",
        f"- 95th-percentile separation: {np.percentile(separation, 95):.3f} arcsec",
        f"- Redshift-confirmed matches (|SGA z - VF vr/c| <= "
        f"{args.redshift_confirm_tolerance:g}): {confirmed.sum()}",
        f"- Redshift-discordant matches (difference > "
        f"{args.redshift_discordant_tolerance:g}): {discordant.sum()}",
        f"- Valid VF CIGALE fits: {valid.sum()}",
        f"- Valid, redshift-confirmed comparison sample: {(valid & confirmed).sum()}",
        f"- Matches with multiple VF candidates inside the search radius: "
        f"{(candidate_count > 1).sum()}",
        f"- Objects sharing the same nearest VF row: {duplicate.sum()}",
        "",
        "A VF fit is valid when Bayesian stellar mass, SFR, dust mass, and the "
        "best reduced chi-square are all finite and positive. This excludes "
        "zero-filled and negative placeholder rows.",
        "",
        "## Parameter comparison",
        "",
        "Deltas are WISEsize CIGALE minus VF CIGALE. Mass, SFR, and dust-mass "
        "deltas are logarithmic; the remaining deltas are linear.",
        "",
        "| Quantity | Subset | log10? | N | Median delta | MAD | p16 | p84 | "
        "Median ratio | Spearman rho |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['quantity']} | {row['subset']} | {row['log10']} | "
            f"{row['n']} | {formatted(row['median_delta'])} | "
            f"{formatted(row['mad_delta'])} | {formatted(row['p16_delta'])} | "
            f"{formatted(row['p84_delta'])} | {formatted(row['median_ratio'])} | "
            f"{formatted(row['spearman_rho'])} |"
        )
    lines.extend(
        [
            "",
            "## Bayesian vs best-fit values within each run",
            "",
            "Deltas are log10(Bayesian/best-fit) for the same galaxies and run.",
            "",
            "| Quantity | Catalog | N | Median delta (dex) | MAD (dex) | p16 | "
            "p84 | Median ratio | Spearman rho |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in bayes_best_rows:
        lines.append(
            f"| {row['quantity']} | {row['subset']} | {row['n']} | "
            f"{formatted(row['median_delta'])} | {formatted(row['mad_delta'])} | "
            f"{formatted(row['p16_delta'])} | {formatted(row['p84_delta'])} | "
            f"{formatted(row['median_ratio'])} | {formatted(row['spearman_rho'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def comparison_plot(
    path: Path,
    overlap: Table,
    rows: list[dict[str, object]],
    parameters: list[dict[str, object]],
    shape: tuple[int, int],
) -> None:
    selection = np.asarray(overlap["REDSHIFT_CONFIRMED"], dtype=bool) & np.asarray(
        overlap["VF_CIGALE_VALID"], dtype=bool
    )
    stats = {
        row["quantity"]: row
        for row in rows
        if row["subset"] == "redshift_confirmed_valid"
    }
    fig, axes = plt.subplots(
        *shape,
        figsize=(5.2 * shape[1], 4.5 * shape[0]),
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).flat
    for axis, parameter in zip(axes_flat, parameters):
        short = parameter["short"]
        wisesize = np.asarray(overlap[f"WISE_CIGALE_{short}"], dtype=float)
        vf = np.asarray(overlap[f"VF_CIGALE_{short}"], dtype=float)
        good = selection & np.isfinite(wisesize) & np.isfinite(vf)
        if parameter["log10"]:
            good &= (wisesize > 0) & (vf > 0)
            x = np.log10(vf[good])
            y = np.log10(wisesize[good])
            prefix = "log10 "
        else:
            x = vf[good]
            y = wisesize[good]
            prefix = ""
        plot_percentiles = parameter.get("plot_percentiles", (0.5, 99.5))
        limits = np.percentile(np.concatenate([x, y]), plot_percentiles)
        pad = 0.05 * np.ptp(limits)
        lo, hi = limits[0] - pad, limits[1] + pad
        axis.hexbin(x, y, gridsize=45, bins="log", mincnt=1, cmap="viridis")
        axis.plot([lo, hi], [lo, hi], color="black", lw=1)
        axis.set(xlim=(lo, hi), ylim=(lo, hi))
        axis.set_xlabel(f"{prefix}VF CIGALE {parameter['label']}")
        axis.set_ylabel(f"{prefix}WISEsize CIGALE {parameter['label']}")
        row = stats[parameter["column"]]
        unit = " dex" if parameter["log10"] else ""
        annotation = (
            f"N = {row['n']}\n"
            f"median delta = {row['median_delta']:.3g}{unit}\n"
            f"MAD = {row['mad_delta']:.3g}{unit}\n"
            f"Spearman rho = {row['spearman_rho']:.2f}"
        )
        if plot_percentiles != (0.5, 99.5):
            annotation += (
                f"\naxes: p{plot_percentiles[0]}-p{plot_percentiles[1]}"
            )
        axis.text(
            0.04,
            0.96,
            annotation,
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
    for axis in list(axes_flat):
        axis.set_visible(False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def match_diagnostics_plot(path: Path, overlap: Table, args: argparse.Namespace) -> None:
    separation = np.asarray(overlap["SEPARATION_ARCSEC"], dtype=float)
    delta_z = np.asarray(overlap["DELTA_Z_SGA_MINUS_VF"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].hist(
        separation,
        bins=np.linspace(0, args.max_separation_arcsec, 61),
        histtype="step",
        lw=1.5,
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("SGA-VF separation [arcsec]")
    axes[0].set_ylabel("Number of matches")
    axes[0].axvline(np.median(separation), color="tab:blue", lw=1)

    clipped = np.clip(delta_z, -0.01, 0.01)
    axes[1].hist(clipped, bins=np.linspace(-0.01, 0.01, 101), histtype="step", lw=1.5)
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$z_{SGA} - v_{r,VF}/c$ (clipped at +/-0.01)")
    axes[1].set_ylabel("Number of matches")
    axes[1].axvline(-args.redshift_confirm_tolerance, color="tab:blue", lw=1)
    axes[1].axvline(args.redshift_confirm_tolerance, color="tab:blue", lw=1)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wisesize = Table.read(args.cigale)
    sgaids = np.asarray(wisesize["SGAID"], dtype=np.int64)
    sga = align_sga_to_cigale(args.sga, sgaids)
    vf_main = Table.read(args.vf_main)
    vf_cigale = Table.read(args.vf_cigale)
    validate_vf_row_alignment(vf_main, vf_cigale)

    vf_index, separation, matched, candidate_count, duplicate = match_catalogs(
        sga, vf_main, args.max_separation_arcsec
    )
    wisesize_index = np.flatnonzero(matched)
    vf_index = vf_index[matched]
    separation = separation[matched]
    candidate_count = candidate_count[matched]
    duplicate = duplicate[matched]

    speed_of_light = c.to_value(u.km / u.s)
    vf_vr = np.asarray(vf_main["vr"][vf_index], dtype=float)
    vf_z = vf_vr / speed_of_light
    sga_z = np.asarray(sga["Z"][wisesize_index], dtype=float)
    delta_z = sga_z - vf_z
    redshift_available = (
        np.isfinite(sga_z) & (sga_z > 0) & np.isfinite(vf_z) & (vf_z > 0)
    )
    redshift_confirmed = redshift_available & (
        np.abs(delta_z) <= args.redshift_confirm_tolerance
    )
    redshift_discordant = redshift_available & (
        np.abs(delta_z) > args.redshift_discordant_tolerance
    )
    vf_valid = valid_vf_fit(vf_cigale)[vf_index]

    overlap = Table()
    overlap["SGAID"] = sgaids[wisesize_index]
    if "CIGALE_CHUNK" in wisesize.colnames:
        overlap["CIGALE_CHUNK"] = wisesize["CIGALE_CHUNK"][wisesize_index]
    overlap["VFID"] = vf_main["VFID"][vf_index]
    overlap["VF_INDEX"] = vf_index
    overlap["SGA_RA"] = sga["RA"][wisesize_index]
    overlap["SGA_DEC"] = sga["DEC"][wisesize_index]
    overlap["VF_RA"] = vf_main["RA"][vf_index]
    overlap["VF_DEC"] = vf_main["DEC"][vf_index]
    overlap["SEPARATION_ARCSEC"] = separation
    overlap["VF_CANDIDATE_COUNT"] = candidate_count
    overlap["VF_DUPLICATE_ASSIGNMENT"] = duplicate
    overlap["SGA_Z"] = sga_z
    overlap["SGA_Z_REF"] = sga["Z_REF"][wisesize_index]
    overlap["VF_VR_KMS"] = vf_vr
    overlap["VF_Z_FROM_VR"] = vf_z
    overlap["DELTA_Z_SGA_MINUS_VF"] = delta_z
    overlap["DELTA_V_APPROX_KMS"] = delta_z * speed_of_light
    overlap["REDSHIFT_AVAILABLE"] = redshift_available
    overlap["REDSHIFT_CONFIRMED"] = redshift_confirmed
    overlap["REDSHIFT_DISCORDANT"] = redshift_discordant
    overlap["VF_CIGALE_VALID"] = vf_valid
    overlap["WISE_CIGALE_REDUCED_CHI_SQUARE"] = wisesize[
        "best.reduced_chi_square"
    ][wisesize_index]
    overlap["VF_CIGALE_REDUCED_CHI_SQUARE"] = vf_cigale[
        "best.reduced_chi_square"
    ][vf_index]

    for parameter in PARAMETERS:
        short = parameter["short"]
        column = parameter["column"]
        overlap[f"WISE_CIGALE_{short}"] = wisesize[column][wisesize_index]
        overlap[f"VF_CIGALE_{short}"] = vf_cigale[column][vf_index]
        error_column = f"{column}_err"
        if error_column in wisesize.colnames and error_column in vf_cigale.colnames:
            overlap[f"WISE_CIGALE_{short}_ERR"] = wisesize[error_column][
                wisesize_index
            ]
            overlap[f"VF_CIGALE_{short}_ERR"] = vf_cigale[error_column][vf_index]

    for name in ["SGA_RA", "SGA_DEC", "VF_RA", "VF_DEC"]:
        overlap[name].unit = u.deg
    overlap["SEPARATION_ARCSEC"].unit = u.arcsec
    overlap["VF_VR_KMS"].unit = u.km / u.s
    overlap["DELTA_V_APPROX_KMS"].unit = u.km / u.s

    overlap_path = args.output_dir / "wisesize_vf_cigale_overlap.fits"
    overlap.write(overlap_path, overwrite=True)

    subsets = [
        ("position_matched_valid", vf_valid),
        ("redshift_confirmed_valid", vf_valid & redshift_confirmed),
    ]
    summary_rows = []
    for parameter in PARAMETERS:
        short = parameter["short"]
        wisesize_values = np.asarray(overlap[f"WISE_CIGALE_{short}"], dtype=float)
        vf_values = np.asarray(overlap[f"VF_CIGALE_{short}"], dtype=float)
        for subset_name, selection in subsets:
            summary_rows.append(
                comparison_stats(
                    parameter["column"],
                    subset_name,
                    wisesize_values,
                    vf_values,
                    selection,
                    parameter["log10"],
                )
            )

    primary_selection = vf_valid & redshift_confirmed
    bayes_best_rows = []
    for catalog in ["WISE", "VF"]:
        for quantity, bayes_short, best_short in [
            ("stellar_mass", "MSTAR", "BEST_MSTAR"),
            ("sfr", "SFR", "BEST_SFR"),
        ]:
            bayes_best_rows.append(
                comparison_stats(
                    quantity,
                    catalog,
                    np.asarray(overlap[f"{catalog}_CIGALE_{bayes_short}"], dtype=float),
                    np.asarray(overlap[f"{catalog}_CIGALE_{best_short}"], dtype=float),
                    primary_selection,
                    True,
                )
            )

    write_summary_csv(args.output_dir / "comparison_summary.csv", summary_rows)
    write_summary_csv(args.output_dir / "bayes_best_summary.csv", bayes_best_rows)
    write_summary_markdown(
        args.output_dir / "comparison_summary.md",
        args,
        overlap,
        summary_rows,
        bayes_best_rows,
    )
    comparison_plot(
        args.output_dir / "cigale_parameter_comparison.png",
        overlap,
        summary_rows,
        PRIMARY_PARAMETERS,
        (2, 2),
    )
    comparison_plot(
        args.output_dir / "cigale_other_parameter_comparison.png",
        overlap,
        summary_rows,
        SECONDARY_PARAMETERS,
        (2, 2),
    )
    match_diagnostics_plot(args.output_dir / "match_diagnostics.png", overlap, args)

    print(f"Matched {len(overlap)}/{len(wisesize)} WISEsize objects")
    print(f"Valid, redshift-confirmed comparison sample: {(vf_valid & redshift_confirmed).sum()}")
    print(f"Wrote {overlap_path}")
    print(f"Wrote {args.output_dir / 'comparison_summary.md'}")


if __name__ == "__main__":
    main()
