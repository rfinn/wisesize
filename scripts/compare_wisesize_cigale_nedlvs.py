#!/usr/bin/env python
"""Positionally match WISEsize CIGALE results to NED-LVS and compare properties."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from astropy.coordinates import SkyCoord, match_coordinates_sky, search_around_sky
from astropy.io import fits
from astropy.table import Table
import astropy.units as u


DEFAULT_CIGALE = Path(
    "/Users/rfinn/research/SGA-CIGALE/results/"
    "wisesize_sga2025_ap03_z0002_0025_w3snr10_errfloor0p10mag_results.fits"
)
DEFAULT_SGA = Path("/Users/rfinn/research/SGA2025/SGA2025-v1.0.fits")
DEFAULT_NEDLVS = Path("/Users/rfinn/research/NED-LVS/NEDLVS_20250602.fits")
DEFAULT_OUTPUT_DIR = Path(
    "/Users/rfinn/research/SGA-CIGALE/comparisons/wisesize_nedlvs_20250602"
)

NED_COLUMNS = [
    "objname",
    "ra",
    "dec",
    "z",
    "z_unc",
    "z_qual",
    "SFR_W4",
    "SFR_W4_unc",
    "SFR_hybrid",
    "SFR_hybrid_unc",
    "ET_flag",
    "Mstar",
    "Mstar_unc",
    "MLratio",
    "GALEXphot",
    "WISEphot",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match WISEsize CIGALE results to NED-LVS by position and compare "
            "stellar mass and SFR."
        )
    )
    parser.add_argument("--cigale", type=Path, default=DEFAULT_CIGALE)
    parser.add_argument("--sga", type=Path, default=DEFAULT_SGA)
    parser.add_argument("--nedlvs", type=Path, default=DEFAULT_NEDLVS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-separation-arcsec", type=float, default=30.0)
    parser.add_argument(
        "--redshift-confirm-tolerance",
        type=float,
        default=0.001,
        help="Maximum absolute redshift difference for a confirmed match.",
    )
    parser.add_argument(
        "--redshift-discordant-tolerance",
        type=float,
        default=0.003,
        help="Redshift differences above this value are flagged as discordant.",
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


def fill_unmatched(values: np.ndarray, matched: np.ndarray) -> np.ndarray:
    result = np.array(values, copy=True)
    if result.dtype.kind in "fc":
        result[~matched] = np.nan
    elif result.dtype.kind in "iu":
        result[~matched] = -1
    elif result.dtype.kind == "b":
        result[~matched] = False
    else:
        result = result.astype(str)
        result[~matched] = ""
    return result


def match_nedlvs(
    nedlvs_path: Path,
    sga_ra: np.ndarray,
    sga_dec: np.ndarray,
    sga_z: np.ndarray,
    max_separation_arcsec: float,
    redshift_confirm_tolerance: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    sga_coord = SkyCoord(sga_ra * u.deg, sga_dec * u.deg)
    with fits.open(nedlvs_path, memmap=True) as hdul:
        ned = hdul[1].data
        ned_coord = SkyCoord(
            np.asarray(ned["ra"], dtype=float) * u.deg,
            np.asarray(ned["dec"], dtype=float) * u.deg,
        )
        nearest_index, separation, _ = match_coordinates_sky(sga_coord, ned_coord)
        separation_arcsec = separation.arcsec
        matched = separation_arcsec <= max_separation_arcsec
        sga_candidate_index, ned_candidate_index, candidate_separation, _ = search_around_sky(
            sga_coord,
            ned_coord,
            max_separation_arcsec * u.arcsec,
        )
        candidate_count = np.bincount(
            sga_candidate_index, minlength=len(sga_coord)
        ).astype(np.int16)
        candidate_z = np.asarray(ned["z"])[ned_candidate_index]
        candidate_delta_z = np.abs(sga_z[sga_candidate_index] - candidate_z)
        candidate_confirmed = (
            np.isfinite(sga_z[sga_candidate_index])
            & (sga_z[sga_candidate_index] > 0)
            & np.isfinite(candidate_z)
            & (candidate_z > 0)
            & (candidate_delta_z <= redshift_confirm_tolerance)
        )
        confirmed_candidate_index = np.full(len(sga_coord), -1, dtype=np.int64)
        confirmed_candidate_separation = np.full(len(sga_coord), np.nan, dtype=float)
        for pair_index in np.flatnonzero(candidate_confirmed):
            sga_index = sga_candidate_index[pair_index]
            pair_separation = candidate_separation.arcsec[pair_index]
            if (
                confirmed_candidate_index[sga_index] < 0
                or pair_separation < confirmed_candidate_separation[sga_index]
            ):
                confirmed_candidate_index[sga_index] = ned_candidate_index[pair_index]
                confirmed_candidate_separation[sga_index] = pair_separation
        values = {
            name: fill_unmatched(np.asarray(ned[name])[nearest_index], matched)
            for name in NED_COLUMNS
        }

    ned_index = np.asarray(nearest_index, dtype=np.int64)
    ned_index[~matched] = -1
    return (
        ned_index,
        separation_arcsec,
        matched,
        values,
        candidate_count,
        confirmed_candidate_index,
        confirmed_candidate_separation,
    )


def duplicate_flags(indices: np.ndarray, matched: np.ndarray) -> np.ndarray:
    flags = np.zeros(len(indices), dtype=bool)
    valid_indices = indices[matched]
    unique, counts = np.unique(valid_indices, return_counts=True)
    duplicated = unique[counts > 1]
    if len(duplicated):
        flags[matched] = np.isin(valid_indices, duplicated)
    return flags


def finite_positive_pair(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return np.isfinite(left) & (left > 0) & np.isfinite(right) & (right > 0)


def summarize_delta(
    quantity: str,
    subset: str,
    cigale_values: np.ndarray,
    ned_values: np.ndarray,
    selection: np.ndarray,
) -> dict[str, object]:
    good = selection & finite_positive_pair(cigale_values, ned_values)
    delta = np.log10(np.asarray(cigale_values)[good]) - np.log10(np.asarray(ned_values)[good])
    row: dict[str, object] = {"quantity": quantity, "subset": subset, "n": len(delta)}
    if len(delta) == 0:
        row.update(
            median_delta_dex=np.nan,
            mad_delta_dex=np.nan,
            p16_delta_dex=np.nan,
            p84_delta_dex=np.nan,
            median_ratio=np.nan,
        )
        return row
    median = float(np.median(delta))
    row.update(
        median_delta_dex=median,
        mad_delta_dex=float(np.median(np.abs(delta - median))),
        p16_delta_dex=float(np.percentile(delta, 16)),
        p84_delta_dex=float(np.percentile(delta, 84)),
        median_ratio=float(10**median),
    )
    return row


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "quantity",
        "subset",
        "n",
        "median_delta_dex",
        "mad_delta_dex",
        "p16_delta_dex",
        "p84_delta_dex",
        "median_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_number(value: object) -> str:
    return "nan" if not np.isfinite(float(value)) else f"{float(value):.4g}"


def write_summary_markdown(
    path: Path,
    args: argparse.Namespace,
    table: Table,
    rows: list[dict[str, object]],
) -> None:
    matched = np.asarray(table["NEDLVS_MATCH"], dtype=bool)
    redshift_available = np.asarray(table["REDSHIFT_AVAILABLE"], dtype=bool)
    confirmed = np.asarray(table["REDSHIFT_CONFIRMED"], dtype=bool)
    discordant = np.asarray(table["REDSHIFT_DISCORDANT"], dtype=bool)
    candidate_count = np.asarray(table["NEDLVS_CANDIDATE_COUNT"], dtype=int)
    has_confirmed_alternative = np.asarray(
        table["NEDLVS_HAS_CONFIRMED_ALTERNATIVE"], dtype=bool
    )
    separations = np.asarray(table["NEDLVS_SEP_ARCSEC"], dtype=float)[matched]
    lines = [
        "# WISEsize CIGALE vs NED-LVS",
        "",
        f"CIGALE: `{args.cigale}`",
        f"SGA: `{args.sga}`",
        f"NED-LVS: `{args.nedlvs}`",
        "",
        "## Match diagnostics",
        "",
        f"- CIGALE objects: {len(table)}",
        f"- NED-LVS matches within {args.max_separation_arcsec:g} arcsec: "
        f"{matched.sum()} ({100 * matched.mean():.2f}%)",
        f"- Median matched separation: {np.median(separations):.3f} arcsec",
        f"- 95th-percentile matched separation: {np.percentile(separations, 95):.3f} arcsec",
        f"- Matches with two usable redshifts: {(matched & redshift_available).sum()}",
        f"- Redshift-confirmed matches (|delta z| <= {args.redshift_confirm_tolerance:g}): "
        f"{confirmed.sum()}",
        f"- Redshift-discordant matches (|delta z| > {args.redshift_discordant_tolerance:g}): "
        f"{discordant.sum()}",
        f"- Matches with more than one NED-LVS candidate inside the search radius: "
        f"{(matched & (candidate_count > 1)).sum()}",
        f"- Discordant nearest matches with a redshift-confirmed alternative: "
        f"{(discordant & has_confirmed_alternative).sum()}",
        f"- Objects sharing the same nearest NED-LVS row: "
        f"{np.asarray(table['NEDLVS_DUPLICATE_ASSIGNMENT'], dtype=bool).sum()}",
        "",
        "Redshift is a confirmation diagnostic; the selected source is always the nearest "
        "NED-LVS position within the stated separation limit.",
        "",
        "## Property comparison",
        "",
        "Deltas are log10(CIGALE) - log10(NED-LVS). `hybrid_or_W4` uses "
        "NED-LVS `SFR_hybrid` when available and otherwise `SFR_W4`.",
        "",
        "| Quantity | Subset | N | Median delta (dex) | MAD (dex) | p16 | p84 | Median ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['quantity']} | {row['subset']} | {row['n']} | "
            f"{format_number(row['median_delta_dex'])} | "
            f"{format_number(row['mad_delta_dex'])} | "
            f"{format_number(row['p16_delta_dex'])} | "
            f"{format_number(row['p84_delta_dex'])} | "
            f"{format_number(row['median_ratio'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def comparison_plot(
    path: Path,
    table: Table,
    rows: list[dict[str, object]],
) -> None:
    confirmed = np.asarray(table["REDSHIFT_CONFIRMED"], dtype=bool)
    quantities = [
        ("stellar_mass", "CIGALE_MSTAR", "NEDLVS_MSTAR", r"$M_\star$ [$M_\odot$]"),
        ("sfr_hybrid", "CIGALE_SFR", "NEDLVS_SFR_HYBRID", r"SFR [$M_\odot$ yr$^{-1}$]"),
        (
            "sfr_hybrid_or_W4",
            "CIGALE_SFR",
            "NEDLVS_SFR_HYBRID_OR_W4",
            r"SFR [$M_\odot$ yr$^{-1}$]",
        ),
    ]
    stats = {
        row["quantity"]: row
        for row in rows
        if row["subset"] == "redshift_confirmed"
    }
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for column, (quantity, cigale_name, ned_name, label) in enumerate(quantities):
        cigale = np.asarray(table[cigale_name], dtype=float)
        ned = np.asarray(table[ned_name], dtype=float)
        good = confirmed & finite_positive_pair(cigale, ned)
        x = np.log10(ned[good])
        y = np.log10(cigale[good])
        delta = y - x
        limits = np.percentile(np.concatenate([x, y]), [0.5, 99.5])
        pad = 0.05 * np.ptp(limits)
        lo, hi = limits[0] - pad, limits[1] + pad

        axes[0, column].hexbin(x, y, gridsize=65, bins="log", mincnt=1, cmap="viridis")
        axes[0, column].plot([lo, hi], [lo, hi], color="black", lw=1)
        axes[0, column].set(xlim=(lo, hi), ylim=(lo, hi))
        axes[0, column].set_xlabel(f"log10 NED-LVS {label}")
        axes[0, column].set_ylabel(f"log10 CIGALE {label}")
        row = stats[quantity]
        axes[0, column].text(
            0.04,
            0.96,
            (
                f"N = {row['n']}\n"
                f"median delta = {row['median_delta_dex']:.2f} dex\n"
                f"MAD = {row['mad_delta_dex']:.2f} dex"
            ),
            transform=axes[0, column].transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )

        axes[1, column].hexbin(x, delta, gridsize=65, bins="log", mincnt=1, cmap="magma")
        axes[1, column].axhline(0, color="black", lw=1)
        axes[1, column].axhline(row["median_delta_dex"], color="tab:blue", lw=1)
        axes[1, column].set_xlabel(f"log10 NED-LVS {label}")
        axes[1, column].set_ylabel("log10 CIGALE - log10 NED-LVS")
        axes[1, column].set_xlim(lo, hi)
        axes[1, column].set_ylim(np.percentile(delta, [0.5, 99.5]))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def match_diagnostics_plot(
    path: Path,
    table: Table,
    confirm_tolerance: float,
    max_separation_arcsec: float,
) -> None:
    matched = np.asarray(table["NEDLVS_MATCH"], dtype=bool)
    available = np.asarray(table["REDSHIFT_AVAILABLE"], dtype=bool)
    separation = np.asarray(table["NEDLVS_SEP_ARCSEC"], dtype=float)
    delta_z = np.asarray(table["DELTA_Z_SGA_MINUS_NEDLVS"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].hist(
        separation[matched],
        bins=np.linspace(0, max_separation_arcsec, 61),
        histtype="step",
        lw=1.5,
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("SGA-NED-LVS separation [arcsec]")
    axes[0].set_ylabel("Number of matches")
    axes[0].axvline(np.median(separation[matched]), color="tab:blue", lw=1)

    redshift_sample = matched & available
    clipped = np.clip(delta_z[redshift_sample], -0.01, 0.01)
    axes[1].hist(clipped, bins=np.linspace(-0.01, 0.01, 101), histtype="step", lw=1.5)
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$z_{SGA} - z_{NED-LVS}$ (clipped at +/-0.01)")
    axes[1].set_ylabel("Number of matches")
    axes[1].axvline(-confirm_tolerance, color="tab:blue", lw=1)
    axes[1].axvline(confirm_tolerance, color="tab:blue", lw=1)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cigale = Table.read(args.cigale)
    sgaids = np.asarray(cigale["SGAID"], dtype=np.int64)
    sga = align_sga_to_cigale(args.sga, sgaids)
    (
        ned_index,
        separation,
        matched,
        ned,
        candidate_count,
        confirmed_candidate_index,
        confirmed_candidate_separation,
    ) = match_nedlvs(
        args.nedlvs,
        sga["RA"],
        sga["DEC"],
        sga["Z"],
        args.max_separation_arcsec,
        args.redshift_confirm_tolerance,
    )

    ned_z = np.asarray(ned["z"], dtype=float)
    sga_z = np.asarray(sga["Z"], dtype=float)
    redshift_available = (
        matched
        & np.isfinite(sga_z)
        & (sga_z > 0)
        & np.isfinite(ned_z)
        & (ned_z > 0)
    )
    delta_z = sga_z - ned_z
    redshift_confirmed = redshift_available & (
        np.abs(delta_z) <= args.redshift_confirm_tolerance
    )
    redshift_discordant = redshift_available & (
        np.abs(delta_z) > args.redshift_discordant_tolerance
    )
    has_confirmed_alternative = (
        matched
        & (confirmed_candidate_index >= 0)
        & (confirmed_candidate_index != ned_index)
    )
    duplicate_assignment = duplicate_flags(ned_index, matched)

    ned_sfr_hybrid = np.asarray(ned["SFR_hybrid"], dtype=float)
    ned_sfr_w4 = np.asarray(ned["SFR_W4"], dtype=float)
    use_hybrid = np.isfinite(ned_sfr_hybrid) & (ned_sfr_hybrid > 0)
    ned_sfr_combined = np.where(use_hybrid, ned_sfr_hybrid, ned_sfr_w4)
    ned_sfr_method = np.full(len(cigale), "", dtype="U6")
    ned_sfr_method[matched & use_hybrid] = "hybrid"
    ned_sfr_method[matched & ~use_hybrid & np.isfinite(ned_sfr_w4) & (ned_sfr_w4 > 0)] = "W4"

    output = Table()
    output["SGAID"] = sgaids
    if "CIGALE_CHUNK" in cigale.colnames:
        output["CIGALE_CHUNK"] = cigale["CIGALE_CHUNK"]
    output["SGA_RA"] = sga["RA"]
    output["SGA_DEC"] = sga["DEC"]
    output["SGA_Z"] = sga_z
    output["SGA_Z_REF"] = sga["Z_REF"]
    output["NEDLVS_MATCH"] = matched
    output["NEDLVS_INDEX"] = ned_index
    output["NEDLVS_SEP_ARCSEC"] = separation
    output["NEDLVS_CANDIDATE_COUNT"] = candidate_count
    output["NEDLVS_DUPLICATE_ASSIGNMENT"] = duplicate_assignment
    output["NEDLVS_OBJNAME"] = ned["objname"]
    output["NEDLVS_RA"] = ned["ra"]
    output["NEDLVS_DEC"] = ned["dec"]
    output["NEDLVS_Z"] = ned_z
    output["NEDLVS_Z_UNC"] = ned["z_unc"]
    output["NEDLVS_Z_QUAL"] = ned["z_qual"]
    output["DELTA_Z_SGA_MINUS_NEDLVS"] = delta_z
    output["REDSHIFT_AVAILABLE"] = redshift_available
    output["REDSHIFT_CONFIRMED"] = redshift_confirmed
    output["REDSHIFT_DISCORDANT"] = redshift_discordant
    output["NEDLVS_CONFIRMED_CANDIDATE_INDEX"] = confirmed_candidate_index
    output["NEDLVS_CONFIRMED_CANDIDATE_SEP_ARCSEC"] = confirmed_candidate_separation
    output["NEDLVS_HAS_CONFIRMED_ALTERNATIVE"] = has_confirmed_alternative
    output["CIGALE_MSTAR"] = cigale["bayes.stellar.m_star"]
    output["CIGALE_MSTAR_ERR"] = cigale["bayes.stellar.m_star_err"]
    output["CIGALE_SFR"] = cigale["bayes.sfh.sfr"]
    output["CIGALE_SFR_ERR"] = cigale["bayes.sfh.sfr_err"]
    output["CIGALE_REDUCED_CHI_SQUARE"] = cigale["best.reduced_chi_square"]
    output["NEDLVS_MSTAR"] = ned["Mstar"]
    output["NEDLVS_MSTAR_UNC"] = ned["Mstar_unc"]
    output["NEDLVS_MLRATIO"] = ned["MLratio"]
    output["NEDLVS_SFR_HYBRID"] = ned_sfr_hybrid
    output["NEDLVS_SFR_HYBRID_UNC"] = ned["SFR_hybrid_unc"]
    output["NEDLVS_SFR_W4"] = ned_sfr_w4
    output["NEDLVS_SFR_W4_UNC"] = ned["SFR_W4_unc"]
    output["NEDLVS_SFR_HYBRID_OR_W4"] = ned_sfr_combined
    output["NEDLVS_SFR_METHOD"] = ned_sfr_method
    output["NEDLVS_ET_FLAG"] = ned["ET_flag"]
    output["NEDLVS_GALEX_PHOT"] = ned["GALEXphot"]
    output["NEDLVS_WISE_PHOT"] = ned["WISEphot"]

    output["SGA_RA"].unit = u.deg
    output["SGA_DEC"].unit = u.deg
    output["NEDLVS_RA"].unit = u.deg
    output["NEDLVS_DEC"].unit = u.deg
    output["NEDLVS_SEP_ARCSEC"].unit = u.arcsec
    output["NEDLVS_CONFIRMED_CANDIDATE_SEP_ARCSEC"].unit = u.arcsec
    for name in ["CIGALE_MSTAR", "CIGALE_MSTAR_ERR", "NEDLVS_MSTAR", "NEDLVS_MSTAR_UNC"]:
        output[name].unit = u.Msun
    for name in [
        "CIGALE_SFR",
        "CIGALE_SFR_ERR",
        "NEDLVS_SFR_HYBRID",
        "NEDLVS_SFR_HYBRID_UNC",
        "NEDLVS_SFR_W4",
        "NEDLVS_SFR_W4_UNC",
        "NEDLVS_SFR_HYBRID_OR_W4",
    ]:
        output[name].unit = u.Msun / u.yr

    crossmatch_path = args.output_dir / "wisesize_cigale_nedlvs_crossmatch.fits"
    output.write(crossmatch_path, overwrite=True)

    cigale_mass = np.asarray(output["CIGALE_MSTAR"], dtype=float)
    cigale_sfr = np.asarray(output["CIGALE_SFR"], dtype=float)
    non_et = ~np.asarray(output["NEDLVS_ET_FLAG"], dtype=bool)
    comparisons = [
        ("stellar_mass", cigale_mass, np.asarray(output["NEDLVS_MSTAR"], dtype=float)),
        ("sfr_hybrid", cigale_sfr, np.asarray(output["NEDLVS_SFR_HYBRID"], dtype=float)),
        ("sfr_W4", cigale_sfr, np.asarray(output["NEDLVS_SFR_W4"], dtype=float)),
        (
            "sfr_hybrid_or_W4",
            cigale_sfr,
            np.asarray(output["NEDLVS_SFR_HYBRID_OR_W4"], dtype=float),
        ),
    ]
    subsets = [
        ("position_matched", matched),
        ("redshift_confirmed", redshift_confirmed),
        ("redshift_confirmed_non_ET", redshift_confirmed & non_et),
    ]
    summary_rows = [
        summarize_delta(quantity, subset, cigale_values, ned_values, selection)
        for quantity, cigale_values, ned_values in comparisons
        for subset, selection in subsets
    ]
    write_summary_csv(args.output_dir / "comparison_summary.csv", summary_rows)
    write_summary_markdown(
        args.output_dir / "comparison_summary.md", args, output, summary_rows
    )
    comparison_plot(args.output_dir / "cigale_vs_nedlvs.png", output, summary_rows)
    match_diagnostics_plot(
        args.output_dir / "match_diagnostics.png",
        output,
        args.redshift_confirm_tolerance,
        args.max_separation_arcsec,
    )

    print(
        f"Matched {matched.sum()}/{len(output)} objects within "
        f"{args.max_separation_arcsec:g} arcsec"
    )
    print(f"Redshift-confirmed: {redshift_confirmed.sum()}")
    print(f"Wrote {crossmatch_path}")
    print(f"Wrote {args.output_dir / 'comparison_summary.md'}")


if __name__ == "__main__":
    main()
