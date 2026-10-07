#!/usr/bin/env python3
"""Cross-match SGA-CIGALE with Durbala et al. and compare masses/SFRs.

Notebook example
----------------
>>> from scripts.compare_cigale_durbala import (
...     match_cigale_durbala,
...     plot_mstar_comparison,
...     plot_mstar_residual_vs_axis_ratio,
...     plot_sfr_comparison,
...     plot_sfr_residual_vs_axis_ratio,
... )
>>> matched = match_cigale_durbala()
>>> fig_mass, axes_mass = plot_mstar_comparison(matched)
>>> fig_sfr, axes_sfr = plot_sfr_comparison(matched)
>>> fig_mass_ba, axes_mass_ba = plot_mstar_residual_vs_axis_ratio(matched)
>>> fig_sfr_ba, axes_sfr_ba = plot_sfr_residual_vs_axis_ratio(matched)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr


SPEED_OF_LIGHT_KMS = 299_792.458
DEFAULT_CIGALE = Path(
    "/Users/rfinn/research/SGA-CIGALE/results/"
    "sga2025_ap04_zlt0040_parent_errfloor0p10mag_"
    "agnfrac0to0p5_results.fits"
)
DEFAULT_DURBALA = Path(
    "/Users/rfinn/research/APPSS/tables/durbala_combined_table1_table2.fits"
)
DEFAULT_OUTPUT_DIR = Path(
    "/Users/rfinn/research/SGA-CIGALE/comparisons/durbala"
)

MASS_ESTIMATORS = (
    ("logMstarTaylor", "Taylor"),
    ("logMstarMcGaugh", "McGaugh"),
    ("logMstarGSWLC", "GSWLC"),
)
SFR_ESTIMATORS = (
    ("logSFR22", r"22 $\mu$m"),
    ("logSFRNUVIR", "NUV+IR"),
    ("logSFRGSWLC", "GSWLC"),
)
DURBALA_COLUMNS = (
    "AGC_1",
    "RA",
    "DEC",
    "Vhelio",
    "Dist",
    "expAB_r",
    "logMstarTaylor",
    "logMstarTaylor_err",
    "logMstarMcGaugh",
    "logMstarMcGaugh_err",
    "logMstarGSWLC",
    "logMstarGSWLC_err",
    "logSFR22",
    "logSFR22_err",
    "logSFRNUVIR",
    "logSFRNUVIR_err",
    "logSFRGSWLC",
    "logSFRGSWLC_err",
)
CIGALE_COLUMNS = (
    "SGAID",
    "RA",
    "DEC",
    "Z",
    "Z_COSMO",
    "DIST",
    "BA",
    "PA",
    "CIGALE_SAMPLE",
    "WISESIZE_SELECTED",
    "best.reduced_chi_square",
    "bayes.stellar.m_star",
    "bayes.stellar.m_star_err",
    "best.stellar.m_star",
    "bayes.sfh.sfr",
    "bayes.sfh.sfr_err",
    "best.sfh.sfr",
)


def _log10_with_error(
    value: np.ndarray, error: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    value = np.asarray(value, dtype=float)
    logged = np.full(value.shape, np.nan, dtype=float)
    valid = np.isfinite(value) & (value > 0.0)
    logged[valid] = np.log10(value[valid])
    if error is None:
        return logged, None
    error = np.asarray(error, dtype=float)
    logged_error = np.full(error.shape, np.nan, dtype=float)
    valid_error = valid & np.isfinite(error) & (error >= 0.0)
    logged_error[valid_error] = (
        error[valid_error] / value[valid_error] / np.log(10.0)
    )
    return logged, logged_error


def _select_unique_matches(
    cigale_index: np.ndarray,
    separation_arcsec: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    ordered = candidate[np.argsort(separation_arcsec[candidate], kind="stable")]
    _, first = np.unique(cigale_index[ordered], return_index=True)
    return np.sort(ordered[first])


def match_cigale_durbala(
    cigale_path: str | Path = DEFAULT_CIGALE,
    durbala_path: str | Path = DEFAULT_DURBALA,
    max_sep_arcsec: float = 30.0,
    max_velocity_offset_kms: float | None = 300.0,
    unique_sga: bool = True,
    verbose: bool = True,
) -> Table:
    """Return coordinate- and velocity-matched CIGALE/Durbala measurements.

    Durbala coordinates are matched to the nearest SGA source. A match must be
    within ``max_sep_arcsec`` and, by default, within 300 km/s using
    ``c * SGA.Z - Durbala.Vhelio``. When ``unique_sga`` is true, only the
    closest Durbala match to each SGA object is retained.
    """
    cigale_path = Path(cigale_path).expanduser()
    durbala_path = Path(durbala_path).expanduser()
    if max_sep_arcsec <= 0.0:
        raise ValueError("max_sep_arcsec must be positive.")
    if max_velocity_offset_kms is not None and max_velocity_offset_kms <= 0.0:
        raise ValueError("max_velocity_offset_kms must be positive or None.")

    cigale = Table.read(cigale_path)
    durbala = Table.read(durbala_path)
    for name in CIGALE_COLUMNS:
        if name not in cigale.colnames:
            raise KeyError(f"Missing CIGALE column: {name}")
    for name in DURBALA_COLUMNS:
        if name not in durbala.colnames:
            raise KeyError(f"Missing Durbala column: {name}")

    cigale_coord = SkyCoord(
        np.asarray(cigale["RA"], float) * u.deg,
        np.asarray(cigale["DEC"], float) * u.deg,
    )
    durbala_coord = SkyCoord(
        np.asarray(durbala["RA"], float) * u.deg,
        np.asarray(durbala["DEC"], float) * u.deg,
    )
    cigale_index, separation, _ = durbala_coord.match_to_catalog_sky(cigale_coord)
    separation_arcsec = separation.arcsec
    velocity_offset = (
        SPEED_OF_LIGHT_KMS * np.asarray(cigale["Z"], float)[cigale_index]
        - np.asarray(durbala["Vhelio"], float)
    )
    keep = np.isfinite(separation_arcsec) & (separation_arcsec <= max_sep_arcsec)
    if max_velocity_offset_kms is not None:
        keep &= np.isfinite(velocity_offset) & (
            np.abs(velocity_offset) <= max_velocity_offset_kms
        )
    durbala_rows = np.flatnonzero(keep)
    n_candidates = len(durbala_rows)
    if unique_sga:
        durbala_rows = _select_unique_matches(
            cigale_index, separation_arcsec, durbala_rows
        )
    cigale_rows = cigale_index[durbala_rows]

    matched = Table()
    for name in CIGALE_COLUMNS:
        matched[f"CIGALE_{name}"] = cigale[name][cigale_rows]
    matched["MATCH_SEP_ARCSEC"] = separation_arcsec[durbala_rows]
    matched["DELTA_V_KMS"] = velocity_offset[durbala_rows]
    for name in DURBALA_COLUMNS:
        output_name = "AGC" if name == "AGC_1" else f"DURBALA_{name}"
        matched[output_name] = durbala[name][durbala_rows]

    log_mass, log_mass_err = _log10_with_error(
        matched["CIGALE_bayes.stellar.m_star"],
        matched["CIGALE_bayes.stellar.m_star_err"],
    )
    matched["CIGALE_LOGMSTAR_BAYES"] = log_mass
    matched["CIGALE_LOGMSTAR_BAYES_ERR"] = log_mass_err
    matched["CIGALE_LOGMSTAR_BEST"] = _log10_with_error(
        matched["CIGALE_best.stellar.m_star"]
    )[0]
    log_sfr, log_sfr_err = _log10_with_error(
        matched["CIGALE_bayes.sfh.sfr"],
        matched["CIGALE_bayes.sfh.sfr_err"],
    )
    matched["CIGALE_LOGSFR_BAYES"] = log_sfr
    matched["CIGALE_LOGSFR_BAYES_ERR"] = log_sfr_err
    matched["CIGALE_LOGSFR_BEST"] = _log10_with_error(
        matched["CIGALE_best.sfh.sfr"]
    )[0]
    matched.meta["MAXSEP"] = (max_sep_arcsec, "Maximum sky separation in arcsec")
    matched.meta["MAXDV"] = (
        max_velocity_offset_kms if max_velocity_offset_kms is not None else -1.0,
        "Max |cZ - Vhelio| (km/s); -1 disables",
    )
    matched.meta["UNIQSGA"] = (unique_sga, "At most one Durbala row per SGAID")

    if verbose:
        print(f"Durbala rows: {len(durbala):,}")
        print(f"Candidates passing cuts: {n_candidates:,}")
        if unique_sga:
            print(f"Duplicate-SGA matches removed: {n_candidates - len(matched):,}")
        print(f"Final matches: {len(matched):,}")
        print(
            "Median separation / |delta-v|: "
            f"{np.median(matched['MATCH_SEP_ARCSEC']):.2f} arcsec / "
            f"{np.median(np.abs(matched['DELTA_V_KMS'])):.1f} km/s"
        )
    return matched


def _comparison_limits(
    matched: Table,
    estimators: tuple[tuple[str, str], ...],
    cigale_columns: tuple[tuple[str, str], ...],
    limits: tuple[float, float] | None,
) -> tuple[float, float]:
    if limits is not None:
        if limits[0] >= limits[1]:
            raise ValueError("limits must be increasing.")
        return limits
    values = []
    for estimator, _ in estimators:
        values.append(np.asarray(matched[f"DURBALA_{estimator}"], float))
    for column, _ in cigale_columns:
        values.append(np.asarray(matched[column], float))
    finite = np.concatenate([value[np.isfinite(value)] for value in values])
    low, high = np.percentile(finite, [0.25, 99.75])
    low = np.floor((low - 0.1) * 2.0) / 2.0
    high = np.ceil((high + 0.1) * 2.0) / 2.0
    return float(low), float(high)


def _plot_grid(
    matched: Table,
    estimators: tuple[tuple[str, str], ...],
    cigale_columns: tuple[tuple[str, str], ...],
    quantity_label: str,
    bins: int,
    limits: tuple[float, float] | None,
    cmap: str,
) -> tuple[plt.Figure, np.ndarray]:
    if bins < 5:
        raise ValueError("bins must be at least 5.")
    low, high = _comparison_limits(matched, estimators, cigale_columns, limits)
    edges = np.linspace(low, high, bins + 1)
    histograms: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    max_count = 1.0
    for row, (cigale_column, _) in enumerate(cigale_columns):
        y = np.asarray(matched[cigale_column], float)
        for col, (estimator, _) in enumerate(estimators):
            x = np.asarray(matched[f"DURBALA_{estimator}"], float)
            finite = np.isfinite(x) & np.isfinite(y)
            histogram, _, _ = np.histogram2d(x[finite], y[finite], bins=(edges, edges))
            histograms[(row, col)] = (histogram, x[finite], y[finite])
            max_count = max(max_count, float(np.max(histogram)))

    fig, axes = plt.subplots(
        len(cigale_columns),
        len(estimators),
        figsize=(13.5, 8.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    norm = LogNorm(vmin=1.0, vmax=max_count)
    mesh = None
    for row, (_, cigale_label) in enumerate(cigale_columns):
        for col, (estimator, estimator_label) in enumerate(estimators):
            ax = axes[row, col]
            histogram, x, y = histograms[(row, col)]
            mesh = ax.pcolormesh(
                edges,
                edges,
                np.ma.masked_equal(histogram.T, 0.0),
                cmap=cmap,
                norm=norm,
                shading="auto",
            )
            ax.plot([low, high], [low, high], color="0.25", lw=1.2, ls="--")
            delta = y - x
            median = np.median(delta) if len(delta) else np.nan
            mad = 1.4826 * np.median(np.abs(delta - median)) if len(delta) else np.nan
            ax.text(
                0.04,
                0.96,
                f"N = {len(delta):,}\nmedian $\\Delta$ = {median:+.2f}\nMAD = {mad:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            )
            if row == 0:
                ax.set_title(f"Durbala {estimator_label}")
            if row == len(cigale_columns) - 1:
                ax.set_xlabel(f"Durbala {quantity_label}")
            if col == 0:
                ax.set_ylabel(f"CIGALE {cigale_label} {quantity_label}")
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            ax.grid(color="0.9", lw=0.6)
    if mesh is not None:
        fig.colorbar(mesh, ax=axes.ravel().tolist(), label="Galaxies per bin")
    return fig, axes


def plot_mstar_comparison(
    matched: Table,
    bins: int = 55,
    limits: tuple[float, float] | None = None,
    cmap: str = "viridis",
) -> tuple[plt.Figure, np.ndarray]:
    """Plot CIGALE Bayes/best log stellar masses against three estimators."""
    return _plot_grid(
        matched,
        MASS_ESTIMATORS,
        (
            ("CIGALE_LOGMSTAR_BAYES", "Bayes"),
            ("CIGALE_LOGMSTAR_BEST", "best"),
        ),
        r"$\log_{10}(M_\star/M_\odot)$",
        bins,
        limits,
        cmap,
    )


def plot_sfr_comparison(
    matched: Table,
    bins: int = 55,
    limits: tuple[float, float] | None = None,
    cmap: str = "magma",
) -> tuple[plt.Figure, np.ndarray]:
    """Plot CIGALE Bayes/best log SFRs against three Durbala estimators."""
    return _plot_grid(
        matched,
        SFR_ESTIMATORS,
        (
            ("CIGALE_LOGSFR_BAYES", "Bayes"),
            ("CIGALE_LOGSFR_BEST", "best"),
        ),
        r"$\log_{10}(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})$",
        bins,
        limits,
        cmap,
    )


def _symmetric_residual_limits(
    matched: Table,
    estimators: tuple[tuple[str, str], ...],
    cigale_columns: tuple[tuple[str, str], ...],
    limits: tuple[float, float] | None,
) -> tuple[float, float]:
    if limits is not None:
        if limits[0] >= limits[1]:
            raise ValueError("limits must be increasing.")
        return limits
    residuals = []
    for cigale_column, _ in cigale_columns:
        y = np.asarray(matched[cigale_column], float)
        for estimator, _ in estimators:
            x = np.asarray(matched[f"DURBALA_{estimator}"], float)
            residual = y - x
            residuals.append(residual[np.isfinite(residual)])
    values = np.concatenate(residuals)
    low, high = np.percentile(values, [2.0, 98.0])
    bound = max(abs(low), abs(high), 0.5)
    bound = np.ceil(bound * 2.0) / 2.0
    return -float(bound), float(bound)


def _binned_residual_percentiles(
    axis_ratio: np.ndarray,
    residual: np.ndarray,
    edges: np.ndarray,
    min_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    median = np.full(len(centers), np.nan)
    p16 = np.full(len(centers), np.nan)
    p84 = np.full(len(centers), np.nan)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = (axis_ratio >= low) & (axis_ratio < high)
        if index == len(centers) - 1:
            in_bin |= axis_ratio == high
        if np.count_nonzero(in_bin) >= min_count:
            p16[index], median[index], p84[index] = np.percentile(
                residual[in_bin], [16.0, 50.0, 84.0]
            )
    return centers, median, p16, p84


def _plot_residual_vs_axis_ratio(
    matched: Table,
    estimators: tuple[tuple[str, str], ...],
    cigale_columns: tuple[tuple[str, str], ...],
    residual_label: str,
    bins: int,
    axis_ratio_bins: int,
    min_bin_count: int,
    limits: tuple[float, float] | None,
    cmap: str,
) -> tuple[plt.Figure, np.ndarray]:
    if bins < 5 or axis_ratio_bins < 3:
        raise ValueError("bins must be >=5 and axis_ratio_bins must be >=3.")
    if min_bin_count < 1:
        raise ValueError("min_bin_count must be positive.")

    residual_low, residual_high = _symmetric_residual_limits(
        matched, estimators, cigale_columns, limits
    )
    axis_ratio_edges = np.linspace(0.0, 1.0, axis_ratio_bins + 1)
    density_x_edges = np.linspace(0.0, 1.0, bins + 1)
    density_y_edges = np.linspace(residual_low, residual_high, bins + 1)
    axis_ratio = np.asarray(matched["CIGALE_BA"], float)
    panels: dict[
        tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    max_count = 1.0
    for row, (cigale_column, _) in enumerate(cigale_columns):
        cigale_value = np.asarray(matched[cigale_column], float)
        for col, (estimator, _) in enumerate(estimators):
            durbala_value = np.asarray(matched[f"DURBALA_{estimator}"], float)
            residual = cigale_value - durbala_value
            finite = (
                np.isfinite(axis_ratio)
                & (axis_ratio > 0.0)
                & (axis_ratio <= 1.0)
                & np.isfinite(residual)
            )
            x = axis_ratio[finite]
            y = residual[finite]
            histogram, _, _ = np.histogram2d(
                x, y, bins=(density_x_edges, density_y_edges)
            )
            panels[(row, col)] = (histogram, x, y)
            max_count = max(max_count, float(np.max(histogram)))

    fig, axes = plt.subplots(
        len(cigale_columns),
        len(estimators),
        figsize=(13.5, 8.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    norm = LogNorm(vmin=1.0, vmax=max_count)
    mesh = None
    for row, (_, cigale_label) in enumerate(cigale_columns):
        for col, (_, estimator_label) in enumerate(estimators):
            ax = axes[row, col]
            histogram, x, residual = panels[(row, col)]
            mesh = ax.pcolormesh(
                density_x_edges,
                density_y_edges,
                np.ma.masked_equal(histogram.T, 0.0),
                cmap=cmap,
                norm=norm,
                shading="auto",
            )
            centers, median, p16, p84 = _binned_residual_percentiles(
                x, residual, axis_ratio_edges, min_bin_count
            )
            good_bins = np.isfinite(median)
            ax.fill_between(
                centers[good_bins],
                p16[good_bins],
                p84[good_bins],
                color="white",
                alpha=0.28,
                linewidth=0.0,
            )
            ax.plot(
                centers[good_bins],
                median[good_bins],
                color="white",
                lw=2.2,
                marker="o",
                ms=3.5,
            )
            ax.axhline(0.0, color="0.35", lw=1.2, ls="--")
            rho, _ = spearmanr(x, residual)
            ax.text(
                0.04,
                0.96,
                f"N = {len(residual):,}\n$\\rho_s$ = {rho:+.2f}\n"
                f"median $\\Delta$ = {np.median(residual):+.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            )
            if row == 0:
                ax.set_title(f"Durbala {estimator_label}")
            if row == len(cigale_columns) - 1:
                ax.set_xlabel(r"SGA axis ratio $b/a$")
            if col == 0:
                ax.set_ylabel(f"CIGALE {cigale_label} − Durbala\n{residual_label}")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(residual_low, residual_high)
            ax.grid(color="0.9", lw=0.6)
    if mesh is not None:
        fig.colorbar(mesh, ax=axes.ravel().tolist(), label="Galaxies per bin")
    return fig, axes


def plot_mstar_residual_vs_axis_ratio(
    matched: Table,
    bins: int = 55,
    axis_ratio_bins: int = 10,
    min_bin_count: int = 30,
    limits: tuple[float, float] | None = None,
    cmap: str = "cividis",
) -> tuple[plt.Figure, np.ndarray]:
    """Plot CIGALE-minus-Durbala log-mass residuals against SGA axis ratio."""
    return _plot_residual_vs_axis_ratio(
        matched,
        MASS_ESTIMATORS,
        (
            ("CIGALE_LOGMSTAR_BAYES", "Bayes"),
            ("CIGALE_LOGMSTAR_BEST", "best"),
        ),
        r"$\Delta\log_{10}(M_\star/M_\odot)$",
        bins,
        axis_ratio_bins,
        min_bin_count,
        limits,
        cmap,
    )


def plot_sfr_residual_vs_axis_ratio(
    matched: Table,
    bins: int = 55,
    axis_ratio_bins: int = 10,
    min_bin_count: int = 30,
    limits: tuple[float, float] | None = None,
    cmap: str = "cividis",
) -> tuple[plt.Figure, np.ndarray]:
    """Plot CIGALE-minus-Durbala log-SFR residuals against SGA axis ratio."""
    return _plot_residual_vs_axis_ratio(
        matched,
        SFR_ESTIMATORS,
        (
            ("CIGALE_LOGSFR_BAYES", "Bayes"),
            ("CIGALE_LOGSFR_BEST", "best"),
        ),
        r"$\Delta\log_{10}(\mathrm{SFR})$",
        bins,
        axis_ratio_bins,
        min_bin_count,
        limits,
        cmap,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-match CIGALE and Durbala and create comparison plots."
    )
    parser.add_argument("--cigale", type=Path, default=DEFAULT_CIGALE)
    parser.add_argument("--durbala", type=Path, default=DEFAULT_DURBALA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-sep", type=float, default=30.0)
    parser.add_argument("--max-dv", type=float, default=300.0)
    parser.add_argument("--keep-duplicate-sga", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matched = match_cigale_durbala(
        args.cigale,
        args.durbala,
        max_sep_arcsec=args.max_sep,
        max_velocity_offset_kms=args.max_dv,
        unique_sga=not args.keep_duplicate_sga,
    )
    matched_path = args.output_dir / "cigale_durbala_matches.fits"
    matched.write(matched_path, overwrite=args.overwrite)

    mass_figure, _ = plot_mstar_comparison(matched)
    mass_path = args.output_dir / "cigale_durbala_mstar_comparison.png"
    mass_figure.savefig(mass_path, dpi=180)
    plt.close(mass_figure)

    sfr_figure, _ = plot_sfr_comparison(matched)
    sfr_path = args.output_dir / "cigale_durbala_sfr_comparison.png"
    sfr_figure.savefig(sfr_path, dpi=180)
    plt.close(sfr_figure)

    mass_axis_ratio_figure, _ = plot_mstar_residual_vs_axis_ratio(matched)
    mass_axis_ratio_path = (
        args.output_dir / "cigale_durbala_mstar_residual_vs_ba.png"
    )
    mass_axis_ratio_figure.savefig(mass_axis_ratio_path, dpi=180)
    plt.close(mass_axis_ratio_figure)

    sfr_axis_ratio_figure, _ = plot_sfr_residual_vs_axis_ratio(matched)
    sfr_axis_ratio_path = args.output_dir / "cigale_durbala_sfr_residual_vs_ba.png"
    sfr_axis_ratio_figure.savefig(sfr_axis_ratio_path, dpi=180)
    plt.close(sfr_axis_ratio_figure)

    print(f"Matched table: {matched_path}")
    print(f"Mass figure: {mass_path}")
    print(f"SFR figure: {sfr_path}")
    print(f"Mass residual-axis-ratio figure: {mass_axis_ratio_path}")
    print(f"SFR residual-axis-ratio figure: {sfr_axis_ratio_path}")


if __name__ == "__main__":
    main()
