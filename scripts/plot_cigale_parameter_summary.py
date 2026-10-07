#!/usr/bin/env python3
"""Create collaborator-facing summary plots for the SGA-CIGALE catalog.

Notebook example
----------------
>>> from astropy.table import Table
>>> from scripts.plot_cigale_parameter_summary import (
...     prepare_bayes_parameters,
...     plot_model_parameter_summary,
...     plot_physical_summary,
... )
>>> catalog = Table.read("path/to/cigale_results.fits")
>>> parameters = prepare_bayes_parameters(catalog)
>>> fig_physical, axes_physical = plot_physical_summary(parameters)
>>> fig_model, axes_model = plot_model_parameter_summary(parameters)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr


DEFAULT_CIGALE = Path(
    "/Users/rfinn/research/SGA-CIGALE/results/"
    "sga2025_ap04_zlt0040_parent_errfloor0p10mag_"
    "agnfrac0to0p5_results.fits"
)
DEFAULT_OUTPUT_DIR = Path(
    "/Users/rfinn/research/SGA-CIGALE/summary_plots"
)
M_SUN_KG = 1.98847e30
Z_SUN = 0.02

REQUIRED_COLUMNS = (
    "bayes.agn.fracAGN",
    "bayes.attenuation.Av_ISM",
    "bayes.dust.mass",
    "bayes.sfh.age",
    "bayes.sfh.burst_age",
    "bayes.sfh.f_burst",
    "bayes.sfh.sfr",
    "bayes.sfh.tau_burst",
    "bayes.sfh.tau_main",
    "bayes.stellar.m_star",
    "bayes.stellar.metallicity",
)


@dataclass(frozen=True)
class Relation:
    x: str
    y: str
    xlabel: str
    ylabel: str
    title: str


PHYSICAL_RELATIONS = (
    Relation(
        "log_mstar",
        "log_sfr",
        r"$\log_{10}(M_\star/M_\odot)$",
        r"$\log_{10}(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})$",
        "Star-forming sequence",
    ),
    Relation(
        "log_mstar",
        "log_ssfr",
        r"$\log_{10}(M_\star/M_\odot)$",
        r"$\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})$",
        "Specific SFR",
    ),
    Relation(
        "log_mstar",
        "log_z_solar",
        r"$\log_{10}(M_\star/M_\odot)$",
        r"$\log_{10}(Z_\star/Z_\odot)$",
        "Stellar metallicity",
    ),
    Relation(
        "log_mstar",
        "log_mdust",
        r"$\log_{10}(M_\star/M_\odot)$",
        r"$\log_{10}(M_\mathrm{dust}/M_\odot)$",
        "Dust mass",
    ),
    Relation(
        "log_mdust",
        "av_ism",
        r"$\log_{10}(M_\mathrm{dust}/M_\odot)$",
        r"$A_V^\mathrm{ISM}$ (mag)",
        "Attenuation and dust mass",
    ),
    Relation(
        "log_mstar",
        "av_ism",
        r"$\log_{10}(M_\star/M_\odot)$",
        r"$A_V^\mathrm{ISM}$ (mag)",
        "Attenuation and stellar mass",
    ),
)

MODEL_RELATIONS = (
    Relation(
        "log_sfr",
        "age_gyr",
        r"$\log_{10}(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})$",
        r"$\mathrm{age}$ (Gyr)",
        "Main population age",
    ),
    Relation(
        "log_sfr",
        "log_tau_main_gyr",
        r"$\log_{10}(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})$",
        r"$\log_{10}(\tau_\mathrm{main}/\mathrm{Gyr})$",
        "Main SFH timescale",
    ),
    Relation(
        "log_sfr",
        "f_burst",
        r"$\log_{10}(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})$",
        r"$f_\mathrm{burst}$",
        "Burst mass fraction",
    ),
    Relation(
        "log_sfr",
        "burst_age_gyr",
        r"$\log_{10}(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})$",
        r"$\mathrm{burst\ age}$ (Gyr)",
        "Burst age",
    ),
    Relation(
        "log_sfr",
        "log_tau_burst_gyr",
        r"$\log_{10}(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})$",
        r"$\log_{10}(\tau_\mathrm{burst}/\mathrm{Gyr})$",
        "Burst timescale",
    ),
    Relation(
        "log_sfr",
        "frac_agn",
        r"$\log_{10}(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})$",
        r"$\mathrm{fracAGN}$",
        "AGN fraction",
    ),
)


def _positive_log10(values: np.ndarray, scale: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=float) / scale
    output = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values) & (values > 0.0)
    output[valid] = np.log10(values[valid])
    return output


def prepare_bayes_parameters(catalog: Table) -> dict[str, np.ndarray]:
    """Return plotting quantities derived from CIGALE Bayes estimates."""
    missing = [column for column in REQUIRED_COLUMNS if column not in catalog.colnames]
    if missing:
        raise KeyError(f"Missing required CIGALE columns: {', '.join(missing)}")

    log_mstar = _positive_log10(catalog["bayes.stellar.m_star"])
    log_sfr = _positive_log10(catalog["bayes.sfh.sfr"])
    return {
        "log_mstar": log_mstar,
        "log_sfr": log_sfr,
        "log_ssfr": log_sfr - log_mstar,
        "log_z_solar": _positive_log10(
            catalog["bayes.stellar.metallicity"], scale=Z_SUN
        ),
        "log_mdust": _positive_log10(
            catalog["bayes.dust.mass"], scale=M_SUN_KG
        ),
        "av_ism": np.asarray(catalog["bayes.attenuation.Av_ISM"], float),
        "age_gyr": np.asarray(catalog["bayes.sfh.age"], float) / 1000.0,
        "burst_age_gyr": np.asarray(catalog["bayes.sfh.burst_age"], float)
        / 1000.0,
        "f_burst": np.asarray(catalog["bayes.sfh.f_burst"], float),
        "log_tau_main_gyr": _positive_log10(
            catalog["bayes.sfh.tau_main"], scale=1000.0
        ),
        "log_tau_burst_gyr": _positive_log10(
            catalog["bayes.sfh.tau_burst"], scale=1000.0
        ),
        "frac_agn": np.asarray(catalog["bayes.agn.fracAGN"], float),
    }


def _percentile_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise ValueError("Cannot determine plot limits from an empty array.")
    low, high = np.percentile(finite, [0.5, 99.5])
    padding = max((high - low) * 0.04, 1e-6)
    return float(low - padding), float(high + padding)


def _binned_median(
    x: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
    min_count: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    medians = np.full(len(centers), np.nan)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (x >= low) & (x < high)
        if index == len(centers) - 1:
            selected |= x == high
        if np.count_nonzero(selected) >= min_count:
            medians[index] = np.median(y[selected])
    return centers, medians


def _plot_relations(
    parameters: dict[str, np.ndarray],
    relations: tuple[Relation, ...],
    bins: int,
    trend_bins: int,
    cmap: str,
    show_trend: bool,
) -> tuple[plt.Figure, np.ndarray]:
    if bins < 10:
        raise ValueError("bins must be >=10.")
    if show_trend and trend_bins < 4:
        raise ValueError("trend_bins must be >=4 when show_trend is true.")

    panel_data = []
    max_count = 1.0
    for relation in relations:
        x = np.asarray(parameters[relation.x], float)
        y = np.asarray(parameters[relation.y], float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        x_limits = _percentile_limits(x)
        y_limits = _percentile_limits(y)
        x_edges = np.linspace(*x_limits, bins + 1)
        y_edges = np.linspace(*y_limits, bins + 1)
        histogram, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
        max_count = max(max_count, float(histogram.max()))
        panel_data.append((relation, x, y, x_edges, y_edges, histogram))

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.8, 8.5),
        constrained_layout=True,
        squeeze=False,
    )
    norm = LogNorm(vmin=1.0, vmax=max_count)
    mesh = None
    for ax, panel in zip(axes.flat, panel_data):
        relation, x, y, x_edges, y_edges, histogram = panel
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            np.ma.masked_equal(histogram.T, 0.0),
            cmap=cmap,
            norm=norm,
            shading="auto",
        )
        if show_trend:
            trend_edges = np.linspace(x_edges[0], x_edges[-1], trend_bins + 1)
            centers, medians = _binned_median(x, y, trend_edges)
            good = np.isfinite(medians)
            ax.plot(
                centers[good],
                medians[good],
                color="white",
                marker="o",
                markersize=3.0,
                linewidth=2.0,
            )
        rho, _ = spearmanr(x, y)
        ax.text(
            0.04,
            0.96,
            f"N = {len(x):,}\n$\\rho_s$ = {rho:+.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
        ax.set_title(relation.title)
        ax.set_xlabel(relation.xlabel)
        ax.set_ylabel(relation.ylabel)
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])
        ax.grid(color="0.9", linewidth=0.6)
    if mesh is not None:
        fig.colorbar(mesh, ax=axes.ravel().tolist(), label="Galaxies per bin")
    return fig, axes


def plot_physical_summary(
    parameters: dict[str, np.ndarray],
    bins: int = 60,
    trend_bins: int = 12,
    cmap: str = "viridis",
    show_trend: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot six physical-property relations from the Bayes estimates."""
    return _plot_relations(
        parameters, PHYSICAL_RELATIONS, bins, trend_bins, cmap, show_trend
    )


def plot_model_parameter_summary(
    parameters: dict[str, np.ndarray],
    bins: int = 60,
    trend_bins: int = 12,
    cmap: str = "magma",
    show_trend: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot Bayes SFH and AGN parameters against log SFR."""
    return _plot_relations(
        parameters, MODEL_RELATIONS, bins, trend_bins, cmap, show_trend
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create multipanel summary plots for the SGA-CIGALE catalog."
    )
    parser.add_argument("--cigale", type=Path, default=DEFAULT_CIGALE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    physical_path = args.output_dir / "cigale_bayes_physical_summary.png"
    model_path = args.output_dir / "cigale_bayes_model_parameter_summary.png"
    for path in (physical_path, model_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    catalog = Table.read(args.cigale)
    parameters = prepare_bayes_parameters(catalog)

    physical_figure, _ = plot_physical_summary(parameters)
    physical_figure.savefig(physical_path, dpi=200)
    plt.close(physical_figure)

    model_figure, _ = plot_model_parameter_summary(parameters)
    model_figure.savefig(model_path, dpi=200)
    plt.close(model_figure)

    print(f"Catalog rows: {len(catalog):,}")
    print(f"Physical summary: {physical_path}")
    print(f"Model-parameter summary: {model_path}")


if __name__ == "__main__":
    main()
