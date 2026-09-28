#!/usr/bin/env python
"""Plot CIGALE input photometry against the best-fit model photometry.

This produces one PNG/PDF per object plus an index CSV. If CIGALE was run with
``analysis_params.save_best_sed = True``, the continuous ``*_best_model.fits``
SED is overplotted as well.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", "/Users/rfinn/research/SGA-CIGALE/cache")
os.environ.setdefault(
    "MPLCONFIGDIR", "/Users/rfinn/research/SGA-CIGALE/cache/matplotlib"
)

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np

# See the note in preprocess_sga2025_ap03_for_cigale.py. This keeps Astropy
# import-compatible in the current cigale environment.
if not hasattr(np, "in1d"):
    np.in1d = np.isin  # type: ignore[attr-defined]
try:
    import numpy.lib._function_base_impl as _np_function_base_impl

    if not hasattr(_np_function_base_impl, "_check_interpolation_as_method"):

        def _check_interpolation_as_method(method, interpolation, fname):
            return interpolation if interpolation is not None else method

        _np_function_base_impl._check_interpolation_as_method = (  # type: ignore[attr-defined]
            _check_interpolation_as_method
        )
except Exception:
    pass

from astropy.io import fits
from pcigale.data import SimpleDatabase as Database


DEFAULT_RUN_DIR = Path(
    "/Users/rfinn/research/SGA-CIGALE/cigale_runs/smoke_test_speclite_legacy"
)
DEFAULT_OUTPUT_DIR = Path("/Users/rfinn/research/SGA-CIGALE/plots/cigale_best_models")
SKIP_DATA_COLUMNS = {"id", "redshift", "distance"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CIGALE input photometry and best-fit model fluxes."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"CIGALE run directory containing out/results.fits. Default: {DEFAULT_RUN_DIR}",
    )
    parser.add_argument(
        "--input-photometry",
        type=Path,
        default=None,
        help=(
            "Optional original CIGALE photometry table. If omitted, the script "
            "uses data_file from pcigale.ini, then falls back to out/observations.fits."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for plots and index CSV. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="Specific object IDs to plot. Default: first --max-plots rows in results.fits.",
    )
    parser.add_argument(
        "--max-plots",
        type=int,
        default=20,
        help="Maximum number of plots when --ids is not supplied. Default: 20",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf"),
        default="png",
        help="Output plot format. Default: png",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="PNG resolution. Default: 160",
    )
    parser.add_argument(
        "--no-continuous-sed",
        action="store_true",
        help="Do not overlay *_best_model.fits, even when present.",
    )
    return parser.parse_args()


def format_id(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and np.isfinite(value):
        rounded = round(float(value))
        if np.isclose(float(value), rounded):
            return str(int(rounded))
    return str(value).strip()


def safe_stem(text: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in text)


def read_table(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("#"):
                names = stripped[1:].strip().split()
                if names:
                    break
        else:
            raise ValueError(f"No commented header line found in {path}.")

    data = np.genfromtxt(path, comments="#", dtype=str, encoding="utf-8")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != len(names):
        raise ValueError(
            f"{path} has {data.shape[1]} data columns but {len(names)} header names."
        )
    return names, data


def read_fits_table(path: Path) -> tuple[list[str], np.recarray]:
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        names = list(data.names)
        copy = np.array(data)
    return names, copy


def parse_pcigale_data_file(run_dir: Path) -> Path | None:
    for config_path in (run_dir / "pcigale.ini", run_dir / "out" / "pcigale.ini"):
        if not config_path.exists():
            continue
        with config_path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped.startswith("data_file ="):
                    value = stripped.split("=", 1)[1].strip()
                    return Path(value) if value else None
    return None


def choose_input_photometry(args: argparse.Namespace) -> tuple[list[str], object, Path]:
    if args.input_photometry is not None:
        names, data = read_table(args.input_photometry)
        return names, data, args.input_photometry

    data_file = parse_pcigale_data_file(args.run_dir)
    if data_file is not None and data_file.exists():
        names, data = read_table(data_file)
        return names, data, data_file

    fallback = args.run_dir / "out" / "observations.fits"
    names, data = read_fits_table(fallback)
    return names, data, fallback


def band_columns(names: list[str]) -> list[str]:
    return [
        name
        for name in names
        if name not in SKIP_DATA_COLUMNS
        and not name.endswith("_err")
        and f"{name}_err" in names
    ]


def index_rows_by_id(names: list[str], data: object) -> dict[str, int]:
    id_index = names.index("id")
    if isinstance(data, np.ndarray) and data.dtype.names is None:
        ids = data[:, id_index]
    else:
        ids = data["id"]  # type: ignore[index]
    return {format_id(value): idx for idx, value in enumerate(ids)}


def scalar_from_row(names: list[str], data: object, row_index: int, name: str) -> float:
    if name not in names:
        return np.nan
    if isinstance(data, np.ndarray) and data.dtype.names is None:
        return float(data[row_index, names.index(name)])
    return float(data[name][row_index])  # type: ignore[index]


def filter_pivots_micron(bands: list[str]) -> dict[str, float]:
    pivots: dict[str, float] = {}
    with Database("filters") as database:
        for band in bands:
            filt = database.get(name=band)
            pivots[band] = float(filt.pivot) / 1000.0
    return pivots


def best_model_path(run_dir: Path, object_id: str) -> Path:
    return run_dir / "out" / f"{object_id}_best_model.fits"


def read_best_sed(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        wavelength_um = np.asarray(data["wavelength"], dtype=np.float64) / 1000.0
        fnu_mjy = np.asarray(data["Fnu"], dtype=np.float64)
    ok = np.isfinite(wavelength_um) & np.isfinite(fnu_mjy) & (fnu_mjy > 0.0)
    return wavelength_um[ok], fnu_mjy[ok]


def finite_positive(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values > 0.0)


def log_value(value: float) -> str:
    if np.isfinite(value) and value > 0.0:
        return f"{np.log10(value):.2f}"
    return "nan"


def plot_one(
    object_id: str,
    results_names: list[str],
    results_data: object,
    results_index: int,
    phot_names: list[str],
    phot_data: object,
    phot_index: int,
    pivots: dict[str, float],
    bands: list[str],
    run_dir: Path,
    output_dir: Path,
    fmt: str,
    dpi: int,
    continuous_sed: bool,
) -> dict[str, object]:
    records: list[dict[str, float | str]] = []
    for band in bands:
        best_name = f"best.{band}"
        if best_name not in results_names:
            continue
        records.append(
            {
                "band": band,
                "wave_um": pivots[band],
                "flux_mjy": scalar_from_row(phot_names, phot_data, phot_index, band),
                "err_mjy": scalar_from_row(
                    phot_names, phot_data, phot_index, f"{band}_err"
                ),
                "model_mjy": scalar_from_row(
                    results_names, results_data, results_index, best_name
                ),
            }
        )
    records.sort(key=lambda item: float(item["wave_um"]))

    wave = np.array([float(item["wave_um"]) for item in records])
    flux = np.array([float(item["flux_mjy"]) for item in records])
    err = np.array([float(item["err_mjy"]) for item in records])
    model = np.array([float(item["model_mjy"]) for item in records])
    labels = [str(item["band"]) for item in records]

    measured = finite_positive(flux) & np.isfinite(err) & (err > 0.0)
    modeled = measured & finite_positive(model)
    upper = finite_positive(flux) & np.isfinite(err) & (err < 0.0)

    sed_path = best_model_path(run_dir, object_id)
    sed_wave = np.array([], dtype=np.float64)
    sed_fnu = np.array([], dtype=np.float64)
    if continuous_sed and sed_path.exists():
        sed_wave, sed_fnu = read_best_sed(sed_path)

    fig = plt.figure(figsize=(8.5, 6.5))
    grid = fig.add_gridspec(2, 1, height_ratios=(3.2, 1.1), hspace=0.08)
    ax = fig.add_subplot(grid[0])
    residual_ax = fig.add_subplot(grid[1], sharex=ax)

    if sed_wave.size:
        ax.plot(sed_wave, sed_fnu, color="0.25", lw=1.3, label="best model SED")

    if np.any(measured):
        ax.errorbar(
            wave[measured],
            flux[measured],
            yerr=err[measured],
            fmt="o",
            ms=5,
            color="tab:blue",
            ecolor="tab:blue",
            capsize=2,
            label="input photometry",
        )
    if np.any(modeled):
        ax.plot(
            wave[modeled],
            model[modeled],
            "s--",
            ms=4,
            color="tab:orange",
            label="best model photometry",
        )
    if np.any(upper):
        ax.scatter(
            wave[upper],
            flux[upper],
            marker="v",
            s=35,
            facecolors="none",
            edgecolors="tab:blue",
            label="upper limit",
        )

    for xval, yval, label in zip(wave[measured], flux[measured], np.array(labels)[measured]):
        ax.annotate(
            label.split(".")[-1],
            (xval, yval),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="0.25",
        )

    if np.any(modeled):
        residual = (flux[modeled] - model[modeled]) / flux[modeled]
        residual_err = err[modeled] / flux[modeled]
        residual_ax.axhline(0.0, color="0.35", lw=1.0, ls=":")
        residual_ax.errorbar(
            wave[modeled],
            residual,
            yerr=residual_err,
            fmt="o",
            ms=4,
            color="tab:blue",
            ecolor="tab:blue",
            capsize=2,
        )
        max_abs = float(np.nanpercentile(np.abs(residual), 90) * 1.4)
        max_abs = min(max(max_abs, 0.5), 5.0)
        residual_ax.set_ylim(-max_abs, max_abs)
    else:
        residual_ax.text(
            0.5,
            0.5,
            "No positive observed/model flux pairs",
            transform=residual_ax.transAxes,
            ha="center",
            va="center",
        )

    positive_y = list(flux[measured]) + list(model[modeled])
    if not positive_y:
        positive_y = list(sed_fnu)
    positive_y = [value for value in positive_y if np.isfinite(value) and value > 0.0]
    if positive_y:
        ymin = min(positive_y) / 3.0
        ymax = max(positive_y) * 3.0
        ax.set_ylim(ymin, ymax)
        ax.set_yscale("log")

    positive_x = wave[np.isfinite(wave) & (wave > 0.0)]
    if positive_x.size:
        ax.set_xlim(positive_x.min() / 1.4, positive_x.max() * 1.4)
    ax.set_xscale("log")
    residual_ax.set_xscale("log")
    ax.tick_params(axis="x", labelbottom=False)

    redshift = scalar_from_row(results_names, results_data, results_index, "best.universe.redshift")
    chi2 = scalar_from_row(results_names, results_data, results_index, "best.reduced_chi_square")
    mstar = scalar_from_row(results_names, results_data, results_index, "best.stellar.m_star")
    sfr = scalar_from_row(results_names, results_data, results_index, "best.sfh.sfr")
    ax.set_title(
        (
            f"SGA {object_id}  z={redshift:.5g}  chi2_red={chi2:.3g}  "
            f"logMstar={log_value(mstar)}  logSFR={log_value(sfr)}"
        ),
        fontsize=11,
    )
    ax.set_ylabel("Fnu (mJy)")
    residual_ax.set_ylabel("(obs-model)/obs")
    residual_ax.set_xlabel("Observed wavelength (um)")
    ax.legend(loc="best", fontsize=8)
    fig.align_ylabels([ax, residual_ax])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"SGA{safe_stem(object_id)}_cigale_best_model.{fmt}"
    fig.savefig(output_path, dpi=dpi if fmt == "png" else None, bbox_inches="tight")
    plt.close(fig)

    return {
        "id": object_id,
        "plot_path": str(output_path),
        "redshift": redshift,
        "chi2_red": chi2,
        "log_mstar": log_value(mstar),
        "log_sfr": log_value(sfr),
        "n_observed": int(np.count_nonzero(measured)),
        "n_modeled": int(np.count_nonzero(modeled)),
        "continuous_sed": bool(sed_wave.size),
        "best_sed_path": str(sed_path) if sed_path.exists() else "",
    }


def write_index(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "id",
        "plot_path",
        "redshift",
        "chi2_red",
        "log_mstar",
        "log_sfr",
        "n_observed",
        "n_modeled",
        "continuous_sed",
        "best_sed_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    results_path = args.run_dir / "out" / "results.fits"
    results_names, results_data = read_fits_table(results_path)
    phot_names, phot_data, phot_path = choose_input_photometry(args)

    result_rows = index_rows_by_id(results_names, results_data)
    phot_rows = index_rows_by_id(phot_names, phot_data)

    if args.ids:
        object_ids = [format_id(value) for value in args.ids]
    else:
        all_ids = list(result_rows)
        object_ids = all_ids[: args.max_plots]

    missing = [obj_id for obj_id in object_ids if obj_id not in result_rows]
    if missing:
        raise KeyError(f"IDs not found in results.fits: {missing}")
    missing = [obj_id for obj_id in object_ids if obj_id not in phot_rows]
    if missing:
        raise KeyError(f"IDs not found in input photometry table {phot_path}: {missing}")

    bands = band_columns(phot_names)
    pivots = filter_pivots_micron(bands)
    rows = [
        plot_one(
            object_id,
            results_names,
            results_data,
            result_rows[object_id],
            phot_names,
            phot_data,
            phot_rows[object_id],
            pivots,
            bands,
            args.run_dir,
            args.output_dir,
            args.format,
            args.dpi,
            continuous_sed=not args.no_continuous_sed,
        )
        for object_id in object_ids
    ]

    index_path = args.output_dir / "cigale_best_model_plots_index.csv"
    write_index(index_path, rows)
    print(f"Input photometry: {phot_path}")
    print(f"Results: {results_path}")
    print(f"Wrote {len(rows)} plots to {args.output_dir}")
    print(f"Wrote {index_path}")


if __name__ == "__main__":
    main()
