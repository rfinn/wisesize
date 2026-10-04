#!/usr/bin/env python3
"""Build SGA-2025 WISEsize/parent CIGALE samples and run directories."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import preprocess_sga2025_ap03_for_cigale as prep
from astropy.io import fits
from astropy.table import Table


DEFAULT_SGA_FITS = Path("/Users/rfinn/research/SGA2025/SGA2025-v1.0.fits")
DEFAULT_INTERMEDIATE = Path(
    "/Users/rfinn/research/SGA-CIGALE/intermediate/"
    "sga2025_ap03_cigale_intermediate.fits"
)
DEFAULT_OUTPUT_ROOT = Path("/Users/rfinn/research/SGA-CIGALE")
DEFAULT_TEMPLATE_CONFIG = Path(
    "/Users/rfinn/research/SGA-CIGALE/cigale_runs/"
    "sga2025_ap03_z0025_0040_singlez003_pruned500/pcigale.ini"
)
DEFAULT_Z_MIN = 0.002
DEFAULT_Z_MAX = 0.025
DEFAULT_W3_SNR_MIN = 10.0
DEFAULT_CHUNK_SIZE = 10_000
DEFAULT_CORES = 8
DEFAULT_HIGH_AV = 3.0
FIT_APERTURES = ("AP00", "AP01", "AP02", "AP03", "AP04")
DEFAULT_FIT_APERTURE = "AP04"
SELECTION_MODES = ("wisesize", "complement", "parent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the WISEsize sample, its complement, or its redshift "
            "parent; write redshift-sorted CIGALE inputs; and prepare portable "
            "CIGALE run directories."
        )
    )
    parser.add_argument("--sga-fits", type=Path, default=DEFAULT_SGA_FITS)
    parser.add_argument(
        "--intermediate-fits", type=Path, default=DEFAULT_INTERMEDIATE
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--template-config", type=Path, default=DEFAULT_TEMPLATE_CONFIG
    )
    parser.add_argument("--filter-map", type=Path)
    parser.add_argument("--z-min", type=float, default=DEFAULT_Z_MIN)
    parser.add_argument("--z-max", type=float, default=DEFAULT_Z_MAX)
    parser.add_argument("--w3-snr-min", type=float, default=DEFAULT_W3_SNR_MIN)
    parser.add_argument(
        "--selection-mode",
        choices=SELECTION_MODES,
        default="wisesize",
        help=(
            "Select the W3-qualified WISEsize sample, its exact redshift-parent "
            "complement, or the full redshift parent. Default: wisesize."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--cores", type=int, default=DEFAULT_CORES)
    parser.add_argument(
        "--fit-aperture",
        type=str.upper,
        choices=FIT_APERTURES,
        default=DEFAULT_FIT_APERTURE,
        help=(
            "SGA2025 aperture used for CIGALE photometry. W3 classification "
            "remains based on AP01/AP03 S/N. Default: AP04."
        ),
    )
    parser.add_argument(
        "--mag-error-floor",
        type=float,
        default=prep.DEFAULT_MAG_ERROR_FLOOR,
        help=(
            "Systematic magnitude uncertainty added in quadrature to CIGALE "
            f"flux errors. Default: {prep.DEFAULT_MAG_ERROR_FLOOR:g} mag."
        ),
    )
    parser.add_argument(
        "--sample-stem",
        help=(
            "Output/run name prefix. By default, derive it from the fitting "
            "aperture and magnitude-error floor."
        ),
    )
    parser.add_argument(
        "--model-redshift",
        type=float,
        help=(
            "Single model redshift for every chunk. By default, use the full "
            "sample median rounded to three decimals."
        ),
    )
    parser.add_argument(
        "--high-av",
        type=float,
        default=DEFAULT_HIGH_AV,
        help="High Av_ISM grid point to add if absent. Default: 3.0.",
    )
    parser.add_argument(
        "--frac-agn-grid",
        nargs="+",
        type=float,
        metavar="FRACTION",
        help=(
            "Replace the template skirtor2016 fracAGN grid. Values must be "
            "unique, increasing, and between 0 and 1. By default, retain "
            "the template grid."
        ),
    )
    parser.add_argument(
        "--exclude-bright-star",
        action="store_true",
        help="Exclude SAMPLE bit 16. The default retains these sources.",
    )
    return parser.parse_args()


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    ratio = np.full(numerator.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0.0)
    np.divide(numerator, denominator, out=ratio, where=valid)
    return ratio


def read_selection_columns(
    path: Path,
    fit_aperture: str,
) -> dict[str, np.ndarray]:
    with fits.open(path, memmap=True) as hdul:
        sga = hdul[prep.SGA_EXT].data
        phot = hdul[prep.PHOT_EXT].data
        if len(sga) != len(phot):
            raise AssertionError("SGA2025 and ELLIPSEPHOT row counts differ.")
        sga_ids = np.asarray(sga[prep.ID_COL], dtype=np.int64)
        phot_ids = np.asarray(phot[prep.ID_COL], dtype=np.int64)
        if not np.array_equal(sga_ids, phot_ids):
            raise AssertionError("SGA2025 and ELLIPSEPHOT are not aligned by SGAID.")
        columns = {
            "SGA_ID": sga_ids,
            "redshift": np.asarray(sga[prep.REDSHIFT_COL], dtype=np.float64),
            "sample_bitmask": np.asarray(sga[prep.SAMPLE_COL], dtype=np.int64),
            "ap01_flux_w3": np.asarray(phot["FLUX_AP01_W3"], dtype=np.float64),
            "ap01_err_w3": np.asarray(
                phot["FLUX_ERR_AP01_W3"], dtype=np.float64
            ),
            "ap03_flux_w3": np.asarray(phot["FLUX_AP03_W3"], dtype=np.float64),
            "ap03_err_w3": np.asarray(
                phot["FLUX_ERR_AP03_W3"], dtype=np.float64
            ),
        }
        for band in prep.BANDS:
            lower = band.lower()
            columns[f"fit_flux_{lower}"] = np.asarray(
                phot[f"FLUX_{fit_aperture}_{band}"], dtype=np.float64
            )
            columns[f"fit_err_{lower}"] = np.asarray(
                phot[f"FLUX_ERR_{fit_aperture}_{band}"], dtype=np.float64
            )
        return columns


def read_intermediate(path: Path) -> dict[str, np.ndarray]:
    with fits.open(path, memmap=True) as hdul:
        data = hdul["SGA_AP03_CIGALE_PREP"].data
        return {name: np.array(data[name]) for name in data.columns.names}


def build_selection(
    source: dict[str, np.ndarray],
    z_min: float,
    z_max: float,
    snr_min: float,
    selection_mode: str,
    exclude_bright_star: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    snr_ap01 = safe_ratio(source["ap01_flux_w3"], source["ap01_err_w3"])
    snr_ap03 = safe_ratio(source["ap03_flux_w3"], source["ap03_err_w3"])
    ap01_flag = snr_ap01 > snr_min
    ap03_flag = snr_ap03 > snr_min
    redshift = source["redshift"]
    zflag = np.isfinite(redshift) & (redshift > z_min) & (redshift < z_max)
    bright_star = (
        source["sample_bitmask"].astype(np.int64) & prep.BRIGHT_STAR_BIT_VALUE
    ) != 0
    near_star = (
        source["sample_bitmask"].astype(np.int64) & prep.NEAR_STAR_BIT_VALUE
    ) != 0

    wisesize_flag = ap01_flag | ap03_flag
    if selection_mode == "wisesize":
        selected = zflag & wisesize_flag
    elif selection_mode == "complement":
        selected = zflag & ~wisesize_flag
    elif selection_mode == "parent":
        selected = zflag.copy()
    else:
        raise ValueError(f"Unsupported selection mode: {selection_mode}")
    if exclude_bright_star:
        selected &= ~bright_star

    details = {
        "snr_ap01_w3": snr_ap01,
        "snr_ap03_w3": snr_ap03,
        "ap01_snr_flag": ap01_flag,
        "ap03_snr_flag": ap03_flag,
        "wisesize_flag": wisesize_flag,
        "zflag": zflag,
        "bright_star": bright_star,
        "near_star": near_star,
    }
    return selected, details


def build_cigale_columns(
    source: dict[str, np.ndarray],
    intermediate: dict[str, np.ndarray],
    selected_indices: np.ndarray,
    filter_map: prep.FilterMap,
    mag_error_floor: float,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    dec = np.asarray(intermediate["declination"], dtype=np.float64)[selected_indices]
    bands = prep.decode_fits_string_array(intermediate["bands"])[selected_indices]
    columns: dict[str, np.ndarray] = {
        "id": np.asarray(intermediate["SGA_ID"], dtype=np.int64)[selected_indices],
        "redshift": np.asarray(
            intermediate["redshift_sga_z"], dtype=np.float64
        )[selected_indices],
    }
    supplied_counts: dict[str, int] = {}

    for band in prep.BANDS:
        lower = band.lower()
        transmission = np.asarray(
            intermediate[f"mw_transmission_{lower}"], dtype=np.float64
        )[selected_indices]
        correction = np.full(len(selected_indices), np.nan, dtype=np.float64)
        good_transmission = np.isfinite(transmission) & (transmission > 0.0)
        correction[good_transmission] = (
            prep.NANOMAGGY_TO_MJY / transmission[good_transmission]
        )
        flux = np.asarray(
            source[f"fit_flux_{lower}"], dtype=np.float64
        )[selected_indices] * correction
        error = np.asarray(
            source[f"fit_err_{lower}"], dtype=np.float64
        )[selected_indices] * correction
        error = prep.add_magnitude_error_floor(flux, error, mag_error_floor)
        usable = np.isfinite(flux) & np.isfinite(error) & (error > 0.0)
        if band in prep.OPTICAL_BANDS:
            usable &= np.char.find(bands, prep.OPTICAL_BANDS[band]) >= 0
        if band == "Z":
            usable &= dec < prep.DEC_Z_CUTOFF_DEG

        for rule in filter_map[band]:
            use_rule = usable & prep.filter_rule_mask(rule, dec)
            columns[rule.name] = np.where(use_rule, flux, np.nan)
            columns[f"{rule.name}_err"] = np.where(use_rule, error, np.nan)
            supplied_counts[rule.name] = int(np.count_nonzero(use_rule))

    return columns, supplied_counts


def subset_columns(
    columns: dict[str, np.ndarray], start: int, stop: int
) -> dict[str, np.ndarray]:
    return {name: values[start:stop] for name, values in columns.items()}


def replace_setting(
    lines: list[str], name: str, value: str, section_header: str | None
) -> list[str]:
    current_header: str | None = None
    setting_pattern = re.compile(rf"^(\s*){re.escape(name)}\s*=.*$")
    header_pattern = re.compile(r"^\s*(\[+[^\]]+\]+)\s*$")
    replaced = 0
    output: list[str] = []

    for line in lines:
        header_match = header_pattern.match(line)
        if header_match:
            current_header = header_match.group(1)
        setting_match = setting_pattern.match(line)
        in_target = (
            current_header == section_header
            if section_header is not None
            else current_header is None
        )
        if setting_match and in_target:
            output.append(f"{setting_match.group(1)}{name} = {value}\n")
            replaced += 1
        else:
            output.append(line)

    if replaced != 1:
        location = "top level" if section_header is None else section_header
        raise ValueError(
            f"Expected one {name!r} setting in {location}; found {replaced}."
        )
    return output


def add_grid_value(
    lines: list[str], name: str, value: float, section_header: str
) -> list[str]:
    current_header: str | None = None
    setting_pattern = re.compile(rf"^(\s*){re.escape(name)}\s*=\s*(.*?)\s*$")
    header_pattern = re.compile(r"^\s*(\[+[^\]]+\]+)\s*$")
    replaced = 0
    output: list[str] = []

    for line in lines:
        header_match = header_pattern.match(line)
        if header_match:
            current_header = header_match.group(1)
        setting_match = setting_pattern.match(line)
        if setting_match and current_header == section_header:
            raw_values = [item.strip() for item in setting_match.group(2).split(",")]
            numeric_values = [float(item) for item in raw_values]
            if not any(np.isclose(value, current) for current in numeric_values):
                raw_values.append(f"{value:g}")
            output.append(
                f"{setting_match.group(1)}{name} = {', '.join(raw_values)}\n"
            )
            replaced += 1
        else:
            output.append(line)

    if replaced != 1:
        raise ValueError(
            f"Expected one {name!r} setting in {section_header}; found {replaced}."
        )
    return output


def write_run_config(
    template: Path,
    destination: Path,
    data_file_relative: Path,
    cores: int,
    model_redshift: float,
    high_av: float,
    frac_agn_grid: list[float] | None,
) -> None:
    lines = template.read_text(encoding="utf-8").splitlines(keepends=True)
    lines = replace_setting(lines, "data_file", str(data_file_relative), None)
    lines = replace_setting(lines, "cores", str(cores), None)
    lines = replace_setting(
        lines, "redshift", f"{model_redshift:.3f}", "[[redshifting]]"
    )
    lines = replace_setting(
        lines, "save_best_sed", "False", "[analysis_params]"
    )
    lines = replace_setting(lines, "save_chi2", "none", "[analysis_params]")
    lines = add_grid_value(
        lines, "Av_ISM", high_av, "[[dustatt_modified_CF00]]"
    )
    if frac_agn_grid is not None:
        grid = ", ".join(f"{value:g}" for value in frac_agn_grid)
        lines = replace_setting(
            lines, "fracAGN", grid, "[[skirtor2016]]"
        )
    destination.write_text("".join(lines), encoding="utf-8")


def write_audit_table(
    path: Path,
    source: dict[str, np.ndarray],
    intermediate: dict[str, np.ndarray],
    details: dict[str, np.ndarray],
    selected_indices: np.ndarray,
    chunk_size: int,
    mag_error_floor: float,
    fit_aperture: str,
    selection_mode: str,
) -> None:
    n_selected = len(selected_indices)
    table = Table()
    table["SGA_ID"] = source["SGA_ID"][selected_indices]
    table["source_row"] = selected_indices
    table["redshift"] = source["redshift"][selected_indices]
    table["declination"] = intermediate["declination"][selected_indices]
    table["sample_bitmask"] = source["sample_bitmask"][selected_indices]
    table["near_star"] = details["near_star"][selected_indices]
    table["bright_star"] = details["bright_star"][selected_indices]
    table["snr_ap01_w3"] = details["snr_ap01_w3"][selected_indices]
    table["snr_ap03_w3"] = details["snr_ap03_w3"][selected_indices]
    table["ap01_snr_flag"] = details["ap01_snr_flag"][selected_indices]
    table["ap03_snr_flag"] = details["ap03_snr_flag"][selected_indices]
    table["chunk"] = np.arange(n_selected, dtype=np.int64) // chunk_size + 1
    criteria = {
        "wisesize": (
            "z_min < Z < z_max and (AP01_W3_SNR > threshold or "
            "AP03_W3_SNR > threshold)"
        ),
        "complement": (
            "z_min < Z < z_max and neither AP01_W3_SNR nor AP03_W3_SNR "
            "exceeds threshold"
        ),
        "parent": "z_min < Z < z_max",
    }
    table.meta["SELMODE"] = selection_mode
    table.meta["SELCRIT"] = criteria[selection_mode]
    table.meta["MAGERRFL"] = (mag_error_floor, "CIGALE systematic floor in mag")
    table.meta["FITAP"] = (fit_aperture, "SGA2025 aperture supplied to CIGALE")
    table.write(path, overwrite=True)


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "chunk",
        "n_objects",
        "z_min",
        "z_median",
        "z_max",
        "model_redshift",
        "input_file",
        "run_directory",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    args: argparse.Namespace,
    source: dict[str, np.ndarray],
    details: dict[str, np.ndarray],
    selected: np.ndarray,
    selected_indices: np.ndarray,
    model_redshift: float,
    supplied_counts: dict[str, int],
    manifest_rows: list[dict[str, object]],
) -> None:
    zflag = details["zflag"]
    ap01_flag = details["ap01_snr_flag"]
    ap03_flag = details["ap03_snr_flag"]
    wisesize_flag = details["wisesize_flag"]
    bright_star = details["bright_star"]
    near_star = details["near_star"]
    redshift = source["redshift"][selected_indices]
    selection_rule = {
        "wisesize": (
            f"`SNR_AP01 > {args.w3_snr_min:g}` or "
            f"`SNR_AP03 > {args.w3_snr_min:g}`"
        ),
        "complement": (
            f"neither `SNR_AP01` nor `SNR_AP03` exceeds {args.w3_snr_min:g}"
        ),
        "parent": "redshift selection only",
    }[args.selection_mode]

    lines = [
        "# SGA-2025 CIGALE Sample",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Source SGA file: `{args.sga_fits}`",
        f"Preprocessing intermediate file: `{args.intermediate_fits}`",
        f"CIGALE fitting photometry: `SGA2025 {args.fit_aperture}`",
        "",
        "## Selection",
        "",
        f"- Selection mode: `{args.selection_mode}`",
        f"- Strict redshift range: `{args.z_min} < Z < {args.z_max}`",
        f"- Selection rule: {selection_rule}",
        (
            "- Bright-star policy: "
            + ("excluded" if args.exclude_bright_star else "retained")
        ),
        f"- Redshift-qualified sources: {int(np.count_nonzero(zflag))}",
        f"- AP01-qualified within redshift range: {int(np.count_nonzero(zflag & ap01_flag))}",
        f"- AP03-qualified within redshift range: {int(np.count_nonzero(zflag & ap03_flag))}",
        f"- Qualified by both apertures: {int(np.count_nonzero(zflag & ap01_flag & ap03_flag))}",
        f"- WISEsize-qualified within redshift range: {int(np.count_nonzero(zflag & wisesize_flag))}",
        f"- W3-S/N complement within redshift range: {int(np.count_nonzero(zflag & ~wisesize_flag))}",
        f"- Final selected sample: {int(np.count_nonzero(selected))}",
        f"- Selected INSTAR sources: {int(np.count_nonzero(selected & bright_star))}",
        f"- Selected NEARSTAR sources: {int(np.count_nonzero(selected & near_star))}",
        "",
        "## Redshift",
        "",
        f"- Minimum: {np.min(redshift):.8f}",
        f"- Median: {np.median(redshift):.8f}",
        f"- Mean: {np.mean(redshift):.8f}",
        f"- Maximum: {np.max(redshift):.8f}",
        f"- Fixed CIGALE model redshift for every chunk: {model_redshift:.3f}",
        "- Rows are sorted by increasing observed redshift.",
        "",
        "## Chunks",
        "",
        "| Chunk | N | z min | z median | z max | Model z |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in manifest_rows:
        lines.append(
            f"| {row['chunk']} | {row['n_objects']} | {row['z_min']} | "
            f"{row['z_median']} | {row['z_max']} | {row['model_redshift']} |"
        )

    lines.extend(["", f"## Supplied {args.fit_aperture} Photometry", ""])
    for name, count in supplied_counts.items():
        lines.append(f"- `{name}`: {count}")

    lines.extend(
        [
            "",
            "## CIGALE Configuration",
            "",
            f"- Cores per run: {args.cores}",
            (
                f"- Photometric error floor: `{args.mag_error_floor:g} mag`, "
                "added in quadrature to every supplied CIGALE flux uncertainty."
            ),
            (
                "- W3 sample selection uses the original catalog uncertainties; "
                "the systematic floor changes fitting errors, not membership."
            ),
            f"- Added high attenuation point: `Av_ISM={args.high_av:g}`",
            (
                "- AGN fraction grid: `"
                + (
                    ", ".join(f"{value:g}" for value in args.frac_agn_grid)
                    if args.frac_agn_grid is not None
                    else "retained from template"
                )
                + "`"
            ),
            "- `tau_main=1e5` is retained from the template grid.",
            "- Best-fit SED files are disabled.",
            "- Raw chi-square files are disabled.",
            f"- Each input uses SGA-2025 {args.fit_aperture} "
            "extinction-corrected fluxes in mJy.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.sga_fits = args.sga_fits.expanduser().resolve()
    args.intermediate_fits = args.intermediate_fits.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.template_config = args.template_config.expanduser().resolve()

    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive.")
    if args.cores < 1:
        raise ValueError("--cores must be positive.")
    if args.mag_error_floor < 0.0:
        raise ValueError("--mag-error-floor must be non-negative.")
    if args.frac_agn_grid is not None:
        grid = np.asarray(args.frac_agn_grid, dtype=float)
        if not np.all(np.isfinite(grid)) or np.any((grid < 0.0) | (grid > 1.0)):
            raise ValueError("--frac-agn-grid values must be finite and between 0 and 1.")
        if len(np.unique(grid)) != len(grid) or np.any(np.diff(grid) <= 0.0):
            raise ValueError("--frac-agn-grid values must be unique and increasing.")
    if args.sample_stem is None:
        floor_label = f"{args.mag_error_floor:.2f}".replace(".", "p")
        if args.selection_mode == "wisesize":
            args.sample_stem = (
                f"wisesize_sga2025_{args.fit_aperture.lower()}_"
                f"z0002_0025_w3snr10_errfloor{floor_label}mag"
            )
        else:
            selection_label = {
                "complement": "w3snr10_complement",
                "parent": "parent",
            }[args.selection_mode]
            args.sample_stem = (
                f"sga2025_{args.fit_aperture.lower()}_z0002_0025_"
                f"{selection_label}_errfloor{floor_label}mag"
            )
    if not args.sample_stem or Path(args.sample_stem).name != args.sample_stem:
        raise ValueError("--sample-stem must be a non-empty filename component.")
    if not args.z_min < args.z_max:
        raise ValueError("--z-min must be less than --z-max.")
    for path in (args.sga_fits, args.intermediate_fits, args.template_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    template_spec = args.template_config.with_suffix(".ini.spec")
    if not template_spec.is_file():
        raise FileNotFoundError(template_spec)

    source = read_selection_columns(args.sga_fits, args.fit_aperture)
    intermediate = read_intermediate(args.intermediate_fits)
    if not np.array_equal(source["SGA_ID"], intermediate["SGA_ID"]):
        raise AssertionError("SGA source and preprocessing rows are not aligned.")

    selected, details = build_selection(
        source,
        args.z_min,
        args.z_max,
        args.w3_snr_min,
        args.selection_mode,
        args.exclude_bright_star,
    )
    selected_indices = np.flatnonzero(selected)
    selected_indices = selected_indices[
        np.argsort(source["redshift"][selected_indices], kind="stable")
    ]
    if not len(selected_indices):
        raise RuntimeError(f"The {args.selection_mode} selection is empty.")

    if args.model_redshift is None:
        model_redshift = float(
            np.round(np.median(source["redshift"][selected_indices]), 3)
        )
    else:
        model_redshift = float(args.model_redshift)

    filter_map = prep.load_filter_map(args.filter_map)
    cigale_columns, supplied_counts = build_cigale_columns(
        source,
        intermediate,
        selected_indices,
        filter_map,
        args.mag_error_floor,
    )

    input_dir = args.output_root / "inputs"
    intermediate_dir = args.output_root / "intermediate"
    report_dir = args.output_root / "reports"
    run_root = args.output_root / "cigale_runs"
    for directory in (input_dir, intermediate_dir, report_dir, run_root):
        directory.mkdir(parents=True, exist_ok=True)

    full_input = input_dir / f"{args.sample_stem}.dat"
    audit_path = intermediate_dir / f"{args.sample_stem}_selection.fits"
    manifest_path = report_dir / f"{args.sample_stem}_chunks.csv"
    report_path = report_dir / f"{args.sample_stem}_report.md"
    prep.write_cigale_ascii(full_input, cigale_columns)
    write_audit_table(
        audit_path,
        source,
        intermediate,
        details,
        selected_indices,
        args.chunk_size,
        args.mag_error_floor,
        args.fit_aperture,
        args.selection_mode,
    )

    manifest_rows: list[dict[str, object]] = []
    n_selected = len(selected_indices)
    for chunk_index, start in enumerate(range(0, n_selected, args.chunk_size), start=1):
        stop = min(start + args.chunk_size, n_selected)
        chunk_columns = subset_columns(cigale_columns, start, stop)
        chunk_name = f"{args.sample_stem}_chunk{chunk_index:02d}"
        chunk_input = input_dir / f"{chunk_name}.dat"
        run_dir = run_root / chunk_name
        run_dir.mkdir(parents=True, exist_ok=True)
        if (run_dir / "out").exists():
            raise FileExistsError(
                f"Refusing to overwrite existing CIGALE output: {run_dir / 'out'}"
            )

        prep.write_cigale_ascii(chunk_input, chunk_columns)
        relative_input = Path("../../inputs") / chunk_input.name
        write_run_config(
            args.template_config,
            run_dir / "pcigale.ini",
            relative_input,
            args.cores,
            model_redshift,
            args.high_av,
            args.frac_agn_grid,
        )
        shutil.copy2(template_spec, run_dir / "pcigale.ini.spec")

        chunk_z = np.asarray(chunk_columns["redshift"], dtype=np.float64)
        manifest_rows.append(
            {
                "chunk": chunk_index,
                "n_objects": stop - start,
                "z_min": f"{np.min(chunk_z):.8f}",
                "z_median": f"{np.median(chunk_z):.8f}",
                "z_max": f"{np.max(chunk_z):.8f}",
                "model_redshift": f"{model_redshift:.3f}",
                "input_file": str(chunk_input),
                "run_directory": str(run_dir),
            }
        )

    write_manifest(manifest_path, manifest_rows)
    write_report(
        report_path,
        args,
        source,
        details,
        selected,
        selected_indices,
        model_redshift,
        supplied_counts,
        manifest_rows,
    )

    print(f"Selected objects: {n_selected}")
    print(f"Fixed model redshift: {model_redshift:.3f}")
    print(f"Full input: {full_input}")
    print(f"Audit table: {audit_path}")
    print(f"Chunk manifest: {manifest_path}")
    print(f"Report: {report_path}")
    for row in manifest_rows:
        print(
            f"Chunk {row['chunk']}: N={row['n_objects']}, "
            f"z={row['z_min']}..{row['z_max']}, run={row['run_directory']}"
        )


if __name__ == "__main__":
    main()
