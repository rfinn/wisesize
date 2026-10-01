#!/usr/bin/env python
"""Prepare SGA-2025 AP03 photometry for a future CIGALE run.

This script preserves the input FITS catalog unchanged. It writes:

* an intermediate FITS table with one row per SGA source and an audit trail;
* a CIGALE-facing whitespace table with the first-pass selected sample;
* a Markdown diagnostic report; and
* a CSV review table of the largest star-flagged galaxies.

No S/N, missing-band, or photometric-quality cuts are applied. Per-band
availability is only used to decide whether a measurement can be supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# The current virgo environment pairs Astropy 7.0.0 with NumPy 2.4.x.
# Astropy still imports np.in1d and a private quantile helper removed from
# this NumPy build. These shims are import-compatibility only; this script
# does not call Astropy's masked-table quantile helpers.
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


DEFAULT_SGA_FITS = Path("/Users/rfinn/research/SGA2025/SGA2025-v1.0.fits")
DEFAULT_OUTPUT_DIR = Path("/Users/rfinn/research/SGA-CIGALE")
DEFAULT_MAG_ERROR_FLOOR = 0.1

DEC_Z_CUTOFF_DEG = 32.375
NANOMAGGY_TO_MJY = 3631.0e-6

SGA_EXT = "SGA2025"
PHOT_EXT = "ELLIPSEPHOT"
ID_COL = "SGAID"
DEC_COL = "DEC"
REDSHIFT_COL = "Z"
SAMPLE_COL = "SAMPLE"
BANDS_COL = "BANDS"

NEAR_STAR_FLAG_NAME = "NEARSTAR"
NEAR_STAR_BIT_VALUE = 8
BRIGHT_STAR_FLAG_NAME = "INSTAR"
BRIGHT_STAR_BIT_VALUE = 16

BANDS = ("FUV", "NUV", "G", "R", "I", "Z", "W1", "W2", "W3", "W4")
OPTICAL_BANDS = {"G": "g", "R": "r", "I": "i", "Z": "z"}
DEFAULT_FILTER_MAP = {
    "FUV": "galex.FUV",
    "NUV": "galex.NUV",
    "G": [
        {"filter": "decamDR1-g", "dec_lt": DEC_Z_CUTOFF_DEG},
        {"filter": "BASS-g", "dec_ge": DEC_Z_CUTOFF_DEG},
    ],
    "R": [
        {"filter": "decamDR1-r", "dec_lt": DEC_Z_CUTOFF_DEG},
        {"filter": "BASS-r", "dec_ge": DEC_Z_CUTOFF_DEG},
    ],
    "I": [{"filter": "decamDR1-i", "dec_lt": DEC_Z_CUTOFF_DEG}],
    "Z": [{"filter": "decamDR1-z", "dec_lt": DEC_Z_CUTOFF_DEG}],
    "W1": "wise.W1",
    "W2": "wise.W2",
    "W3": "wise.W3",
    "W4": "wise.W4",
}
REVIEW_COLS = (
    "SGANAME",
    "GALAXY",
    "OBJNAME",
    "RA",
    "D26",
    "GROUP_NAME",
    "GROUP_DIAMETER",
)
DEFAULT_LARGE_REVIEW_LIMIT = 200


@dataclass(frozen=True)
class FilterRule:
    name: str
    dec_lt: float | None = None
    dec_ge: float | None = None


FilterMap = dict[str, tuple[FilterRule, ...]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess SGA-2025 AP03 photometry for CIGALE."
    )
    parser.add_argument(
        "--sga-fits",
        type=Path,
        default=DEFAULT_SGA_FITS,
        help=f"Input SGA-2025 FITS catalog. Default: {DEFAULT_SGA_FITS}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Root directory for generated outputs. Subdirectories inputs/, "
            f"intermediate/, and reports/ are created below it. Default: {DEFAULT_OUTPUT_DIR}"
        ),
    )
    parser.add_argument(
        "--filter-map",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping from SGA band labels to CIGALE filter names or "
            "rule lists. Defaults to installed GALEX, Legacy Survey, and WISE filters."
        ),
    )
    parser.add_argument(
        "--large-review-limit",
        type=int,
        default=DEFAULT_LARGE_REVIEW_LIMIT,
        help=(
            "Number of largest NEARSTAR/INSTAR galaxies to write to the review CSV. "
            f"Default: {DEFAULT_LARGE_REVIEW_LIMIT}"
        ),
    )
    parser.add_argument(
        "--mag-error-floor",
        type=float,
        default=DEFAULT_MAG_ERROR_FLOOR,
        help=(
            "Systematic magnitude uncertainty added in quadrature to each "
            f"CIGALE flux error. Default: {DEFAULT_MAG_ERROR_FLOOR:g} mag."
        ),
    )
    return parser.parse_args()


def normalize_filter_rule(value: Any, band: str) -> tuple[FilterRule, ...]:
    if isinstance(value, str):
        return (FilterRule(name=value),)
    if isinstance(value, dict):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise TypeError(
            f"Filter map value for {band} must be a filter name, object, or list."
        )

    rules: list[FilterRule] = []
    for item in values:
        if isinstance(item, str):
            rules.append(FilterRule(name=item))
            continue
        if not isinstance(item, dict):
            raise TypeError(
                f"Filter map rule for {band} must be a filter name or JSON object."
            )
        extra = sorted(set(item) - {"filter", "dec_lt", "dec_ge"})
        if extra:
            raise KeyError(f"Filter map rule for {band} has unknown keys: {extra}")
        if "filter" not in item:
            raise KeyError(f"Filter map rule for {band} is missing 'filter'.")
        rules.append(
            FilterRule(
                name=str(item["filter"]),
                dec_lt=None if item.get("dec_lt") is None else float(item["dec_lt"]),
                dec_ge=None if item.get("dec_ge") is None else float(item["dec_ge"]),
            )
        )

    if not rules:
        raise ValueError(f"Filter map for {band} must contain at least one rule.")
    return tuple(rules)


def load_filter_map(path: Path | None) -> FilterMap:
    if path is None:
        loaded = DEFAULT_FILTER_MAP
    else:
        with path.open(encoding="utf-8") as handle:
            loaded = json.load(handle)

    if not isinstance(loaded, dict):
        label = "default filter map" if path is None else str(path)
        raise TypeError(f"Filter map must be a JSON object: {label}")

    raw_map = {str(key).upper(): value for key, value in loaded.items()}
    missing = sorted(set(BANDS) - set(raw_map))
    extra = sorted(set(raw_map) - set(BANDS))
    if missing:
        raise KeyError(f"Filter map is missing SGA bands: {missing}")
    if extra:
        raise KeyError(f"Filter map contains unknown SGA bands: {extra}")

    filter_map = {band: normalize_filter_rule(raw_map[band], band) for band in BANDS}
    filter_names = [rule.name for rules in filter_map.values() for rule in rules]
    if len(set(filter_names)) != len(filter_names):
        raise ValueError("Filter map contains duplicate CIGALE filter names.")
    return filter_map


def get_required_columns() -> tuple[list[str], list[str]]:
    sga_cols = [ID_COL, DEC_COL, REDSHIFT_COL, SAMPLE_COL, BANDS_COL]
    sga_cols.extend(REVIEW_COLS)
    sga_cols.extend(f"MW_TRANSMISSION_{band}" for band in BANDS)

    phot_cols = [ID_COL]
    phot_cols.extend(f"FLUX_AP03_{band}" for band in BANDS)
    phot_cols.extend(f"FLUX_ERR_AP03_{band}" for band in BANDS)
    return sga_cols, phot_cols


def require_columns(hdu: fits.BinTableHDU, required: Iterable[str], extname: str) -> None:
    missing = sorted(set(required) - set(hdu.columns.names))
    if missing:
        raise KeyError(f"{extname} is missing required columns: {missing}")


def read_needed_columns(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], fits.Header, fits.Header]:
    sga_cols, phot_cols = get_required_columns()
    with fits.open(path, memmap=True) as hdul:
        sga_hdu = hdul[SGA_EXT]
        phot_hdu = hdul[PHOT_EXT]
        require_columns(sga_hdu, sga_cols, SGA_EXT)
        require_columns(phot_hdu, phot_cols, PHOT_EXT)
        sga = {name: np.array(sga_hdu.data[name]) for name in sga_cols}
        phot = {name: np.array(phot_hdu.data[name]) for name in phot_cols}
        return sga, phot, sga_hdu.header.copy(), phot_hdu.header.copy()


def decode_fits_string_array(values: np.ndarray) -> np.ndarray:
    return np.char.strip(values.astype(str))


def assert_unique_ids(ids: np.ndarray, label: str) -> None:
    unique_count = len(np.unique(ids))
    if unique_count != len(ids):
        raise AssertionError(f"{label} {ID_COL} values are not unique.")


def build_intermediate_columns(
    sga: dict[str, np.ndarray],
    phot: dict[str, np.ndarray],
    sga_header: fits.Header,
    phot_header: fits.Header,
    source_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    sga_ids = np.asarray(sga[ID_COL])
    phot_ids = np.asarray(phot[ID_COL])
    assert_unique_ids(sga_ids, SGA_EXT)
    assert_unique_ids(phot_ids, PHOT_EXT)
    if not np.array_equal(sga_ids, phot_ids):
        raise AssertionError(f"{SGA_EXT} and {PHOT_EXT} rows are not matched by {ID_COL}.")

    n_rows = len(sga_ids)
    dec = np.asarray(sga[DEC_COL], dtype=np.float64)
    redshift = np.asarray(sga[REDSHIFT_COL], dtype=np.float64)
    sample = np.asarray(sga[SAMPLE_COL], dtype=np.int64)
    bands_available = decode_fits_string_array(np.asarray(sga[BANDS_COL]))

    near_star = (sample & NEAR_STAR_BIT_VALUE) != 0
    bright_star = (sample & BRIGHT_STAR_BIT_VALUE) != 0
    either_star_flag = near_star | bright_star
    use_cigale = ~bright_star

    exclusion_reason = np.full(n_rows, "", dtype="U32")
    exclusion_reason[bright_star] = "INSTAR"

    out: dict[str, np.ndarray] = {
        "SGA_ID": sga_ids,
        "use_cigale": use_cigale,
        "exclusion_reason": exclusion_reason,
        "declination": dec,
        "redshift_sga_z": redshift,
        "sample_bitmask": sample,
        "near_star": near_star,
        "bright_star": bright_star,
        "bands": bands_available,
    }

    usable_counts: dict[str, int] = {}
    input_columns: dict[str, list[str]] = {
        SGA_EXT: [ID_COL, DEC_COL, REDSHIFT_COL, SAMPLE_COL, BANDS_COL, *REVIEW_COLS],
        PHOT_EXT: [ID_COL],
    }
    extinction_columns: dict[str, str] = {}

    for band in BANDS:
        flux_col = f"FLUX_AP03_{band}"
        err_col = f"FLUX_ERR_AP03_{band}"
        trans_col = f"MW_TRANSMISSION_{band}"
        input_columns[PHOT_EXT].extend([flux_col, err_col])
        input_columns[SGA_EXT].append(trans_col)
        extinction_columns[band] = trans_col

        observed_flux = np.asarray(phot[flux_col], dtype=np.float64)
        observed_err = np.asarray(phot[err_col], dtype=np.float64)
        transmission = np.asarray(sga[trans_col], dtype=np.float64)

        valid_numeric = (
            np.isfinite(observed_flux)
            & np.isfinite(observed_err)
            & (observed_err > 0.0)
            & np.isfinite(transmission)
            & (transmission > 0.0)
        )

        band_available = valid_numeric.copy()
        if band in OPTICAL_BANDS:
            char = OPTICAL_BANDS[band]
            band_available &= np.char.find(bands_available, char) >= 0
        if band == "Z":
            band_available &= dec < DEC_Z_CUTOFF_DEG

        correction = np.full(n_rows, np.nan, dtype=np.float64)
        good_trans = np.isfinite(transmission) & (transmission > 0.0)
        correction[good_trans] = NANOMAGGY_TO_MJY / transmission[good_trans]

        lower = band.lower()
        out[f"flux_ap03_{lower}_mjy_corr"] = observed_flux * correction
        out[f"flux_err_ap03_{lower}_mjy_corr"] = observed_err * correction
        out[f"mw_transmission_{lower}"] = transmission
        out[f"use_band_{lower}"] = use_cigale & band_available
        usable_counts[band] = int(np.count_nonzero(out[f"use_band_{lower}"]))

    cigale_ids = out["SGA_ID"][use_cigale]
    assert np.all(np.isin(cigale_ids, sga_ids, assume_unique=True))
    assert not np.any(use_cigale & bright_star)
    assert not np.any((dec >= DEC_Z_CUTOFF_DEG) & out["use_band_z"])
    assert np.all(dec[out["use_band_z"]] < DEC_Z_CUTOFF_DEG)

    diagnostics = {
        "source_path": str(source_path),
        "total_sources": n_rows,
        "near_star_count": int(np.count_nonzero(near_star)),
        "bright_star_count": int(np.count_nonzero(bright_star)),
        "near_and_bright_count": int(np.count_nonzero(near_star & bright_star)),
        "either_star_flag_count": int(np.count_nonzero(either_star_flag)),
        "near_star_retained_count": int(np.count_nonzero(use_cigale & near_star)),
        "excluded_count": int(np.count_nonzero(~use_cigale)),
        "remaining_count": int(np.count_nonzero(use_cigale)),
        "remaining_dec_below": int(np.count_nonzero(use_cigale & (dec < DEC_Z_CUTOFF_DEG))),
        "remaining_dec_above_equal": int(
            np.count_nonzero(use_cigale & (dec >= DEC_Z_CUTOFF_DEG))
        ),
        "remaining_with_finite_positive_redshift": int(
            np.count_nonzero(use_cigale & np.isfinite(redshift) & (redshift > 0.0))
        ),
        "usable_counts": usable_counts,
        "input_columns": input_columns,
        "extinction_columns": extinction_columns,
        "sga_header_summary": {
            key: sga_header.get(key) for key in ("EXTNAME", "NAXIS2", "TFIELDS")
        },
        "phot_header_summary": {
            key: phot_header.get(key) for key in ("EXTNAME", "NAXIS2", "TFIELDS")
        },
    }
    return out, diagnostics


def fits_column(name: str, data: np.ndarray) -> fits.Column:
    if data.dtype.kind in {"U", "S", "O"}:
        as_text = data.astype(str)
        width = max(1, max(len(value) for value in as_text))
        return fits.Column(name=name, format=f"{width}A", array=as_text.astype(f"S{width}"))
    if data.dtype.kind == "b":
        return fits.Column(name=name, format="L", array=data)
    if data.dtype.kind in {"i", "u"}:
        return fits.Column(name=name, format="K", array=data.astype(np.int64))
    if data.dtype.kind == "f":
        return fits.Column(name=name, format="D", array=data.astype(np.float64))
    raise TypeError(f"Unsupported dtype for FITS column {name}: {data.dtype}")


def write_intermediate_fits(
    path: Path, columns: dict[str, np.ndarray], diagnostics: dict[str, object]
) -> None:
    coldefs = [fits_column(name, columns[name]) for name in columns]
    hdu = fits.BinTableHDU.from_columns(coldefs, name="SGA_AP03_CIGALE_PREP")
    hdu.header["SRCFILE"] = diagnostics["source_path"]
    hdu.header["IDCOL"] = ("SGAID", "Original SGA unique identifier")
    hdu.header["FLXUNIT"] = ("mJy", "Corrected flux-density unit")
    hdu.header["ORGUNIT"] = ("nanomaggies", "Original AP03 flux unit")
    hdu.header["NMGY2MJ"] = (NANOMAGGY_TO_MJY, "nanomaggy to mJy factor")
    hdu.header["ZDECCUT"] = (DEC_Z_CUTOFF_DEG, "Dec cutoff for z-band rule")
    hdu.header["ZCOL"] = (REDSHIFT_COL, "Redshift column copied to output")
    hdu.header["EXCLPOL"] = ("INSTAR only", "Automatic source-exclusion policy")
    hdu.header["NEARBIT"] = (
        f"{SAMPLE_COL}:{NEAR_STAR_FLAG_NAME}={NEAR_STAR_BIT_VALUE}",
        "NEAR_STAR mapping; tracked but retained",
    )
    hdu.header["BRITBIT"] = (
        f"{SAMPLE_COL}:{BRIGHT_STAR_FLAG_NAME}={BRIGHT_STAR_BIT_VALUE}",
        "BRIGHT_STAR mapping; excluded",
    )
    hdu.header["CREATED"] = datetime.now(timezone.utc).isoformat()
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)


def filter_rule_mask(rule: FilterRule, dec: np.ndarray) -> np.ndarray:
    mask = np.ones(dec.shape, dtype=bool)
    if rule.dec_lt is not None:
        mask &= dec < rule.dec_lt
    if rule.dec_ge is not None:
        mask &= dec >= rule.dec_ge
    return mask


def describe_filter_rule(rule: FilterRule) -> str:
    conditions: list[str] = []
    if rule.dec_lt is not None:
        conditions.append(f"Dec < {rule.dec_lt:g} deg")
    if rule.dec_ge is not None:
        conditions.append(f"Dec >= {rule.dec_ge:g} deg")
    if not conditions:
        return f"`{rule.name}`"
    return f"`{rule.name}` when {' and '.join(conditions)}"


def add_magnitude_error_floor(
    flux: np.ndarray, error: np.ndarray, mag_error_floor: float
) -> np.ndarray:
    """Add a magnitude systematic to a flux-density error in quadrature."""
    if mag_error_floor < 0.0:
        raise ValueError("Magnitude error floor must be non-negative.")
    fractional_floor = np.log(10.0) / 2.5 * mag_error_floor
    return np.hypot(error, fractional_floor * np.abs(flux))


def build_cigale_columns(
    intermediate: dict[str, np.ndarray],
    filter_map: FilterMap,
    mag_error_floor: float = DEFAULT_MAG_ERROR_FLOOR,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    use_cigale = np.asarray(intermediate["use_cigale"], dtype=bool)
    dec = np.asarray(intermediate["declination"], dtype=np.float64)[use_cigale]
    cigale: dict[str, np.ndarray] = {
        "id": np.asarray(intermediate["SGA_ID"][use_cigale]),
        "redshift": np.asarray(intermediate["redshift_sga_z"][use_cigale]),
    }
    filter_supplied_counts: dict[str, int] = {}

    for band in BANDS:
        lower = band.lower()
        source_flux = np.asarray(intermediate[f"flux_ap03_{lower}_mjy_corr"])[use_cigale]
        source_err = np.asarray(intermediate[f"flux_err_ap03_{lower}_mjy_corr"])[use_cigale]
        source_err = add_magnitude_error_floor(
            source_flux, source_err, mag_error_floor
        )
        use_band = np.asarray(intermediate[f"use_band_{lower}"], dtype=bool)[use_cigale]
        for rule in filter_map[band]:
            use_rule = use_band & filter_rule_mask(rule, dec)
            cigale[rule.name] = np.where(use_rule, source_flux, np.nan)
            cigale[f"{rule.name}_err"] = np.where(use_rule, source_err, np.nan)
            filter_supplied_counts[rule.name] = int(np.count_nonzero(use_rule))

    return cigale, filter_supplied_counts


def write_cigale_ascii(path: Path, columns: dict[str, np.ndarray]) -> None:
    names = list(columns)
    n_rows = len(columns[names[0]])
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# " + " ".join(names) + "\n")
        for i in range(n_rows):
            row = [str(int(columns["id"][i]))]
            for name in names[1:]:
                value = float(columns[name][i])
                row.append("NaN" if np.isnan(value) else f"{value:.10g}")
            handle.write(" ".join(row) + "\n")


def clean_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def write_starflag_review_csv(
    path: Path,
    sga: dict[str, np.ndarray],
    intermediate: dict[str, np.ndarray],
    limit: int,
) -> int:
    near_star = np.asarray(intermediate["near_star"], dtype=bool)
    bright_star = np.asarray(intermediate["bright_star"], dtype=bool)
    star_flagged = near_star | bright_star
    d26 = np.asarray(sga["D26"], dtype=np.float64)
    sort_key = np.where(np.isfinite(d26), d26, -np.inf)
    ordered = np.argsort(sort_key)[::-1]
    selected = [idx for idx in ordered if star_flagged[idx]][: max(0, limit)]

    fields = [
        "SGA_ID",
        "SGANAME",
        "GALAXY",
        "OBJNAME",
        "RA",
        "DEC",
        "redshift_sga_z",
        "D26",
        "GROUP_NAME",
        "GROUP_DIAMETER",
        "sample_bitmask",
        "near_star",
        "bright_star",
        "use_cigale",
        "exclusion_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx in selected:
            writer.writerow(
                {
                    "SGA_ID": int(intermediate["SGA_ID"][idx]),
                    "SGANAME": clean_text(sga["SGANAME"][idx]),
                    "GALAXY": clean_text(sga["GALAXY"][idx]),
                    "OBJNAME": clean_text(sga["OBJNAME"][idx]),
                    "RA": f"{float(sga['RA'][idx]):.8f}",
                    "DEC": f"{float(intermediate['declination'][idx]):.8f}",
                    "redshift_sga_z": f"{float(intermediate['redshift_sga_z'][idx]):.8g}",
                    "D26": f"{float(d26[idx]):.6g}",
                    "GROUP_NAME": clean_text(sga["GROUP_NAME"][idx]),
                    "GROUP_DIAMETER": f"{float(sga['GROUP_DIAMETER'][idx]):.6g}",
                    "sample_bitmask": int(intermediate["sample_bitmask"][idx]),
                    "near_star": bool(near_star[idx]),
                    "bright_star": bool(bright_star[idx]),
                    "use_cigale": bool(intermediate["use_cigale"][idx]),
                    "exclusion_reason": clean_text(intermediate["exclusion_reason"][idx]),
                }
            )
    return len(selected)


def write_report(
    path: Path,
    sga_fits: Path,
    intermediate_path: Path,
    cigale_path: Path,
    starflag_review_path: Path,
    diagnostics: dict[str, object],
    filter_map: FilterMap,
    review_rows: int,
) -> None:
    usable_counts: dict[str, int] = diagnostics["usable_counts"]  # type: ignore[assignment]
    filter_supplied_counts: dict[str, int] = diagnostics["filter_supplied_counts"]  # type: ignore[assignment]
    input_columns: dict[str, list[str]] = diagnostics["input_columns"]  # type: ignore[assignment]
    extinction_columns: dict[str, str] = diagnostics["extinction_columns"]  # type: ignore[assignment]
    mag_error_floor = float(diagnostics["mag_error_floor"])

    lines = [
        "# SGA-2025 AP03 CIGALE Preprocessing Diagnostics",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Input FITS: `{sga_fits}`",
        f"Intermediate FITS: `{intermediate_path}`",
        f"CIGALE-facing photometry table: `{cigale_path}`",
        f"Large star-flagged galaxy review table: `{starflag_review_path}`",
        "",
        "## Source Counts",
        "",
        f"- Total SGA sources: {diagnostics['total_sources']}",
        (
            f"- NEAR_STAR diagnostic flag: {diagnostics['near_star_count']} "
            f"(`{SAMPLE_COL}` bit `{NEAR_STAR_FLAG_NAME}` = {NEAR_STAR_BIT_VALUE})"
        ),
        (
            f"- Excluded by BRIGHT_STAR requirement: {diagnostics['bright_star_count']} "
            f"(`{SAMPLE_COL}` bit `{BRIGHT_STAR_FLAG_NAME}` = {BRIGHT_STAR_BIT_VALUE})"
        ),
        f"- Sources with either star flag: {diagnostics['either_star_flag_count']}",
        f"- Sources with both star flags: {diagnostics['near_and_bright_count']}",
        f"- NEAR_STAR sources retained for first-pass CIGALE: {diagnostics['near_star_retained_count']}",
        f"- Sources automatically excluded: {diagnostics['excluded_count']}",
        f"- Remaining for CIGALE: {diagnostics['remaining_count']}",
        (
            f"- Remaining with Dec < {DEC_Z_CUTOFF_DEG} deg: "
            f"{diagnostics['remaining_dec_below']}"
        ),
        (
            f"- Remaining with Dec >= {DEC_Z_CUTOFF_DEG} deg: "
            f"{diagnostics['remaining_dec_above_equal']}"
        ),
        (
            "- Remaining with finite positive `SGA2025.Z`: "
            f"{diagnostics['remaining_with_finite_positive_redshift']}"
        ),
        "",
        "## Usable Measurements",
        "",
    ]
    for band in BANDS:
        lines.append(f"- {band}: {usable_counts[band]}")

    lines.extend(
        [
            "",
            "## Input Columns Used",
            "",
            f"- SGA identifier: `{SGA_EXT}.{ID_COL}`",
            f"- Declination: `{SGA_EXT}.{DEC_COL}`",
            f"- Redshift copied without cuts: `{SGA_EXT}.{REDSHIFT_COL}`",
            f"- Star-flag bitmask: `{SGA_EXT}.{SAMPLE_COL}`",
            f"- Optical availability: `{SGA_EXT}.{BANDS_COL}`",
            (
                "- NEAR_STAR mapping: no standalone `NEAR_STAR` column was used; "
                f"the SGA bitmask documentation defines `{SAMPLE_COL}` bit "
                f"`{NEAR_STAR_FLAG_NAME}` = {NEAR_STAR_BIT_VALUE}. This flag is "
                "tracked but not excluded in the first-pass sample."
            ),
            (
                "- BRIGHT_STAR mapping: no standalone `BRIGHT_STAR` column was used; "
                f"the SGA bitmask documentation defines `{SAMPLE_COL}` bit "
                f"`{BRIGHT_STAR_FLAG_NAME}` = {BRIGHT_STAR_BIT_VALUE}. This flag "
                "is the automatic exclusion for this run."
            ),
            "",
            f"### `{SGA_EXT}`",
            "",
        ]
    )
    for column in input_columns[SGA_EXT]:
        lines.append(f"- `{column}`")
    lines.extend(["", f"### `{PHOT_EXT}`", ""])
    for column in input_columns[PHOT_EXT]:
        lines.append(f"- `{column}`")

    lines.extend(["", "## Extinction Columns", ""])
    for band in BANDS:
        lines.append(f"- {band}: `{SGA_EXT}.{extinction_columns[band]}`")

    lines.extend(
        [
            "",
            "## Unit Conversions",
            "",
            (
                f"- AP03 fluxes and uncertainties are read from `{PHOT_EXT}` in "
                "nanomaggies."
            ),
            (
                f"- Nanomaggies are converted to mJy with "
                f"`flux_mJy = flux_nanomaggy * {NANOMAGGY_TO_MJY:.6g}` "
                "because 1 nanomaggy = 3631 Jy * 1e-9 = 0.003631 mJy."
            ),
            (
                f"- Milky Way extinction is stored as transmission fractions in "
                f"`{SGA_EXT}.MW_TRANSMISSION_{{band}}`; corrected fluxes use "
                "`flux_corr_mJy = flux_nanomaggy * 0.003631 / MW_TRANSMISSION`."
            ),
            (
                "- This is equivalent to using "
                "`A_lambda = -2.5 * log10(MW_TRANSMISSION)` and "
                "`flux_corrected = flux_observed * 10**(0.4 * A_lambda)`."
            ),
            "- Flux uncertainties are scaled by the same multiplicative factor.",
            (
                f"- CIGALE-facing uncertainties include a `{mag_error_floor:g} mag` "
                "systematic added in quadrature: "
                "`sigma_total^2 = sigma_formal^2 + "
                "[(ln(10)/2.5) * sigma_mag * |flux|]^2`."
            ),
            (
                "- The intermediate FITS table retains the formal catalog "
                "uncertainties without this systematic floor."
            ),
            "",
            "## Band-Supply Rules",
            "",
            "- No S/N cuts, missing-band cuts, or photometric-quality cuts were applied.",
            (
                "- A band is supplied only when the source is not excluded by `INSTAR`, "
                "the AP03 flux is finite, the AP03 uncertainty is finite and positive, "
                "the Milky Way transmission is finite and positive, and any optical "
                "coverage rule is satisfied."
            ),
            (
                f"- For Dec >= {DEC_Z_CUTOFF_DEG} deg, z-band values are set to "
                "`NaN` in the CIGALE-facing table. Northern galaxies are retained."
            ),
            "- Negative fluxes are preserved when their uncertainties are usable.",
            (
                "- The CIGALE-facing table uses `NaN` for unavailable measurements; "
                "CIGALE documentation supports `NaN` for missing fluxes."
            ),
            "",
            "## CIGALE Column Naming",
            "",
            "- The CIGALE-facing photometry columns use installed CIGALE filter names.",
            (
                "- Legacy Survey optical AP03 photometry is split by declination so "
                "southern DECam measurements and northern BASS measurements are not "
                "mixed into one synthetic filter column."
            ),
        ]
    )
    for band in BANDS:
        for rule in filter_map[band]:
            count = filter_supplied_counts.get(rule.name, 0)
            lines.append(f"- {band}: {describe_filter_rule(rule)} ({count} supplied)")

    lines.extend(
        [
            (
                "- The Legacy Survey filters were exported from `speclite.filters` "
                "and imported into the local CIGALE filter database."
            ),
            "- `MzLS-z` is exported for reference but not supplied under the current z-band rule.",
            "",
            "## Star-Flag Review",
            "",
            (
                f"- Wrote the largest {review_rows} sources carrying either "
                f"`{NEAR_STAR_FLAG_NAME}` or `{BRIGHT_STAR_FLAG_NAME}` to "
                f"`{starflag_review_path}`."
            ),
            "- This supports later review of very large galaxies affected by Gaia star masks.",
            "",
            "## Assertions Run",
            "",
            f"- Unique `{SGA_EXT}.{ID_COL}` values.",
            f"- Unique `{PHOT_EXT}.{ID_COL}` values.",
            f"- `{SGA_EXT}` and `{PHOT_EXT}` row matching by `{ID_COL}`.",
            "- CIGALE output IDs are a subset of input IDs.",
            "- No `INSTAR` flagged sources reach the CIGALE sample.",
            f"- No z-band measurements are supplied for Dec >= {DEC_Z_CUTOFF_DEG} deg.",
            f"- All supplied z-band measurements have Dec < {DEC_Z_CUTOFF_DEG} deg.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.mag_error_floor < 0.0:
        raise ValueError("--mag-error-floor must be non-negative.")
    filter_map = load_filter_map(args.filter_map)

    input_dir = args.output_dir / "inputs"
    intermediate_dir = args.output_dir / "intermediate"
    report_dir = args.output_dir / "reports"
    for directory in (input_dir, intermediate_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    sga, phot, sga_header, phot_header = read_needed_columns(args.sga_fits)
    intermediate, diagnostics = build_intermediate_columns(
        sga, phot, sga_header, phot_header, args.sga_fits
    )
    cigale, filter_supplied_counts = build_cigale_columns(
        intermediate, filter_map, args.mag_error_floor
    )
    diagnostics["filter_supplied_counts"] = filter_supplied_counts
    diagnostics["mag_error_floor"] = args.mag_error_floor

    intermediate_path = intermediate_dir / "sga2025_ap03_cigale_intermediate.fits"
    cigale_path = input_dir / "sga2025_ap03_cigale_photometry.dat"
    report_path = report_dir / "sga2025_ap03_cigale_preprocessing_report.md"
    starflag_review_path = report_dir / "sga2025_ap03_starflag_large_galaxy_review.csv"

    write_intermediate_fits(intermediate_path, intermediate, diagnostics)
    write_cigale_ascii(cigale_path, cigale)
    review_rows = write_starflag_review_csv(
        starflag_review_path, sga, intermediate, args.large_review_limit
    )
    write_report(
        report_path,
        args.sga_fits,
        intermediate_path,
        cigale_path,
        starflag_review_path,
        diagnostics,
        filter_map,
        review_rows,
    )

    print(f"Wrote {intermediate_path}")
    print(f"Wrote {cigale_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {starflag_review_path}")
    print(f"Remaining for CIGALE: {diagnostics['remaining_count']}")


if __name__ == "__main__":
    main()
