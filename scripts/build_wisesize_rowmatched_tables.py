#!/usr/bin/env python
"""Build row-matched WISEsize tables tied to the z < 0.025 SGA sample."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Column, MaskedColumn, Table
from astropy.units import UnitsWarning


SGA_PATH = Path("/Users/rfinn/research/SGA2025/SGA2025-v1.0.fits")
OUTDIR = Path("/Users/rfinn/research/WISEsize/tables")
MATCH_RADIUS = 30.0 * u.arcsec

CIGALE_PATH = Path(
    "/Users/rfinn/research/SGA-CIGALE/results/"
    "sga2025_ap04_zlt0025_parent_errfloor0p10mag_agnfrac0to0p5_results.fits"
)
VF_PATH = Path("/Users/rfinn/research/Virgo/tables-north/v2/vf_v2_main.fits")
VIRGOWISE_PATH = Path("/Users/rfinn/research/WISEsize/conger_tables/virgowise_data.fits")
GALFIT_R_PATH = Path("/Users/rfinn/research/WISEsize/conger_tables/wisesize_galfit_r.fits")
GALFIT_W1_PATH = Path("/Users/rfinn/research/WISEsize/conger_tables/wisesize_galfit_W1-fixBA.fits")
GALFIT_W3_PATH = Path("/Users/rfinn/research/WISEsize/conger_tables/wisesize_galfit_W3-fixBA.fits")
OLD_PARENT_PATH = Path("/Users/rfinn/research/WISEsize/conger_tables/wisesize_parent_v0.fits")
M200_PATH = Path("/Users/rfinn/research/WISEsize/conger_tables/wisesize_m200.fits")
NEDLVS_PATH = Path("/Users/rfinn/research/NED-LVS/NEDLVS_20250602.fits")

INPUT_PATHS = [
    SGA_PATH,
    CIGALE_PATH,
    VF_PATH,
    VIRGOWISE_PATH,
    GALFIT_R_PATH,
    GALFIT_W1_PATH,
    GALFIT_W3_PATH,
    OLD_PARENT_PATH,
    M200_PATH,
    NEDLVS_PATH,
]

MATCH_TABLES = [
    {
        "path": CIGALE_PATH,
        "output": "wisesize_cigale_v1.fits",
        "method": "sgaid",
    },
    {
        "path": VF_PATH,
        "output": "wisesize_vfs_v1.fits",
        "method": "coord",
        "ra": "RA",
        "dec": "DEC",
    },
    {
        "path": VIRGOWISE_PATH,
        "output": "wisesize_virgowise.fits",
        "method": "coord",
        "ra": "RA",
        "dec": "DEC",
    },
    {
        "path": GALFIT_R_PATH,
        "output": "wisesize_galfit_r_v1.fits",
        "method": "padded_objid_sgaid",
    },
    {
        "path": GALFIT_W1_PATH,
        "output": "wisesize_galfit_W1_fixBA_v1.fits",
        "method": "padded_objid_sgaid",
    },
    {
        "path": GALFIT_W3_PATH,
        "output": "wisesize_galfit_W3_fixBA_v1.fits",
        "method": "padded_objid_sgaid",
    },
    {
        "path": OLD_PARENT_PATH,
        "output": "wisesize_parent_v0_matched_v1.fits",
        "method": "sgaid",
    },
    {
        "path": M200_PATH,
        "output": "wisesize_m200_v1.fits",
        "method": "sgaid",
        "hdu": "Joined",
    },
    {
        "path": NEDLVS_PATH,
        "output": "wisesize_nedlvs_v1.fits",
        "method": "coord",
        "ra": "ra",
        "dec": "dec",
    },
]


def inspect_fits(paths: list[Path]) -> None:
    for path in paths:
        print(f"\nPATH {path}")
        with fits.open(path, memmap=True) as hdul:
            for index, hdu in enumerate(hdul):
                shape = getattr(hdu.data, "shape", None)
                if hasattr(hdu, "columns"):
                    names = list(hdu.columns.names)
                    print(
                        f"{index:2d} {hdu.name:20s} shape={shape} "
                        f"ncols={len(names)} firstcols={names[:30]}"
                    )
                else:
                    print(f"{index:2d} {hdu.name:20s} shape={shape}")


def as_plain_array(column: Column | MaskedColumn, fill_value=np.nan) -> np.ndarray:
    if isinstance(column, MaskedColumn):
        return np.asarray(column.filled(fill_value))
    return np.asarray(column)


def snr_column(ellipse: Table, flux_col: str, err_col: str) -> np.ndarray:
    flux = as_plain_array(ellipse[flux_col], fill_value=np.nan).astype(float)
    err = as_plain_array(ellipse[err_col], fill_value=np.nan).astype(float)
    snr = np.full(len(ellipse), np.nan, dtype=np.float32)
    good = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    snr[good] = flux[good] / err[good]
    return snr


def build_parent() -> Table:
    sga = Table.read(SGA_PATH, hdu="SGA2025")
    ellipse = Table.read(SGA_PATH, hdu="ELLIPSEPHOT")

    if not np.array_equal(np.asarray(sga["SGAID"]), np.asarray(ellipse["SGAID"])):
        raise ValueError("SGA2025 and ELLIPSEPHOT extensions are not aligned by SGAID.")

    z = as_plain_array(sga["Z"], fill_value=np.nan).astype(float)
    parent_mask = (z < 0.025) & (z != 0.0)
    parent = sga[parent_mask]
    parent.sort("SGAID")

    sgaid = np.asarray(parent["SGAID"])
    if len(np.unique(sgaid)) != len(sgaid):
        raise ValueError("Parent SGAID values are not unique after z < 0.025 selection.")

    ellipse_parent = ellipse[parent_mask]
    order = np.argsort(np.asarray(ellipse_parent["SGAID"]))
    ellipse_parent = ellipse_parent[order]
    if not np.array_equal(np.asarray(parent["SGAID"]), np.asarray(ellipse_parent["SGAID"])):
        raise ValueError("Parent and ELLIPSEPHOT rows are not aligned after SGAID sort.")

    snr_specs = [
        ("SNR_AP01_W1", "FLUX_AP01_W1", "FLUX_ERR_AP01_W1"),
        ("SNR_AP01_W3", "FLUX_AP01_W3", "FLUX_ERR_AP01_W3"),
        ("SNR_AP03_W1", "FLUX_AP03_W1", "FLUX_ERR_AP03_W1"),
        ("SNR_AP03_W3", "FLUX_AP03_W3", "FLUX_ERR_AP03_W3"),
    ]
    for out_col, flux_col, err_col in snr_specs:
        parent[out_col] = snr_column(ellipse_parent, flux_col, err_col)

    snrs = np.vstack([np.asarray(parent[name]) for name, _, _ in snr_specs])
    parent["WISESIZE_FLAG"] = np.any(snrs > 10.0, axis=0)
    parent["RADEC_FLAG"] = (
        (as_plain_array(parent["RA"], fill_value=np.nan) > 87.0)
        & (as_plain_array(parent["RA"], fill_value=np.nan) < 300.0)
        & (as_plain_array(parent["DEC"], fill_value=np.nan) > -10.0)
        & (as_plain_array(parent["DEC"], fill_value=np.nan) < 85.0)
    )
    return parent


def normalize_sgaid(values: Column | MaskedColumn) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind in "iu":
        return arr.astype(np.int64)

    out = np.full(len(arr), -1, dtype=np.int64)
    for i, value in enumerate(arr):
        if isinstance(value, bytes | np.bytes_):
            text = value.decode("utf-8", errors="ignore")
        else:
            text = str(value)
        text = text.strip()
        if text:
            out[i] = int(text)
    return out


def first_index_by_key(keys: np.ndarray) -> tuple[dict[int, int], int]:
    index: dict[int, int] = {}
    duplicates = 0
    for i, key in enumerate(keys):
        key_int = int(key)
        if key_int < 0:
            continue
        if key_int in index:
            duplicates += 1
            continue
        index[key_int] = i
    return index, duplicates


def match_by_sgaid(parent: Table, source: Table, key_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    source_keys = normalize_sgaid(source[key_col])
    lookup, duplicates = first_index_by_key(source_keys)
    matched_indices = np.full(len(parent), -1, dtype=np.int64)
    for parent_i, sgaid in enumerate(np.asarray(parent["SGAID"])):
        matched_indices[parent_i] = lookup.get(int(sgaid), -1)
    match_flag = matched_indices >= 0
    return matched_indices, match_flag, None, duplicates


def match_by_coord(
    parent: Table,
    source: Table,
    ra_col: str,
    dec_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    parent_ra = as_plain_array(parent["RA"], fill_value=np.nan).astype(float)
    parent_dec = as_plain_array(parent["DEC"], fill_value=np.nan).astype(float)
    source_ra = as_plain_array(source[ra_col], fill_value=np.nan).astype(float)
    source_dec = as_plain_array(source[dec_col], fill_value=np.nan).astype(float)

    parent_good = np.isfinite(parent_ra) & np.isfinite(parent_dec)
    source_good = np.isfinite(source_ra) & np.isfinite(source_dec)
    source_good_index = np.flatnonzero(source_good)

    matched_indices = np.full(len(parent), -1, dtype=np.int64)
    matched_dist = np.full(len(parent), np.nan, dtype=np.float32)
    if len(source_good_index) == 0:
        return matched_indices, matched_indices >= 0, matched_dist, 0

    parent_coord = SkyCoord(parent_ra[parent_good] * u.deg, parent_dec[parent_good] * u.deg)
    source_coord = SkyCoord(source_ra[source_good] * u.deg, source_dec[source_good] * u.deg)
    nearest, dist, _ = parent_coord.match_to_catalog_sky(source_coord)
    parent_good_index = np.flatnonzero(parent_good)
    within = dist < MATCH_RADIUS

    matched_parent_rows = parent_good_index[within]
    matched_indices[matched_parent_rows] = source_good_index[nearest[within]]
    matched_dist[matched_parent_rows] = dist[within].to_value(u.arcsec).astype(np.float32)
    return matched_indices, matched_indices >= 0, matched_dist, 0


def empty_like_column(source_col: Column | MaskedColumn, length: int) -> MaskedColumn:
    shape = getattr(source_col, "shape", (len(source_col),))[1:]
    data = np.empty((length,) + shape, dtype=source_col.dtype)
    mask = np.ones((length,) + shape, dtype=bool)
    return MaskedColumn(data=data, mask=mask, name=source_col.name, unit=getattr(source_col, "unit", None))


def build_matched_table(parent: Table, spec: dict[str, object]) -> tuple[Table, int, int]:
    source = Table.read(spec["path"], hdu=spec.get("hdu"))
    method = spec["method"]
    if method == "sgaid":
        matched_indices, match_flag, match_dist, duplicates = match_by_sgaid(parent, source, "SGAID")
    elif method == "padded_objid_sgaid":
        matched_indices, match_flag, match_dist, duplicates = match_by_sgaid(parent, source, "OBJID")
    elif method == "coord":
        matched_indices, match_flag, match_dist, duplicates = match_by_coord(
            parent,
            source,
            str(spec["ra"]),
            str(spec["dec"]),
        )
    else:
        raise ValueError(f"Unknown match method: {method}")

    out = Table(masked=True)
    out["SGAID"] = np.asarray(parent["SGAID"], dtype=np.int64)
    out["MATCH_FLAG"] = np.asarray(match_flag, dtype=bool)
    if match_dist is not None:
        out["MATCH_DIST_ARCSEC"] = match_dist

    good_parent_rows = np.flatnonzero(match_flag)
    good_source_rows = matched_indices[good_parent_rows]
    for colname in source.colnames:
        out_colname = colname if colname not in out.colnames else f"SOURCE_{colname}"
        col = empty_like_column(source[colname], len(parent))
        col.name = out_colname
        col[good_parent_rows] = source[colname][good_source_rows]
        out.add_column(col)

    return out, int(np.count_nonzero(match_flag)), duplicates


def validate_order(table: Table, parent_sgaid: np.ndarray, label: str) -> None:
    if len(table) != len(parent_sgaid):
        raise ValueError(f"{label}: row count {len(table)} != parent row count {len(parent_sgaid)}")
    if not np.array_equal(np.asarray(table["SGAID"], dtype=np.int64), parent_sgaid):
        raise ValueError(f"{label}: SGAID order does not match parent table")


def write_table(table: Table, path: Path) -> None:
    table.write(path, overwrite=True)


def build_all() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    parent = build_parent()
    parent_sgaid = np.asarray(parent["SGAID"], dtype=np.int64)

    parent_path = OUTDIR / "wisesize_sga_v1.fits"
    write_table(parent, parent_path)
    validate_order(Table.read(parent_path), parent_sgaid, parent_path.name)

    print(f"Wrote {parent_path} rows={len(parent)} WISESIZE_FLAG={np.count_nonzero(parent['WISESIZE_FLAG'])}")
    print(f"RADEC_FLAG={np.count_nonzero(parent['RADEC_FLAG'])}")

    for spec in MATCH_TABLES:
        matched, nmatch, duplicates = build_matched_table(parent, spec)
        outpath = OUTDIR / str(spec["output"])
        write_table(matched, outpath)
        validate_order(Table.read(outpath), parent_sgaid, outpath.name)
        print(
            f"Wrote {outpath} rows={len(matched)} matches={nmatch} "
            f"unmatched={len(matched) - nmatch} duplicate_source_keys={duplicates}"
        )

    print("Validated identical row count and SGAID order for all outputs.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true", help="Print FITS HDU summaries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inspect:
        inspect_fits(INPUT_PATHS)
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnitsWarning)
        build_all()


if __name__ == "__main__":
    main()
