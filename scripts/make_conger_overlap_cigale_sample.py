#!/usr/bin/env python
"""Build a small SGA2025/AP03 CIGALE test sample from Conger's VFS input."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

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

from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u


DEFAULT_CONGER_INPUT = Path("/Users/rfinn/research/SGA-CIGALE/conger_test/vf_data.txt")
DEFAULT_CONGER_INI = Path("/Users/rfinn/research/SGA-CIGALE/conger_test/pcigale.ini")
DEFAULT_VF_MAIN = Path(
    "/Users/rfinn/research/Virgo/tables-north/v2/vf_v2_main.fits"
)
DEFAULT_SGA = Path("/Users/rfinn/research/SGA2025/SGA2025-v1.0.fits")
DEFAULT_SGA_CIGALE = Path(
    "/Users/rfinn/research/SGA-CIGALE/inputs/sga2025_ap03_cigale_photometry.dat"
)
DEFAULT_OUTPUT = Path(
    "/Users/rfinn/research/SGA-CIGALE/inputs/conger_overlap_sga2025_ap03_cigale20.dat"
)
DEFAULT_MATCHES = Path(
    "/Users/rfinn/research/SGA-CIGALE/reports/conger_overlap_sga2025_matches.csv"
)
DEFAULT_RUN_DIR = Path(
    "/Users/rfinn/research/SGA-CIGALE/cigale_runs/conger_overlap_sga2025_ap03_20"
)
DEFAULT_DEC_CUT = 32.375


OUTPUT_BANDS = [
    "galex.FUV",
    "galex.NUV",
    "decamDR1-g",
    "BASS-g",
    "decamDR1-r",
    "BASS-r",
    "decamDR1-i",
    "decamDR1-z",
    "wise.W1",
    "wise.W2",
    "wise.W3",
    "wise.W4",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a VFS/Conger overlap CIGALE sample using new SGA2025 AP03 photometry."
    )
    parser.add_argument("--conger-input", type=Path, default=DEFAULT_CONGER_INPUT)
    parser.add_argument("--conger-ini", type=Path, default=DEFAULT_CONGER_INI)
    parser.add_argument("--vf-main", type=Path, default=DEFAULT_VF_MAIN)
    parser.add_argument("--sga", type=Path, default=DEFAULT_SGA)
    parser.add_argument("--sga-cigale", type=Path, default=DEFAULT_SGA_CIGALE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--match-radius-arcsec", type=float, default=10.0)
    parser.add_argument("--min-bands", type=int, default=8)
    parser.add_argument("--dec-cut", type=float, default=DEFAULT_DEC_CUT)
    parser.add_argument(
        "--no-balance-dec",
        action="store_true",
        help="Select the first qualifying rows instead of balancing across --dec-cut.",
    )
    return parser.parse_args()


def read_cigale_ascii(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("#"):
                names = stripped[1:].strip().split()
                break
        else:
            raise ValueError(f"No commented header found in {path}")
    data = np.genfromtxt(path, comments="#", dtype=str, encoding="utf-8")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != len(names):
        raise ValueError(f"{path} has {data.shape[1]} columns but {len(names)} names")
    return names, data


def numeric_column(names: list[str], data: np.ndarray, name: str) -> np.ndarray:
    return data[:, names.index(name)].astype(float)


def usable_band_count(names: list[str], data: np.ndarray) -> np.ndarray:
    counts = np.zeros(data.shape[0], dtype=int)
    for band in OUTPUT_BANDS:
        if band not in names or f"{band}_err" not in names:
            continue
        flux = numeric_column(names, data, band)
        err = numeric_column(names, data, f"{band}_err")
        counts += np.isfinite(flux) & np.isfinite(err) & (flux > 0.0) & (err > 0.0)
    return counts


def match_vf_to_sga(
    conger_ids: set[str],
    vf_main_path: Path,
    sga_path: Path,
    radius_arcsec: float,
) -> dict[str, dict[str, object]]:
    vf = Table.read(vf_main_path)
    vf = vf[[str(vfid) in conger_ids for vfid in vf["VFID"]]]
    sga = Table.read(sga_path, hdu="SGA2025")

    vf_coord = SkyCoord(np.asarray(vf["RA"], float) * u.deg, np.asarray(vf["DEC"], float) * u.deg)
    sga_coord = SkyCoord(np.asarray(sga["RA"], float) * u.deg, np.asarray(sga["DEC"], float) * u.deg)
    nearest, sep, _ = vf_coord.match_to_catalog_sky(sga_coord)

    matches: dict[str, dict[str, object]] = {}
    for vf_row, sga_idx, sep_arcsec in zip(vf, nearest, sep.arcsec):
        if sep_arcsec > radius_arcsec:
            continue
        vfid = str(vf_row["VFID"])
        matches[vfid] = {
            "vfid": vfid,
            "vf_ra": float(vf_row["RA"]),
            "vf_dec": float(vf_row["DEC"]),
            "vf_objname": str(vf_row["objname"]),
            "vf_nedname": str(vf_row["NEDname"]),
            "sgaid": int(sga["SGAID"][sga_idx]),
            "sga_ra": float(sga["RA"][sga_idx]),
            "sga_dec": float(sga["DEC"][sga_idx]),
            "sga_name": str(sga["SGANAME"][sga_idx]),
            "separation_arcsec": float(sep_arcsec),
        }
    return matches


def write_sample(
    names: list[str],
    data: np.ndarray,
    conger_names: list[str],
    conger_data: np.ndarray,
    matches: dict[str, dict[str, object]],
    n_rows: int,
    min_bands: int,
    dec_cut: float,
    balance_dec: bool,
    output_path: Path,
) -> tuple[list[dict[str, object]], list[str]]:
    sga_row_by_id = {row[names.index("id")]: row for row in data}
    conger_ids = list(conger_data[:, conger_names.index("id")])
    counts = usable_band_count(names, data)
    count_by_id = {data[i, names.index("id")]: int(counts[i]) for i in range(data.shape[0])}

    candidates: list[tuple[np.ndarray, dict[str, object]]] = []
    missing: list[str] = []
    for vfid in conger_ids:
        match = matches.get(vfid)
        if match is None:
            missing.append(vfid)
            continue
        sgaid = str(match["sgaid"])
        row = sga_row_by_id.get(sgaid)
        if row is None:
            missing.append(vfid)
            continue
        if count_by_id[sgaid] < min_bands:
            continue
        outrow = row.copy()
        outrow[names.index("id")] = vfid
        candidates.append((outrow, match | {"usable_bands": count_by_id[sgaid]}))

    if balance_dec:
        south_target = n_rows // 2
        north_target = n_rows - south_target
        south = [item for item in candidates if float(item[1]["sga_dec"]) < dec_cut]
        north = [item for item in candidates if float(item[1]["sga_dec"]) >= dec_cut]
        if len(south) < south_target or len(north) < north_target:
            raise RuntimeError(
                "Could not build a balanced conger sample: "
                f"need {south_target} south/{north_target} north rows, found "
                f"{len(south)} south/{len(north)} north."
            )
        chosen = south[:south_target] + north[:north_target]
        chosen.sort(key=lambda item: item[1]["vfid"])
    else:
        chosen = candidates[:n_rows]

    if len(chosen) < n_rows:
        raise RuntimeError(f"Only selected {len(chosen)} rows; requested {n_rows}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# " + " ".join(names) + "\n")
        for row, _match in chosen:
            handle.write(" ".join("NaN" if value.lower() == "nan" else value for value in row) + "\n")
    return [match for _row, match in chosen], missing


def write_matches(path: Path, selected: list[dict[str, object]], missing: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "vfid",
        "sgaid",
        "separation_arcsec",
        "usable_bands",
        "vf_ra",
        "vf_dec",
        "sga_ra",
        "sga_dec",
        "vf_objname",
        "vf_nedname",
        "sga_name",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    missing_path = path.with_name(path.stem + "_unmatched_first_pass.txt")
    missing_path.write_text("\n".join(missing[:200]) + ("\n" if missing else ""), encoding="utf-8")


def rewrite_pcigale_ini(template: Path, run_dir: Path, data_file: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template.with_suffix(".ini.spec"), run_dir / "pcigale.ini.spec")
    lines = template.read_text(encoding="utf-8").splitlines()
    fit_bands = []
    for band in OUTPUT_BANDS:
        fit_bands.extend([band, f"{band}_err"])
    analysis_bands = ", ".join(OUTPUT_BANDS)
    fit_band_line = ", ".join(fit_bands)

    in_analysis = False
    rewritten: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == "[analysis_params]":
            in_analysis = True
            rewritten.append(line)
        elif stripped.startswith("[") and stripped != "[analysis_params]":
            in_analysis = False
            rewritten.append(line)
        elif stripped.startswith("data_file ="):
            rewritten.append(f"data_file = {data_file}")
        elif not in_analysis and stripped.startswith("bands ="):
            rewritten.append(f"bands = {fit_band_line}")
        elif in_analysis and stripped.startswith("bands ="):
            rewritten.append(f"  bands = {analysis_bands}")
        elif stripped.startswith("filters = V_B90 &"):
            rewritten.append("    filters = generic.bessell.V & galex.FUV")
        else:
            rewritten.append(line)

    (run_dir / "pcigale.ini").write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    conger_names, conger_data = read_cigale_ascii(args.conger_input)
    sga_names, sga_data = read_cigale_ascii(args.sga_cigale)
    conger_ids = set(conger_data[:, conger_names.index("id")])
    matches = match_vf_to_sga(conger_ids, args.vf_main, args.sga, args.match_radius_arcsec)
    selected, missing = write_sample(
        sga_names,
        sga_data,
        conger_names,
        conger_data,
        matches,
        args.n,
        args.min_bands,
        args.dec_cut,
        not args.no_balance_dec,
        args.output,
    )
    write_matches(args.matches, selected, missing)
    rewrite_pcigale_ini(args.conger_ini, args.run_dir, args.output)
    print(f"Matched {len(matches)} / {len(conger_ids)} Conger IDs within {args.match_radius_arcsec} arcsec.")
    print(f"Wrote {len(selected)} rows to {args.output}")
    print(f"Wrote match report to {args.matches}")
    print(f"Wrote CIGALE run config to {args.run_dir / 'pcigale.ini'}")


if __name__ == "__main__":
    main()
