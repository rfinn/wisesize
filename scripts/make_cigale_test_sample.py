#!/usr/bin/env python
"""Make a small deterministic CIGALE input file from SGA-2025 AP03 preprocessing."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

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


DEFAULT_INPUT = Path(
    "/Users/rfinn/research/SGA-CIGALE/inputs/sga2025_ap03_cigale_photometry.dat"
)
DEFAULT_OUTPUT = Path(
    "/Users/rfinn/research/SGA-CIGALE/inputs/sga2025_ap03_cigale_smoke20.dat"
)
DEFAULT_REPORT = Path(
    "/Users/rfinn/research/SGA-CIGALE/reports/sga2025_ap03_cigale_smoke20_report.md"
)
DEFAULT_INTERMEDIATE = Path(
    "/Users/rfinn/research/SGA-CIGALE/intermediate/sga2025_ap03_cigale_intermediate.fits"
)
DEFAULT_N = 20
DEFAULT_MIN_BANDS = 8
DEFAULT_DEC_CUT = 32.375


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a small SGA-2025 AP03 CIGALE smoke-test input file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Full CIGALE-facing photometry table. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Smoke-test CIGALE table to write. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"Smoke-test selection report to write. Default: {DEFAULT_REPORT}",
    )
    parser.add_argument(
        "--intermediate-fits",
        type=Path,
        default=DEFAULT_INTERMEDIATE,
        help=(
            "Intermediate preprocessing FITS table used to balance the smoke sample "
            f"across the Dec cut. Default: {DEFAULT_INTERMEDIATE}"
        ),
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        help=f"Number of rows to select. Default: {DEFAULT_N}",
    )
    parser.add_argument(
        "--min-bands",
        type=int,
        default=DEFAULT_MIN_BANDS,
        help=f"Minimum usable photometric bands for selected rows. Default: {DEFAULT_MIN_BANDS}",
    )
    parser.add_argument(
        "--allow-negative-flux",
        action="store_true",
        help="Allow finite negative fluxes to count toward --min-bands.",
    )
    parser.add_argument(
        "--dec-cut",
        type=float,
        default=DEFAULT_DEC_CUT,
        help=f"Declination cut used for balanced smoke selection. Default: {DEFAULT_DEC_CUT}",
    )
    parser.add_argument(
        "--no-balance-dec",
        action="store_true",
        help="Select the first qualifying rows instead of balancing across --dec-cut.",
    )
    parser.add_argument(
        "--z-min",
        type=float,
        default=None,
        help="Optional inclusive lower redshift bound for selected rows.",
    )
    parser.add_argument(
        "--z-max",
        type=float,
        default=None,
        help="Optional exclusive upper redshift bound for selected rows.",
    )
    return parser.parse_args()


def read_header_names(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("#"):
                names = stripped[1:].strip().split()
                if names:
                    return names
    raise ValueError(f"No commented header line found in {path}")


def read_data(path: Path, names: list[str]) -> np.ndarray:
    data = np.genfromtxt(path, comments="#", dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != len(names):
        raise ValueError(
            f"Expected {len(names)} columns from header but read {data.shape[1]} columns."
        )
    return data


def read_dec_for_input_rows(intermediate_path: Path, ids: np.ndarray) -> np.ndarray:
    with fits.open(intermediate_path, memmap=True) as hdul:
        table = hdul["SGA_AP03_CIGALE_PREP"].data
        use = np.asarray(table["use_cigale"], dtype=bool)
        source_ids = np.asarray(table["SGA_ID"], dtype=np.int64)[use]
        source_dec = np.asarray(table["declination"], dtype=np.float64)[use]

    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) == len(source_ids) and np.array_equal(ids, source_ids):
        return source_dec

    order = np.argsort(source_ids)
    sorted_ids = source_ids[order]
    positions = np.searchsorted(sorted_ids, ids)
    matched = (
        positions < len(sorted_ids)
    ) & (sorted_ids[np.minimum(positions, len(sorted_ids) - 1)] == ids)
    if not np.all(matched):
        missing = ids[~matched][:10]
        raise ValueError(
            f"Could not match input IDs to intermediate FITS rows: {missing.tolist()}"
        )
    return source_dec[order][positions]


def photometry_columns(names: list[str]) -> list[str]:
    return [
        name
        for name in names
        if name not in {"id", "redshift"} and not name.endswith("_err")
    ]


def usable_band_counts(
    data: np.ndarray,
    names: list[str],
    require_positive_flux: bool,
) -> tuple[np.ndarray, list[str]]:
    counts = np.zeros(data.shape[0], dtype=np.int64)
    usable_filter_names: list[str] = []
    for flux_name in photometry_columns(names):
        err_name = f"{flux_name}_err"
        if err_name not in names:
            continue
        flux = data[:, names.index(flux_name)]
        err = data[:, names.index(err_name)]
        usable = np.isfinite(flux) & np.isfinite(err) & (err > 0.0)
        if require_positive_flux:
            usable &= flux > 0.0
        counts += usable
        usable_filter_names.append(flux_name)
    return counts, usable_filter_names


def select_rows(
    data: np.ndarray,
    names: list[str],
    n_rows: int,
    min_bands: int,
    require_positive_flux: bool,
    dec: np.ndarray | None,
    dec_cut: float,
    balance_dec: bool,
    z_min: float | None,
    z_max: float | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    counts, usable_filter_names = usable_band_counts(data, names, require_positive_flux)
    ok = counts >= min_bands
    if "redshift" in names:
        redshift = data[:, names.index("redshift")]
        ok &= np.isfinite(redshift) & (redshift > 0.0)
        if z_min is not None:
            ok &= redshift >= z_min
        if z_max is not None:
            ok &= redshift < z_max

    if balance_dec:
        if dec is None:
            raise ValueError("Balanced Dec selection requires --intermediate-fits.")
        south_target = n_rows // 2
        north_target = n_rows - south_target
        south = np.flatnonzero(ok & (dec < dec_cut))
        north = np.flatnonzero(ok & (dec >= dec_cut))
        if len(south) < south_target or len(north) < north_target:
            raise RuntimeError(
                "Could not build a balanced smoke sample: "
                f"need {south_target} south/{north_target} north rows, found "
                f"{len(south)} south/{len(north)} north."
            )
        selected = np.concatenate([south[:south_target], north[:north_target]])
        selected.sort()
        return selected, counts, usable_filter_names

    selected = np.flatnonzero(ok)[:n_rows]
    if len(selected) < n_rows:
        raise RuntimeError(
            f"Only found {len(selected)} rows satisfying n={n_rows}, "
            f"min_bands={min_bands}, require_positive_flux={require_positive_flux}."
        )
    return selected, counts, usable_filter_names


def format_value(name: str, value: float) -> str:
    if name == "id":
        return str(int(value))
    return "NaN" if np.isnan(value) else f"{value:.10g}"


def write_cigale_ascii(path: Path, names: list[str], rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# " + " ".join(names) + "\n")
        for row in rows:
            handle.write(" ".join(format_value(name, value) for name, value in zip(names, row)))
            handle.write("\n")


def write_report(
    path: Path,
    input_path: Path,
    output_path: Path,
    names: list[str],
    data: np.ndarray,
    selected: np.ndarray,
    counts: np.ndarray,
    usable_filter_names: list[str],
    min_bands: int,
    require_positive_flux: bool,
    dec: np.ndarray | None,
    dec_cut: float,
    balance_dec: bool,
    z_min: float | None,
    z_max: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    id_index = names.index("id")
    lines = [
        "# SGA-2025 AP03 CIGALE Smoke-Test Sample",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Input table: `{input_path}`",
        f"Output table: `{output_path}`",
        "",
        "## Selection",
        "",
        f"- Input rows: {data.shape[0]}",
        f"- Selected rows: {len(selected)}",
        f"- Minimum usable bands: {min_bands}",
        f"- Required finite positive redshift: {'redshift' in names}",
        f"- Redshift lower bound: {z_min if z_min is not None else 'none'}",
        f"- Redshift upper bound: {z_max if z_max is not None else 'none'}",
        f"- Required positive flux to count as usable: {require_positive_flux}",
        f"- Balanced across Dec cut: {balance_dec}",
        f"- Dec cut: {dec_cut:g}",
        "",
        "## Filters Counted",
        "",
    ]
    for name in usable_filter_names:
        lines.append(f"- `{name}`")
    lines.extend(["", "## Selected IDs", ""])
    for idx in selected:
        suffix = ""
        if dec is not None:
            region = "north" if dec[idx] >= dec_cut else "south"
            suffix = f", Dec={dec[idx]:.6f} ({region})"
        lines.append(
            f"- {int(data[idx, id_index])}: {int(counts[idx])} usable bands{suffix}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.n <= 0:
        raise ValueError("--n must be positive.")
    if args.min_bands <= 0:
        raise ValueError("--min-bands must be positive.")

    names = read_header_names(args.input)
    data = read_data(args.input, names)
    dec = None
    if not args.no_balance_dec:
        dec = read_dec_for_input_rows(args.intermediate_fits, data[:, names.index("id")])
    selected, counts, usable_filter_names = select_rows(
        data,
        names,
        args.n,
        args.min_bands,
        require_positive_flux=not args.allow_negative_flux,
        dec=dec,
        dec_cut=args.dec_cut,
        balance_dec=not args.no_balance_dec,
        z_min=args.z_min,
        z_max=args.z_max,
    )
    write_cigale_ascii(args.output, names, data[selected])
    write_report(
        args.report,
        args.input,
        args.output,
        names,
        data,
        selected,
        counts,
        usable_filter_names,
        args.min_bands,
        require_positive_flux=not args.allow_negative_flux,
        dec=dec,
        dec_cut=args.dec_cut,
        balance_dec=not args.no_balance_dec,
        z_min=args.z_min,
        z_max=args.z_max,
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
