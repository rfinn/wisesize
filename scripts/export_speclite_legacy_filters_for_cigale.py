#!/usr/bin/env python
"""Export speclite Legacy Survey filters for CIGALE.

The output files use the simple text format expected by ``pcigale-filters add``:
filter name, response type, description, then wavelength in Angstrom and
dimensionless response.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy import units as u
from speclite import filters


DEFAULT_OUTPUT_DIR = Path("/Users/rfinn/research/SGA-CIGALE/filters/speclite_legacy")
DEFAULT_FILTERS = (
    ("decamDR1-g", "decamDR1-g"),
    ("decamDR1-r", "decamDR1-r"),
    ("decamDR1-i", "decamDR1-i"),
    ("decamDR1-z", "decamDR1-z"),
    ("BASS-g", "BASS-g"),
    ("BASS-r", "BASS-r"),
    ("MzLS-z", "MzLS-z"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export speclite Legacy Survey filter curves for CIGALE."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for exported filter files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--response-type",
        choices=("photon", "energy"),
        default="photon",
        help="Response type written in the CIGALE filter header. Default: photon",
    )
    parser.add_argument(
        "--add-to-cigale",
        action="store_true",
        help="After export, run pcigale-filters add on the generated files.",
    )
    return parser.parse_args()


def safe_filename(cigale_name: str) -> str:
    return f"{cigale_name.replace('.', '_').replace('-', '_')}.dat"


def angstrom_value(value: object) -> float:
    if hasattr(value, "to_value"):
        return float(value.to_value(u.Angstrom))  # type: ignore[attr-defined]
    return float(value)


def clean_description(value: object) -> str:
    return " ".join(str(value).split())


def export_one(
    speclite_name: str,
    cigale_name: str,
    output_dir: Path,
    response_type: str,
) -> dict[str, object]:
    response = filters.load_filter(speclite_name)
    wavelength = np.asarray(response.wavelength, dtype=np.float64)
    throughput = np.asarray(response.response, dtype=np.float64)
    finite = np.isfinite(wavelength) & np.isfinite(throughput)
    wavelength = wavelength[finite]
    throughput = throughput[finite]

    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    throughput = throughput[order]

    negative_count = int(np.count_nonzero(throughput < 0.0))
    throughput = np.clip(throughput, 0.0, None)

    meta = dict(response.meta)
    description = clean_description(meta.get("description", speclite_name))
    output_path = output_dir / safe_filename(cigale_name)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {cigale_name}\n")
        handle.write(f"# {response_type}\n")
        handle.write(f"# speclite {speclite_name}: {description}\n")
        for wave, trans in zip(wavelength, throughput):
            handle.write(f"{wave:.10g} {trans:.10g}\n")

    return {
        "speclite_name": speclite_name,
        "cigale_name": cigale_name,
        "path": str(output_path),
        "response_type": response_type,
        "n_points": int(wavelength.size),
        "wavelength_min_angstrom": float(wavelength.min()),
        "wavelength_max_angstrom": float(wavelength.max()),
        "effective_wavelength_angstrom": angstrom_value(response.effective_wavelength),
        "airmass": meta.get("airmass"),
        "description": description,
        "negative_values_clipped": negative_count,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_module": "speclite.filters",
        "output_dir": str(args.output_dir),
        "filters": [
            export_one(speclite_name, cigale_name, args.output_dir, args.response_type)
            for speclite_name, cigale_name in DEFAULT_FILTERS
        ],
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    paths = [item["path"] for item in manifest["filters"]]
    if args.add_to_cigale:
        subprocess.run(["pcigale-filters", "add", *paths], check=True)

    for path in paths:
        print(path)
    print(manifest_path)


if __name__ == "__main__":
    main()
