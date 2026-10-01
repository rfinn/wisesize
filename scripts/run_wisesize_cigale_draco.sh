#!/usr/bin/env bash

# Run the redshift-sorted WISEsize CIGALE chunks sequentially on Draco.

set -euo pipefail

ROOT=${SGA_CIGALE_ROOT:-/data-pool/rfinn/SGA-CIGALE}
PCIGALE=${PCIGALE:-pcigale}
STEM=${CIGALE_STEM:-wisesize_sga2025_ap03_z0002_0025_w3snr10_errfloor0p10mag}

if [[ $# -gt 0 ]]; then
    chunks=("$@")
else
    chunks=(01 02 03)
fi

export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${ROOT}/cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-${ROOT}/cache/matplotlib}
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

if ! command -v "$PCIGALE" >/dev/null 2>&1; then
    echo "pcigale is not available: $PCIGALE" >&2
    exit 1
fi

for chunk in "${chunks[@]}"; do
    run_dir="${ROOT}/cigale_runs/${STEM}_chunk${chunk}"
    if [[ ! -f "${run_dir}/pcigale.ini" ]]; then
        echo "Missing run configuration: ${run_dir}/pcigale.ini" >&2
        exit 1
    fi
    if [[ -e "${run_dir}/out" ]]; then
        echo "Refusing to overwrite existing output: ${run_dir}/out" >&2
        exit 1
    fi

    echo "Starting WISEsize chunk ${chunk}: $(date --iso-8601=seconds)"
    cd "$run_dir"
    "$PCIGALE" check
    /usr/bin/time -v "$PCIGALE" run 2>&1 | tee "run-cores8-chunk${chunk}.log"
    echo "Finished WISEsize chunk ${chunk}: $(date --iso-8601=seconds)"
done
