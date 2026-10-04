#!/usr/bin/env bash

# Prepare matched solo and concurrent CIGALE benchmarks on Draco local storage.

set -euo pipefail

ROOT=${SGA_CIGALE_ROOT:-/data-pool/rfinn/SGA-CIGALE}
PYTHON=${PYTHON:-python}
N_OBJECTS=${N_OBJECTS:-2000}
CORES=${CIGALE_CORES:-8}
SOURCE_STEM=${CIGALE_STEM:-wisesize_sga2025_ap04_z0002_0025_w3snr10_errfloor0p10mag_agnfrac0to0p5}
BENCH_STEM=${CIGALE_BENCH_STEM:-wisesize_agnfrac0to0p5_bench${N_OBJECTS}}
PREP_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/prepare_cigale_subset_benchmark.py"

prepare_run() {
    local source_chunk=$1
    local suffix=$2

    "$PYTHON" "$PREP_SCRIPT" \
        --source-root "$ROOT" \
        --destination-root "$ROOT" \
        --source-run "${SOURCE_STEM}_chunk${source_chunk}" \
        --n-objects "$N_OBJECTS" \
        --cores "$CORES" \
        --name "${BENCH_STEM}_${suffix}"
}

# The solo and concurrent versions of each letter use the same source rows.
prepare_run 01 solo_a
prepare_run 02 solo_b
prepare_run 01 concurrent_a
prepare_run 02 concurrent_b

printf '\nPrepared four %s-object, %s-core benchmarks under %s/cigale_runs.\n' \
    "$N_OBJECTS" "$CORES" "$ROOT"
printf 'Run the baseline first:\n'
printf '  bash scripts/run_cigale_concurrency_benchmark_draco.sh sequential %s_solo_a %s_solo_b\n' \
    "$BENCH_STEM" "$BENCH_STEM"
printf 'Then run the concurrent pair:\n'
printf '  bash scripts/run_cigale_concurrency_benchmark_draco.sh concurrent %s_concurrent_a %s_concurrent_b\n' \
    "$BENCH_STEM" "$BENCH_STEM"
