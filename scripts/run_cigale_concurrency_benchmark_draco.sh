#!/usr/bin/env bash

# Run two prepared CIGALE benchmarks sequentially or concurrently on Draco.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_cigale_concurrency_benchmark_draco.sh MODE RUN_A RUN_B

MODE must be "sequential" or "concurrent". RUN_A and RUN_B are run-directory
names under $SGA_CIGALE_ROOT/cigale_runs. System measurements are written under
$SGA_CIGALE_ROOT/benchmarks.
EOF
}

if [[ $# -ne 3 ]]; then
    usage >&2
    exit 2
fi

MODE=$1
RUN_A=$2
RUN_B=$3
if [[ "$MODE" != sequential && "$MODE" != concurrent ]]; then
    usage >&2
    exit 2
fi

ROOT=${SGA_CIGALE_ROOT:-/data-pool/rfinn/SGA-CIGALE}
RUN_ROOT="$ROOT/cigale_runs"
STAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="$ROOT/benchmarks/${STAMP}_${MODE}"
mkdir -p "$REPORT_DIR"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

for run_name in "$RUN_A" "$RUN_B"; do
    run_dir="$RUN_ROOT/$run_name"
    if [[ ! -x "$run_dir/run-benchmark.sh" ]]; then
        echo "Missing benchmark launcher: $run_dir/run-benchmark.sh" >&2
        exit 1
    fi
    if [[ -e "$run_dir/out" ]]; then
        echo "Refusing to overwrite existing output: $run_dir/out" >&2
        exit 1
    fi
done

ACTIVE_FILE="$REPORT_DIR/.monitor-active"
touch "$ACTIVE_FILE"

monitor_system() {
    echo "timestamp,epoch,mem_available_kib,swap_used_kib,pcigale_processes,pcigale_rss_kib,pcigale_cpu_percent,pswpin_pages,pswpout_pages,load1"
    while [[ -e "$ACTIVE_FILE" ]]; do
        read -r mem_available swap_used < <(
            free -k | awk '
                /^Mem:/ {available=$7}
                /^Swap:/ {swap_used=$3}
                END {print available+0, swap_used+0}'
        )
        read -r process_count rss_kib cpu_percent < <(
            ps -u "$USER" -o rss=,pcpu=,args= | awk '
                /[p]cigale run/ {count += 1; rss += $1; cpu += $2}
                END {printf "%d %.0f %.1f\n", count+0, rss+0, cpu+0}'
        )
        read -r pswpin pswpout < <(
            awk '
                /^pswpin / {input=$2}
                /^pswpout / {output=$2}
                END {print input+0, output+0}' /proc/vmstat
        )
        load1=$(awk '{print $1}' /proc/loadavg)
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$(date --iso-8601=seconds)" "$(date +%s)" \
            "$mem_available" "$swap_used" "$process_count" "$rss_kib" \
            "$cpu_percent" "$pswpin" "$pswpout" "$load1"
        sleep 5
    done
}

monitor_system > "$REPORT_DIR/system-monitor.csv" &
MONITOR_PID=$!

stop_monitor() {
    rm -f "$ACTIVE_FILE"
    wait "$MONITOR_PID" 2>/dev/null || true
}
trap stop_monitor EXIT

run_one() {
    local label=$1
    local run_name=$2
    local run_dir="$RUN_ROOT/$run_name"
    local start end status

    start=$(date +%s)
    echo "Starting $label ($run_name): $(date --iso-8601=seconds)"
    set +e
    "$run_dir/run-benchmark.sh" > "$REPORT_DIR/${label}-${run_name}.log" 2>&1
    status=$?
    set -e
    end=$(date +%s)
    printf '%s,%s,%s,%s,%s\n' \
        "$label" "$run_name" "$start" "$end" "$status" \
        > "$REPORT_DIR/${label}-status.csv"
    echo "Finished $label ($run_name): $(date --iso-8601=seconds), status=$status"
    return "$status"
}

OVERALL_START=$(date +%s)
STATUS=0
if [[ "$MODE" == sequential ]]; then
    run_one A "$RUN_A" || STATUS=1
    run_one B "$RUN_B" || STATUS=1
else
    run_one A "$RUN_A" &
    PID_A=$!
    run_one B "$RUN_B" &
    PID_B=$!
    wait "$PID_A" || STATUS=1
    wait "$PID_B" || STATUS=1
fi
OVERALL_END=$(date +%s)

stop_monitor
trap - EXIT

{
    echo "mode,overall_start_epoch,overall_end_epoch,overall_elapsed_seconds,status"
    echo "$MODE,$OVERALL_START,$OVERALL_END,$((OVERALL_END - OVERALL_START)),$STATUS"
} > "$REPORT_DIR/summary.csv"

{
    echo "label,run_name,start_epoch,end_epoch,status"
    tail -n 1 "$REPORT_DIR/A-status.csv"
    tail -n 1 "$REPORT_DIR/B-status.csv"
} > "$REPORT_DIR/jobs.csv"

echo "Benchmark report: $REPORT_DIR"
exit "$STATUS"
