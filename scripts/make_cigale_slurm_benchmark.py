#!/usr/bin/env python3
"""Generate a same-node SLURM scaling benchmark for CIGALE.

The generated batch job runs the requested core counts sequentially on one
exclusive node. This keeps the input table, model grid, node, and competing
workload fixed while measuring CIGALE's multiprocessing scaling.
"""

from __future__ import annotations

import argparse
import re
import shlex
import shutil
import subprocess
from pathlib import Path


DEFAULT_CORES = (4, 8, 16)
DEFAULT_PCIGALE = Path.home() / ".conda" / "envs" / "cigale" / "bin" / "pcigale"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create CIGALE run directories and a SLURM script for a clean "
            "single-node core-scaling benchmark."
        )
    )
    parser.add_argument(
        "--template-config",
        type=Path,
        required=True,
        help="CIGALE pcigale.ini to use as the benchmark template.",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        required=True,
        help="CIGALE input table visible from the compute node.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which to create the benchmark bundle and run directories.",
    )
    parser.add_argument(
        "--cores",
        type=int,
        nargs="+",
        default=list(DEFAULT_CORES),
        help="Core counts to benchmark sequentially. Default: 4 8 16.",
    )
    parser.add_argument(
        "--pcigale",
        default=str(DEFAULT_PCIGALE),
        help=f"pcigale executable on the compute node. Default: {DEFAULT_PCIGALE}",
    )
    parser.add_argument(
        "--partition",
        default="normal",
        help="SLURM partition. Default: normal.",
    )
    parser.add_argument(
        "--time",
        default="02:00:00",
        help="SLURM wall-time request. Default: 02:00:00.",
    )
    parser.add_argument(
        "--job-name",
        default="cigale-benchmark",
        help="SLURM job name. Default: cigale-benchmark.",
    )
    parser.add_argument(
        "--module",
        action="append",
        default=[],
        help="Environment module to load in the batch job; may be repeated.",
    )
    parser.add_argument("--account", help="Optional SLURM account.")
    parser.add_argument("--qos", help="Optional SLURM quality of service.")
    parser.add_argument("--constraint", help="Optional SLURM node constraint.")
    parser.add_argument("--mem", help="Optional SLURM memory request, e.g. 64G.")
    parser.add_argument(
        "--shared-node",
        action="store_true",
        help=(
            "Allow other jobs on the benchmark node. By default the generated "
            "job requests the node exclusively for reproducible timing."
        ),
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit the generated script with sbatch after writing it.",
    )
    return parser.parse_args()


def replace_setting(
    lines: list[str], name: str, value: str, section: str | None
) -> list[str]:
    """Replace one ConfigObj setting, optionally within a top-level section."""
    current_section: str | None = None
    replaced = 0
    setting_pattern = re.compile(rf"^(\s*){re.escape(name)}\s*=.*$")
    section_pattern = re.compile(r"^\s*\[([^\[\]]+)\]\s*$")

    output: list[str] = []
    for line in lines:
        section_match = section_pattern.match(line)
        if section_match:
            current_section = section_match.group(1).strip()

        in_target = current_section == section if section is not None else current_section is None
        setting_match = setting_pattern.match(line)
        if in_target and setting_match:
            output.append(f"{setting_match.group(1)}{name} = {value}\n")
            replaced += 1
        else:
            output.append(line)

    location = "top level" if section is None else f"[{section}]"
    if replaced != 1:
        raise ValueError(
            f"Expected exactly one {name!r} setting in {location}; found {replaced}."
        )
    return output


def write_config(
    template: Path, destination: Path, data_file: Path, cores: int
) -> None:
    lines = template.read_text(encoding="utf-8").splitlines(keepends=True)
    lines = replace_setting(lines, "data_file", str(data_file), section=None)
    lines = replace_setting(lines, "cores", str(cores), section=None)
    lines = replace_setting(lines, "save_best_sed", "False", section="analysis_params")
    lines = replace_setting(lines, "save_chi2", "none", section="analysis_params")
    destination.write_text("".join(lines), encoding="utf-8")


def sbatch_directives(args: argparse.Namespace, max_cores: int) -> list[str]:
    output_path = args.output_dir / "slurm-%j.out"
    error_path = args.output_dir / "slurm-%j.err"
    directives = [
        f"#SBATCH --job-name={args.job_name}",
        f"#SBATCH --output={output_path}",
        f"#SBATCH --error={error_path}",
        f"#SBATCH --partition={args.partition}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={max_cores}",
        f"#SBATCH --time={args.time}",
    ]
    if not args.shared_node:
        directives.append("#SBATCH --exclusive")
    if args.account:
        directives.append(f"#SBATCH --account={args.account}")
    if args.qos:
        directives.append(f"#SBATCH --qos={args.qos}")
    if args.constraint:
        directives.append(f"#SBATCH --constraint={args.constraint}")
    if args.mem:
        directives.append(f"#SBATCH --mem={args.mem}")
    return directives


def shell_quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def write_slurm_script(args: argparse.Namespace, run_dirs: dict[int, Path]) -> Path:
    script_path = args.output_dir / "run_cigale_scaling.sbatch"
    max_cores = max(run_dirs)
    lines = ["#!/bin/bash -l", *sbatch_directives(args, max_cores), "", "set -euo pipefail"]

    lines.extend(
        [
            "",
            "if type module >/dev/null 2>&1; then",
            "    module purge",
            "fi",
        ]
    )
    for module in args.module:
        lines.extend(
            [
                "if ! type module >/dev/null 2>&1; then",
                f"    echo 'Cannot load module {module}: module command is unavailable.' >&2",
                "    exit 1",
                "fi",
                f"module load {shell_quote(module)}",
            ]
        )

    lines.extend(
        [
            "",
            f"CIGALE_EXE={shell_quote(args.pcigale)}",
            'if [[ "$CIGALE_EXE" == */* ]]; then',
            '    if [[ ! -x "$CIGALE_EXE" ]]; then',
            '        echo "pcigale executable is missing or not executable: $CIGALE_EXE" >&2',
            "        exit 1",
            "    fi",
            "elif ! command -v \"$CIGALE_EXE\" >/dev/null 2>&1; then",
            '    echo "pcigale command is not available: $CIGALE_EXE" >&2',
            "    exit 1",
            "fi",
            "",
            'export TMPDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"',
            "export OMP_NUM_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "export VECLIB_MAXIMUM_THREADS=1",
            "export NUMEXPR_NUM_THREADS=1",
            "",
            'echo "Benchmark started: $(date --iso-8601=seconds)"',
            'echo "Host: $(hostname)"',
            'echo "SLURM job: ${SLURM_JOB_ID:-unset}"',
            'echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK:-unset}"',
            'echo "TMPDIR: $TMPDIR"',
            'echo "pcigale: $CIGALE_EXE"',
            '"$CIGALE_EXE" --version || true',
            "lscpu",
            "free -h",
            "numactl --hardware 2>/dev/null || true",
            'df -h "$TMPDIR"',
            "",
            "run_benchmark() {",
            "    local cores=$1",
            "    local run_dir=$2",
            '    local log_file="run-cores${cores}-slurm-${SLURM_JOB_ID}.log"',
            "",
            '    echo "============================================================"',
            '    echo "Starting ${cores}-core benchmark: $(date --iso-8601=seconds)"',
            '    echo "Run directory: $run_dir"',
            '    if [[ -e "$run_dir/out" ]]; then',
            '        echo "Refusing to overwrite existing output: $run_dir/out" >&2',
            "        exit 1",
            "    fi",
            '    cd "$run_dir"',
            '    grep -E "^(data_file|cores)[[:space:]]*=" pcigale.ini',
            '    "$CIGALE_EXE" check',
            '    /usr/bin/time -v "$CIGALE_EXE" run 2>&1 | tee "$log_file"',
            '    echo "Finished ${cores}-core benchmark: $(date --iso-8601=seconds)"',
            "}",
            "",
        ]
    )

    for cores, run_dir in run_dirs.items():
        lines.append(f"run_benchmark {cores} {shell_quote(run_dir)}")

    lines.extend(["", 'echo "Benchmark completed: $(date --iso-8601=seconds)"', ""])
    script_path.write_text("\n".join(lines), encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def main() -> None:
    args = parse_args()
    args.template_config = args.template_config.expanduser().resolve()
    args.data_file = args.data_file.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    if not args.template_config.is_file():
        raise FileNotFoundError(f"Template configuration not found: {args.template_config}")
    template_spec = args.template_config.with_suffix(".ini.spec")
    if not template_spec.is_file():
        raise FileNotFoundError(
            "CIGALE configuration specification not found next to the template: "
            f"{template_spec}"
        )
    if not args.data_file.is_file():
        raise FileNotFoundError(f"CIGALE input table not found: {args.data_file}")
    if not args.cores or any(cores < 1 for cores in args.cores):
        raise ValueError("All --cores values must be positive integers.")
    if len(set(args.cores)) != len(args.cores):
        raise ValueError("--cores values must be unique.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs: dict[int, Path] = {}
    for cores in args.cores:
        run_dir = args.output_dir / f"cores{cores}"
        if (run_dir / "out").exists():
            raise FileExistsError(
                f"Existing CIGALE output would be overwritten: {run_dir / 'out'}"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        write_config(
            args.template_config,
            run_dir / "pcigale.ini",
            args.data_file,
            cores,
        )
        shutil.copy2(template_spec, run_dir / "pcigale.ini.spec")
        run_dirs[cores] = run_dir

    script_path = write_slurm_script(args, run_dirs)
    print(f"Wrote benchmark bundle: {args.output_dir}")
    for cores, run_dir in run_dirs.items():
        print(f"  {cores:>2} cores: {run_dir / 'pcigale.ini'}")
    print(f"SLURM script: {script_path}")

    if args.submit:
        completed = subprocess.run(
            ["sbatch", str(script_path)], check=True, text=True, capture_output=True
        )
        print(completed.stdout.strip())
    else:
        print(f"Submit with: sbatch {script_path}")


if __name__ == "__main__":
    main()
