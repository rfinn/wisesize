# wisesize

Reusable scripts and notes for SGA/WiseSize analysis.

## SGA-2025 AP03 CIGALE preprocessing

Code and notes:

- `scripts/preprocess_sga2025_ap03_for_cigale.py`
- `scripts/export_speclite_legacy_filters_for_cigale.py`
- `scripts/make_cigale_test_sample.py`
- `scripts/make_conger_overlap_cigale_sample.py`
- `scripts/make_wisesize_cigale_sample.py`
- `scripts/make_cigale_slurm_benchmark.py`
- `configs/sga2025_ap03_filter_map.json`
- `docs/sga2025_ap03_cigale_preprocessing.md`

Generated products are written outside this git repository, under:

```text
/Users/rfinn/research/SGA-CIGALE
```

## WISEsize CIGALE sample

`make_wisesize_cigale_sample.py` applies the WISEsize redshift and W3 S/N
selection, retains `INSTAR` sources by default, sorts the result by redshift,
and writes 10,000-object CIGALE chunks plus an audit table and report. It also
creates portable run directories using relative input paths.

On Draco, run all prepared chunks sequentially with:

```bash
conda activate cigale
bash ~/github/wisesize/scripts/run_wisesize_cigale_draco.sh
```

Pass one or more chunk numbers to run only those chunks, for example `01` or
`02 03`. The launcher refuses to overwrite an existing `out/` directory.

After all chunks finish, validate and combine their CIGALE catalogs with:

```bash
python ~/github/wisesize/scripts/collect_cigale_chunk_results.py
```

This writes
`/mnt/astro/SGA-CIGALE/results/wisesize_sga2025_ap03_z0002_0025_w3snr10_results.fits`.
The combined table includes an explicit `SGAID` cross-match key and a
`CIGALE_CHUNK` provenance column. The collector checks each result ID against
the corresponding CIGALE input and stops on missing, unexpected, or duplicate
IDs. To collect one completed chunk independently, pass `--chunks 1`.

## CIGALE SLURM benchmark

`make_cigale_slurm_benchmark.py` creates separate CIGALE configurations for
each requested core count and one SLURM job that runs them sequentially on the
same node. The generated job requests an exclusive node by default so CPU and
memory-bandwidth contention from unrelated jobs does not distort the timing.

Example on Siena's cluster:

```bash
python ~/github/wisesize/scripts/make_cigale_slurm_benchmark.py \
  --template-config /mnt/astrophysics/rfinn/SGA-CIGALE/cigale_runs/sga2025_ap03_z0025_0040_singlez003_pruned500/pcigale.ini \
  --data-file /mnt/astrophysics/rfinn/SGA-CIGALE/inputs/sga2025_ap03_z0025_0040_cigale500.dat \
  --output-dir /mnt/astrophysics/rfinn/SGA-CIGALE/slurm_benchmarks/siena_single_job_scaling \
  --cores 4 8 16 \
  --partition normal
```

The generated command is printed without being submitted. Add `--submit` to
submit immediately. By default the compute-node executable is
`~/.conda/envs/cigale/bin/pcigale`; override it with `--pcigale` if Siena uses
a different environment. Use `--module Python3` only when the CIGALE
installation depends on that environment module.
