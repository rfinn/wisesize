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
adds a 0.1 mag systematic uncertainty in quadrature to the CIGALE fitting
errors, and writes 10,000-object chunks plus an audit table and report. The W3
selection continues to use the original catalog errors. The script also creates
portable run directories using relative input paths and gives error-floor runs
a distinct `errfloor0p10mag` name so earlier outputs are preserved.

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
`/data-pool/rfinn/SGA-CIGALE/results/wisesize_sga2025_ap03_z0002_0025_w3snr10_errfloor0p10mag_results.fits`.
The combined table includes an explicit `SGAID` cross-match key and a
`CIGALE_CHUNK` provenance column. The collector checks each result ID against
the corresponding CIGALE input and stops on missing, unexpected, or duplicate
IDs. To collect one completed chunk independently, pass `--chunks 1`.

To test the first 500 objects from chunk 1 using Draco's local ZFS storage,
prepare and run an isolated benchmark with:

```bash
python ~/github/wisesize/scripts/prepare_cigale_subset_benchmark.py
/data-pool/rfinn/SGA-CIGALE/cigale_runs/\
wisesize_sga2025_ap03_z0002_0025_w3snr10_chunk01_first500_localdisk/\
run-benchmark.sh
```

The benchmark copies only the configuration and selected input rows from
`/mnt/astro`; its cache, run files, log, and results remain on `/data-pool`.
Use `--config-run` and `--input-file` to pair a configuration with a different
input table for controlled timing comparisons. The generated configuration
uses eight CIGALE workers unless `--cores` is specified.

## NED-LVS comparison

Compare the combined WISEsize CIGALE results with NED-LVS stellar masses and
SFRs using a nearest-neighbor sky match with a 30 arcsec limit:

```bash
python scripts/compare_wisesize_cigale_nedlvs.py
```

The script uses SGA redshifts as a secondary match diagnostic, writes a
left-join FITS table with match and redshift-quality flags, and produces
summary tables and plots under
`/Users/rfinn/research/SGA-CIGALE/comparisons/wisesize_nedlvs_20250602`.

Cross-check the WISEsize CIGALE results against the row-aligned Virgo Filament
CIGALE metallicity catalog with:

```bash
python scripts/compare_wisesize_cigale_vf.py
```

This uses `vf_v2_main.fits` for coordinates and radial velocities, applies the
same 30 arcsec positional limit, and uses `vr/c` as a secondary redshift
confirmation. Bayesian and best-fit stellar masses and SFRs are the primary
comparison, with other shared CIGALE parameters retained as secondary checks.
Outputs are written under
`/Users/rfinn/research/SGA-CIGALE/comparisons/wisesize_vf_cigale_metallicity_20260305`.

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
