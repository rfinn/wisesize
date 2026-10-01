**Draco System Cheat Sheet**

### Overall State

```bash
uptime
```

Shows uptime, logged-in users, and load averages. Draco has 48 logical CPUs, so a load of `8` means roughly eight runnable processes.

```bash
free -h
swapon --show
```

Shows RAM and swap usage. Occupied swap is not necessarily a problem unless swapping is currently active.

### Live CPU, Memory, and Swap

```bash
vmstat 1 10
```

Important columns:

- `r`: runnable processes
- `si`, `so`: swap input/output; sustained nonzero values are concerning
- `us`, `sy`: user and system CPU percentages
- `id`: idle CPU percentage
- `wa`: CPU waiting for I/O

For continuous monitoring during CIGALE:

```bash
vmstat 5 | tee /data-pool/rfinn/SGA-CIGALE/vmstat-cigare.log
```

Stop with `Ctrl-C`.

### Largest Processes

Sort by CPU:

```bash
ps -eo user,pid,psr,pcpu,pmem,rss,etime,comm \
  --sort=-pcpu | head -30
```

Sort by memory:

```bash
ps -eo user,pid,psr,pcpu,pmem,rss,etime,comm \
  --sort=-rss | head -30
```

- `PSR`: most recently used logical CPU
- `RSS`: resident memory in KiB
- `ETIME`: process age
- `%CPU = 800` means approximately eight logical CPUs

Inspect one process:

```bash
ps -fp PID
pstree -sp PID
tr '\0' ' ' < /proc/PID/cmdline
echo
```

Only terminate processes you own or are authorized to manage:

```bash
kill PID
```

### NUMA Layout

```bash
lscpu | grep -E \
'Model name|Socket|Core|Thread|NUMA|CPU max MHz|CPU min MHz'
```

```bash
numactl --hardware
```

Shows CPUs and free memory associated with each NUMA node.

Inspect a process’s memory distribution:

```bash
sudo numastat -p PID
```

Run a command on specific physical CPUs and memory node:

```bash
numactl --physcpubind=0-5 --membind=0 COMMAND
```

### Filesystem Type

```bash
findmnt -T /mnt/astro -o TARGET,SOURCE,FSTYPE,OPTIONS
findmnt -T /data-pool -o TARGET,SOURCE,FSTYPE,OPTIONS
df -h /mnt/astro /data-pool
```

On Draco:

- `/mnt/astro`: shared NFS storage
- `/data-pool`: local ZFS storage

### Python Numerical Threads

```bash
python - <<'PY'
import numpy
from threadpoolctl import threadpool_info

for pool in threadpool_info():
    print(pool)
PY
```

Temporarily restrict numerical libraries to one thread:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Return to the environment defaults:

```bash
unset OPENBLAS_NUM_THREADS OMP_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS
```

### Screen Sessions

```bash
screen -S cigale_wisesize
```

Detach: `Ctrl-A`, then `D`.

```bash
screen -list
screen -r cigale_wisesize
screen -S cigale_wisesize -X quit
```

### Extract CIGALE Timing

```bash
grep -E \
"Start:|Computing models|Model .*100%|Object .*100%|End:|Total duration|Elapsed.*wall|Percent of CPU|Maximum resident|Major.*page|Minor.*page|context switches|Swaps" \
run-cores8-chunk01.log
```
