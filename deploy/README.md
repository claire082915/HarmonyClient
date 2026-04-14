# Harmony Deploy Scripts

Scripts for launching, monitoring, and managing Harmony — a distributed approximate nearest neighbour (ANN) search system using MPI and FAISS.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Manual Launch](#manual-launch)
- [Automated Launch via start_services.py](#automated-launch-via-start_servicespy)
- [Monitoring and Logs](#monitoring-and-logs)
- [Running Experiments](#running-experiments)
- [Supported Datasets](#supported-datasets)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

Harmony uses MPI for distributed search across multiple nodes/processes:

```
┌─────────────────┐        TCP (port 7777)       ┌──────────────────┐
│  harmony_client │ ─────────────────────────────▶│   Master (rank 0)│
│  (insert/query) │                               │   MPI rank 0     │
└─────────────────┘                               └────────┬─────────┘
                                                           │ MPI
                                                  ┌────────▼─────────┐
                                                  │  Workers (rank 1+)│
                                                  │  MPI ranks 1..N  │
                                                  └──────────────────┘
```

**Node roles:**
- `master` — MPI rank 0, runs `query` binary with `--serve`, accepts TCP connections from client
- `worker` — MPI ranks 1..N, run `query` binary (worker path via MPI coordination)
- `client` — runs `harmony_client`, connects to master over TCP, sends vectors and queries

**Phase 1 — INSERT:**
Client sends base vectors to master over TCP. Master accumulates vectors, calls `index.add()` on `OP_BUILD_DONE`, then distributes the index to workers via `preSearch()`.

**Phase 2 — QUERY:**
`MasterTcpServer` accepts query batches on a background thread. Master drives MPI search loop, workers execute search, results are sent back to client.

---

## Prerequisites

### Required on all nodes

- Intel oneAPI (MPI + MKL): `/opt/intel/oneapi/setvars.sh`
- GCC 13.2: `~/gcc-13.2/`
- FAISS (built as part of Harmony)
- CMake 3.20+

### Python dependencies (deploy scripts only)

```bash
pip install pyyaml
# Optional: for Azure VM discovery
pip install azure-mgmt-compute azure-mgmt-network azure-identity
```

### SSH setup (required for local simulation)

If running everything on one machine, enable passwordless SSH to localhost:

```bash
ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa      # skip if key exists
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
ssh -i ~/.ssh/id_rsa $(whoami)@127.0.0.1 echo ok   # verify
```

---

## Installation

```bash
git clone https://github.com/claire082915/HarmonyClient ~/Harmony
cd ~/Harmony

# Source oneAPI environment
source /opt/intel/oneapi/setvars.sh --force

# Build
cmake -B release -DCMAKE_BUILD_TYPE=Release
cmake --build release -j$(nproc)

# Verify binaries
ls release/bin/query release/bin/harmony_client
```

---

## Configuration

All settings are in `config.yaml`. Key sections:

```yaml
azure:
  username: csl12                          # SSH username on all nodes
  ssh_private_key: ~/.ssh/id_rsa           # SSH key (use id_rsa for local simulation)

experiment:
  benchmarks_path: /data/csl12/Harmony/benchmarks
  dataset: sift1b                          # dataset name (matches directory name)
  input_format: bvecs                      # fvecs | bvecs | bin
  group: 1                                 # number of query groups (parallelism)
  team: 1                                  # number of worker teams
  block: 1                                 # block count (DIVIDE_DIM mode only)
  nprobe: 100                              # IVF lists to search (recall vs QPS knob)
  nlist: 1000                              # number of IVF clusters (0 = auto sqrt(nb))
  mode: group                              # group | base | block | original | brute
  cache: false                             # load index from disk instead of rebuilding
  skip_insert: false                       # skip INSERT phase (requires cache: true)
  k: 100                                   # number of nearest neighbours
  query_batch: 5000                        # queries per batch
  query_loop: 1                            # repeat query phase N times

serve:
  nb: 1000000                              # number of base vectors to expect
  dim: 128                                 # vector dimension
  train_data: ~/Harmony/benchmarks/sift1m/origin/sift1m_base.fvecs

client:
  base_file: "/data/csl12/Harmony/benchmarks/sift1b/origin/sift1b_base.bvecs"
  query_file: "/data/csl12/Harmony/benchmarks/sift1b/origin/sift1b_query.bvecs"
  groundtruth_file: "/data/csl12/Harmony/benchmarks/sift1b/origin/gnd/idx_50M.ivecs"
  nb: 1000000                              # vectors to insert
  nq: 0                                    # query vectors (0 = all)
  csv_path: "~/Harmony/benchmarks/sift1b/result/log_serve.csv"

mpi:
  hosts_file: ~/mpi_hosts
  omp_num_threads: 1                       # set to 1 when using many MPI ranks
  mpirun_bin: /home/csl12/intel/oneapi/mpi/2021.14/bin/mpirun

nodes:
  - name: horizann-client
    ip: 127.0.0.1
    private_ip: 127.0.0.1
    type: client
  - name: horizann-1
    ip: 127.0.0.1
    private_ip: 127.0.0.1
    type: master
    slots: 1                               # MPI slots on this node
  - name: horizann-2
    ip: 127.0.0.1
    private_ip: 127.0.0.1
    type: worker
    slots: 1
```

### Choosing nlist

`nlist` controls how many IVF clusters the index is divided into. Rule of thumb: `nlist = sqrt(nb)`.

| nb (vectors) | recommended nlist |
|---|---|
| 1M | 1000 |
| 10M | 3162 |
| 50M | 7071 |
| 100M | 10000 |
| 1B | 31623 |

---

## Manual Launch

Use this approach for debugging or when `start_services.py` is not available.

### Step 1 — Write the MPI hostfile

For local simulation (all processes on one machine):
```
# ~/mpi_hosts
127.0.0.1
127.0.0.1
```

One line per MPI rank. For N workers you need N+1 lines (1 master + N workers).

For multi-node:
```
10.0.0.10    # master
10.0.0.11    # worker 1
10.0.0.12    # worker 2
```

### Step 2 — Source oneAPI

```bash
source /opt/intel/oneapi/setvars.sh --force
export OMP_NUM_THREADS=1
```

### Step 3 — Start the server (Terminal 1)

```bash
cd ~/Harmony

mpirun -n 2 \
  -f ~/mpi_hosts \
  -genv LD_LIBRARY_PATH $LD_LIBRARY_PATH \
  -genv OMP_NUM_THREADS 1 \
  ./release/bin/query \
    --benchmarks_path /data/csl12/Harmony/benchmarks \
    --dataset sift1b \
    --input_format bvecs \
    --serve \
    --tcp_port 7777 \
    --nprobe 100 \
    --nlist 1000 \
    --group 1 \
    --team 1 \
    --block 1 \
    --mode group \
    --train_data /data/csl12/Harmony/benchmarks/sift1m/origin/sift1m_base.fvecs
```

Wait until you see:
```
[Master] INSERT server listening on port 7777
[Master] Waiting for insert client...
```

### Step 4 — Start the client (Terminal 2)

**Insert + query with recall:**
```bash
./release/bin/harmony_client \
  --base /data/csl12/Harmony/benchmarks/sift1b/origin/sift1b_base.bvecs \
  --query /data/csl12/Harmony/benchmarks/sift1b/origin/sift1b_query.bvecs \
  --groundtruth /data/csl12/Harmony/benchmarks/sift1b/origin/gnd/idx_50M.ivecs \
  --host 127.0.0.1 \
  --port 7777 \
  --k 100 \
  --nb 1000000 \
  --insert_batch 100000 \
  --query_batch 5000
```

**Insert only (no query):**
```bash
./release/bin/harmony_client \
  --base /data/csl12/Harmony/benchmarks/sift1b/origin/sift1b_base.bvecs \
  --host 127.0.0.1 --port 7777 \
  --nb 1000000 \
  --insert_batch 100000
```

**Query only (index already built with --cache):**
```bash
./release/bin/harmony_client \
  --query /data/csl12/Harmony/benchmarks/sift1b/origin/sift1b_query.bvecs \
  --groundtruth /data/csl12/Harmony/benchmarks/sift1b/origin/gnd/idx_50M.ivecs \
  --host 127.0.0.1 --port 7777 \
  --k 100 --skip_build \
  --query_batch 5000
```

### Varying nprobe for recall/QPS sweep

`--nprobes` accepts multiple values in benchmark (non-serve) mode:

```bash
mpirun -n 5 ./release/bin/query \
  --benchmarks_path ./benchmarks \
  --dataset sift1m \
  --cache \
  --nprobes 10 50 100 200 \
  --group 2 --team 2 --block 4 \
  --k 100 --mode group
```

---

## Automated Launch via start_services.py

### Start all services

```bash
python start_services.py --config config.yaml --start
```

This will:
1. Verify binaries on master and client nodes
2. Write the MPI hostfile to the master node
3. Launch `mpirun` on master (spanning all worker ranks)
4. Wait until port 7777 is open (handles long index training time)
5. Launch `harmony_client` on the client node

### Stop all services

```bash
python start_services.py --config config.yaml --stop
```

### Restart

```bash
python start_services.py --config config.yaml --restart
```

### Check status

```bash
python start_services.py --config config.yaml --status
```

### Override config values without editing the file

```bash
# Change nprobe for a single run
python start_services.py --config config.yaml --start \
  --set experiment.nprobe=200

# Change number of vectors
python start_services.py --config config.yaml --start \
  --set client.nb=50000000 \
  --set serve.nb=50000000
```

---

## Monitoring and Logs

### View logs via script

```bash
# Server logs (master + workers)
python start_services.py --config config.yaml --logs horizann-1

# Client logs
python start_services.py --config config.yaml --logs horizann-client

# More lines
python start_services.py --config config.yaml --logs horizann-1 --lines 200
```

### View logs directly

```bash
# Follow server log in real time
tail -f ~/Harmony/logs/server_*.log

# Follow client log
tail -f ~/Harmony/logs/client_*.log

# Most recent log
tail -f $(ls -1t ~/Harmony/logs/*.log | head -1)
```

### Download logs

```bash
# Download logs for a specific run timestamp
python start_services.py --config config.yaml --download-logs 20260413_105701

# Download to custom directory
python start_services.py --config config.yaml \
  --download-logs ~/my_results 20260413_105701
```

Downloaded logs are saved to `<local_logs_dir>/<dataset>_<nworkers>w_<timestamp>/` and an experiment record is appended to `experiments.txt` for later reference.

---

## Running Experiments

The four scaling experiments insert increasing numbers of vectors with increasing worker counts.

### Experiment 1 — 1 worker, 50M vectors

```yaml
# config.yaml
cluster:
  num_workers: 1
serve:
  nb: 50000000
client:
  nb: 50000000
  groundtruth_file: "/data/csl12/Harmony/benchmarks/sift1b/origin/gnd/idx_50M.ivecs"
nodes:
  - {name: horizann-1, ip: 127.0.0.1, private_ip: 127.0.0.1, type: master, slots: 1}
  - {name: horizann-2, ip: 127.0.0.1, private_ip: 127.0.0.1, type: worker, slots: 1}
```

### Experiment 2 — 2 workers, 100M vectors

Add a second worker node and update nb:
```yaml
cluster:
  num_workers: 2
serve:
  nb: 100000000
client:
  nb: 100000000
nodes:
  - {name: horizann-1, ..., type: master, slots: 1}
  - {name: horizann-2, ..., type: worker, slots: 1}
  - {name: horizann-3, ..., type: worker, slots: 1}
```

### Experiment 3 — 3 workers, 150M vectors

```yaml
cluster:
  num_workers: 3
serve:
  nb: 150000000
client:
  nb: 150000000
```

### Experiment 4 — 4 workers, 200M vectors

```yaml
cluster:
  num_workers: 4
serve:
  nb: 200000000
client:
  nb: 200000000
```

---

## Supported Datasets

### SIFT1B (bvecs format)

```
/data/csl12/Harmony/benchmarks/sift1b/origin/
  sift1b_base.bvecs       # 1B vectors, 128-dim, uint8
  sift1b_query.bvecs      # 10K query vectors
  sift1b_learn.bvecs      # 100M training vectors
  gnd/idx_50M.ivecs       # groundtruth for 50M subset
```

Config:
```yaml
experiment:
  dataset: sift1b
  input_format: bvecs
```

### SIFT1M (fvecs format)

```
benchmarks/sift1m/origin/
  sift1m_base.fvecs       # 1M vectors, 128-dim, float32
  sift1m_query.fvecs
```

Config:
```yaml
experiment:
  dataset: sift1m
  input_format: fvecs
```

### SPACEV1B (sharded bin format)

```
/data/anns_data/SPTAG/datasets/SPACEV1B/
  vectors.bin/            # directory of 33 shards, 1.4B vectors, 100-dim, int8
    vectors_1.bin         # header: [int32 count][int32 dim], then raw int8
    vectors_2.bin ... vectors_33.bin   # raw int8, no header
  query.bin               # 29,316 query vectors
  truth.bin               # groundtruth: [int32 count][int32 topk][int32 ids][float dists]
```

Config:
```yaml
experiment:
  dataset: spacev1b
  input_format: bin
client:
  base_file: "/data/anns_data/SPTAG/datasets/SPACEV1B/vectors.bin"
  query_file: "/data/anns_data/SPTAG/datasets/SPACEV1B/query.bin"
  groundtruth_file: "/data/anns_data/SPTAG/datasets/SPACEV1B/truth.bin"
```

---

## Troubleshooting

### `libmkl_intel_lp64.so.2: cannot open shared object file`

oneAPI libs not in `LD_LIBRARY_PATH`. The deploy scripts pass these via `-genv` automatically. For manual runs, source oneAPI first:
```bash
source /opt/intel/oneapi/setvars.sh --force
```

### `MPI tag exceeds MPI_TAG_UB`

Occurs in `--mode block` with large query counts. Use `--mode group` instead.

### `Unknown INSERT-phase opcode`

Client crashed before sending `OP_BUILD_DONE`. Check client log for the real error — usually a file not found or connection refused before the server was ready.

### `connect() failed — is the server running with --serve?`

Either the server isn't running or it hasn't opened the TCP port yet (still training). The deploy script waits for port 7777 automatically. For manual runs, wait until you see `INSERT server listening on port 7777` before starting the client.

### `Inconsistent dim` error in client

The base file is being read with the wrong loader. Ensure `--base` points to the correct file and the extension matches the format (`.bvecs` for uint8 bvecs, `.fvecs` for float32 fvecs).

### Server exits immediately with exit code 1

Check `~/Harmony/logs/server_*.log`. Common causes:
- Wrong `--dataset` name (directory doesn't exist under `benchmarks_path`)
- Missing `--train_data` file
- `--nlist` too large for the number of inserted vectors