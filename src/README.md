# Harmony TCP Client-Server Mode

This extends Harmony with a `--serve` mode that allows an external client to
insert vectors and serve queries over a plain TCP socket.  The existing
MPI-based master/worker architecture is completely unchanged.

## Architecture

```
┌─────────────────────┐         TCP socket          ┌──────────────────────────────────────┐
│   harmony_client    │  ──── vectors / queries ──>  │         query (rank 0, master)       │
│   (no MPI needed)   │  <─── distances + labels ──  │                                      │
└─────────────────────┘                              │  MPI_Bcast / MPI_Send / MPI_Recv     │
                                                     │                                      │
                                                     │  query (rank 1..N, workers)          │
                                                     │  GroupWorker::receiveQuery()         │
                                                     │  GroupWorker::search()               │
                                                     └──────────────────────────────────────┘
```

The client is a plain binary with no MPI dependency.  The server is the existing
`query` binary launched with `mpirun` as usual.

## Two-Phase Protocol

### Phase 1 — INSERT (opcode protocol, single TCP connection to `--tcp_port`)

```
Client → Server: [uint8 op=0x01][uint64 n][n*d floats]  →  [uint8 status]   (repeat per batch)
Client → Server: [uint8 op=0x02]                         →  [uint8 status]   (BUILD_DONE)
```

All batches are buffered on the master; `index.add()` is called **once** with
the full dataset on `BUILD_DONE`, then `preSearch()` distributes the index to
workers.

### Phase 2 — QUERY (MasterTcpServer, new TCP connection to same `--tcp_port`)

```
Client → Server: [uint64 nq][uint64 k][nq*d floats]      →  [nq*k floats][nq*k int64]   (repeat)
Client → Server: [uint64 nq=0][uint64 k=0]                                               (shutdown)
```

No opcode prefix in this phase — `MasterTcpServer` reads raw headers.

## Files Changed / Added

| File | Change |
|------|--------|
| `src/query.cpp` | Serve mode, INSERT buffering, MasterTcpServer query loop, MPI signal fixes |
| `src/client.cpp` | New standalone client binary (no MPI) |
| `src/node.cpp` | `GroupWorker::receiveQuery()` — zero distance buffers and reset heaps between queries |
| `src/CMakeLists.txt` | Added `harmony_client` build target |

### Key changes in query.cpp

- Added `--serve`, `--nprobe`, `--tcp_port`, `--skip_insert`, `--insert_port`, `--train_data` arguments
- **INSERT phase**: receives batches via opcode protocol, buffers all vectors, calls `index.add()` once on `OP_BUILD_DONE`, then calls `preSearch()` and broadcasts `WORKER_SIGNAL_INIT` to workers
- **QUERY phase**: `MasterTcpServer` runs on a background thread accepting connections and posting `SearchJob`s to a queue; master thread dequeues jobs, broadcasts `WORKER_SIGNAL_SEARCH`, calls `index.search()`, marks job done
- Worker MPI signal flow:
  - `WORKER_SIGNAL_INIT (2)`: unblocks workers to call `GroupWorker::init()` + `MPI_Barrier`
  - `WORKER_SIGNAL_SEARCH (1)`: triggers `receiveQuery()` + `search()` per query
  - `WORKER_SIGNAL_SHUTDOWN (0)`: clean exit
- Benchmark mode (no `--serve` flag) is **completely unchanged**

## Building

```bash
# 1. Copy files into Harmony
cp src/query.cpp  /path/to/Harmony/src/query.cpp
cp src/client.cpp /path/to/Harmony/src/client.cpp
# Apply node.cpp patch manually (see above)

# 2. Add harmony_client target to src/CMakeLists.txt:
#    add_executable(harmony_client client.cpp)
#    target_link_libraries(harmony_client argparse)

# 3. Build
source /home/csl12/intel/oneapi/setvars.sh
cd /path/to/Harmony
cmake -B release -DCMAKE_BUILD_TYPE=Release .
cmake --build release -j
```

## Generating Groundtruth

Run benchmark mode once to generate the groundtruth file:

```bash
mpirun -n 5 -bind-to none ./release/bin/query \
    --benchmarks_path ./benchmarks --dataset sift1m \
    --k 100 --nprobes 1000 --run_faiss --verbose
```

Saved to `{benchmarks_path}/{dataset}/result/groundtruth_{k}.bin`.

## Running — skip_insert mode (index loaded from disk)

Fastest way to test the query path. Requires a cached index on disk.

**Terminal 1 — server:**
```bash
mpirun -n 5 -bind-to none ./release/bin/query \
    --benchmarks_path ./benchmarks --dataset sift1m \
    --serve --skip_insert --cache \
    --tcp_port 7777 --nprobe 100 \
    --group 2 --team 2 --block 4 --mode group
```

**Terminal 2 — client:**
```bash
./release/bin/harmony_client \
    --query ./benchmarks/sift1m/origin/sift1m_query.fvecs \
    --groundtruth ./benchmarks/sift1m/result/groundtruth_100.bin \
    --host 127.0.0.1 --port 7777 \
    --skip_build --query_batch 96 --k 100 \
    --group 2 --block 4
```

## Running — INSERT then QUERY (two separate clients)

**Terminal 1 — server:**
```bash
mpirun -n 5 -bind-to none ./release/bin/query \
    --benchmarks_path ./benchmarks --dataset sift1m \
    --serve \
    --train_data ./benchmarks/sift1m/origin/sift1m_base.fvecs \
    --tcp_port 7777 --nprobe 100 \
    --group 2 --team 2 --block 4 --mode group
```

**Terminal 2 — INSERT client** (run first, wait for it to finish):
```bash
./release/bin/harmony_client \
    --base ./benchmarks/sift1m/origin/sift1m_base.fvecs \
    --host 127.0.0.1 --port 7777 \
    --insert_batch 50000
```

**Terminal 2 — QUERY client** (run after INSERT client exits):
```bash
./release/bin/harmony_client \
    --query ./benchmarks/sift1m/origin/sift1m_query.fvecs \
    --groundtruth ./benchmarks/sift1m/result/groundtruth_100.bin \
    --host 127.0.0.1 --port 7777 \
    --skip_build --query_batch 96 --k 100 \
    --group 2 --block 4
```

## Running — INSERT and QUERY in one client

```bash
./release/bin/harmony_client \
    --base  ./benchmarks/sift1m/origin/sift1m_base.fvecs \
    --query ./benchmarks/sift1m/origin/sift1m_query.fvecs \
    --groundtruth ./benchmarks/sift1m/result/groundtruth_100.bin \
    --host 127.0.0.1 --port 7777 \
    --insert_batch 50000 --query_batch 96 --k 100 \
    --group 2 --block 4
```

## Client Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--base` | `""` | Path to base `.fvecs` for INSERT phase |
| `--query` | `""` | Path to query `.fvecs` for QUERY phase |
| `--groundtruth` | `""` | Path to groundtruth `.ivecs` or `.bin` for recall |
| `--host` | `127.0.0.1` | Master server IP address |
| `--port` | `7777` | TCP port |
| `--k` | `10` | Number of nearest neighbours |
| `--nq` | `0` (all) | Max query vectors to send |
| `--nb` | `0` (all) | Max base vectors to insert |
| `--insert_batch` | `10000` | Vectors per INSERT batch |
| `--query_batch` | `2000` | Queries per QUERY batch (auto-rounded to multiple of `group*block`) |
| `--skip_build` | off | Skip INSERT phase (server already has index) |
| `--query_loop` | `1` | Repeat query phase N times |
| `--group` | `1` | Server `groupCount` — used to align `query_batch` |
| `--block` | `1` | Server `blockCount` — used to align `query_batch` |

## Server Arguments (new ones only)

| Argument | Default | Description |
|----------|---------|-------------|
| `--serve` | off | Enable TCP serve mode |
| `--nprobe` | `100` | nprobe for all serve-mode searches |
| `--tcp_port` | `7777` | TCP port |
| `--skip_insert` | off | Skip INSERT phase — load index from disk (`--cache` required) |
| `--insert_port` | same as `--tcp_port` | Separate port for INSERT phase (optional) |
| `--train_data` | base file | Seed `.fvecs` for `train()` in INSERT mode |

## nq Divisibility Requirement

In `DIVIDE_GROUP` mode, `nq` per batch must satisfy:

```
query_batch % (groupCount × blockCount) == 0
```

With `--group 2 --block 4` this means multiples of **8** (e.g. 96, 104, 112).
The client automatically rounds `query_batch` down to the nearest valid multiple
when `--group` and `--block` are provided.

## Benchmark Mode (unchanged)

```bash
mpirun -n 5 -bind-to none ./release/bin/query \
    --benchmarks_path ./benchmarks --dataset sift1m \
    --cache --group 2 --team 2 --block 4 --nprobes 100 --k 100
```

Expected recall: ~0.9975.

## Troubleshooting

**Server hangs after `[Master] MPI search: nq=N k=K`**
Workers are out of sync with master's MPI collective sequence. Most likely cause:
the `node.cpp` fix (zeroing `distancesForBlocks`) was not applied, or the
`MPI_Barrier` after `GroupWorker::init()` is missing in the INSERT worker path
of `query.cpp`.

**Low recall (~0.003)**
`Index::add()` resets IVF lists on every call — all vectors must be inserted in
a single `add()` call. The current `query.cpp` buffers all batches and calls
`add()` once on `BUILD_DONE`. If recall is low, verify the server prints
`[Master] index.add() done` (not per-batch add messages).

**`r2: inf` in server output**
Distance buffers were not zeroed between queries. Apply the `node.cpp` fix.

**Client prints `recall@k: 0.000x` but server shows correct recall**
Groundtruth file offset mismatch. Ensure `--groundtruth` points to the correct
`.bin` file generated for the same `k`.

**MPI OFI errors on Intel MPI**
Intel MPI's OFI dynamic process management does not work on this machine.
This implementation uses plain TCP sockets instead, avoiding the issue entirely.