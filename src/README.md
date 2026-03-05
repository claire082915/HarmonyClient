# Harmony TCP Client-Server Mode

This extends Harmony with a `--serve` mode that allows an external client to send
query vectors over a plain TCP socket and receive nearest-neighbour results back.
The existing MPI-based master/worker architecture is completely unchanged.

## Architecture

```
┌─────────────────────┐         TCP socket          ┌──────────────────────────────────────┐
│   harmony_client    │  ──── queries (floats) ───>  │         query (rank 0, master)       │
│   (no MPI needed)   │  <─── distances + labels ──  │                                      │
└─────────────────────┘                              │  MPI_Bcast / MPI_Send / MPI_Recv     │
                                                     │                                      │
                                                     │  query (rank 1..N, workers)          │
                                                     │  GroupWorker::receiveQuery()         │
                                                     │  GroupWorker::search()               │
                                                     └──────────────────────────────────────┘
```

The client is a plain binary with no MPI dependency. The server is the existing
`query` binary launched with `mpirun` as usual. Communication between client and
server uses a TCP socket on a configurable port.

## Files Changed / Added

| File | Change |
|------|--------|
| `src/query.cpp` | Added `--serve`, `--nprobe`, `--tcp_port` flags and serve loop |
| `src/client.cpp` | New standalone client binary (no MPI) |
| `src/CMakeLists.txt` | Added `harmony_client` build target |

### What changed in query.cpp

- Added three new arguments: `--serve`, `--nprobe <int>`, `--tcp_port <int>`
- In serve mode, rank 0 (master) opens a TCP socket, accepts one client, then enters
  a loop: receive queries → signal workers → `MPI_Barrier` → `index.search()` →
  compute recall → send results back over TCP
- Workers loop on `MPI_Bcast` signal: `1` = run `receiveQuery()` + `search()`,
  `0` = shutdown
- Benchmark mode (no `--serve` flag) is **completely unchanged**
- `preSearch()` is called once before the loop — index is distributed to workers
  only once regardless of how many queries are served

## Building

```bash
# 1. Copy files into Harmony
cp src/query.cpp  /path/to/Harmony/src/query.cpp
cp src/client.cpp /path/to/Harmony/src/client.cpp

# 2. Add harmony_client target to src/CMakeLists.txt
#    (add_executable(harmony_client client.cpp) with no MPI linkage)

# 3. Build with Intel MPI (oneAPI must be sourced)
source /home/csl12/intel/oneapi/setvars.sh
cd /path/to/Harmony
rm -rf release/
cmake -B release -DCMAKE_BUILD_TYPE=Release .
cmake --build release -j
```

## Generating Groundtruth

Groundtruth is generated automatically the first time benchmark mode runs.
It uses FAISS with `nprobe = nlist` (exhaustive) and saves to
`{benchmarks_path}/{dataset}/result/groundtruth_{k}.bin`.

```bash
# Generate groundtruth_100.bin for sift1m
mpirun -n 5 ./release/bin/query \
    --benchmarks_path ./benchmarks --dataset sift1m \
    --cache --group=2 --team=2 --block=4 --nprobes 100 --k 100
```

Only needs to be run once. Subsequent runs (benchmark or serve) load it from disk.

## Running Serve Mode

**Step 1 — Terminal 1: start the server**

Wait until you see `[Master] Waiting for client...` before starting the client.

```bash
mpirun -n 5 ./release/bin/query \
    --benchmarks_path ./benchmarks --dataset sift1m \
    --cache --group=2 --team=2 --block=4 \
    --nprobe 100 --serve --tcp_port 7777
```

**Step 2 — Terminal 2: run the client**

`nq` must be divisible by `groupCount × blockCount` (here 2 × 4 = 8).

```bash
./release/bin/harmony_client \
    --query ./benchmarks/sift1m/origin/sift1m_query.fvecs \
    --host 127.0.0.1 --port 7777 --k 100 --nq 96
```

### Client Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--query` | required | Path to query `.fvecs` file |
| `--host` | `127.0.0.1` | Master server IP address |
| `--port` | `7777` | TCP port |
| `--k` | `10` | Number of nearest neighbours |
| `--nq` | `0` (all) | Max queries to send (must be multiple of `group × block`) |
| `--loop` | `1` | Repeat N times (for benchmarking throughput) |

### Server Arguments (new ones only)

| Argument | Default | Description |
|----------|---------|-------------|
| `--serve` | off | Enable TCP serve mode |
| `--nprobe` | `100` | nprobe to use for all serve-mode searches |
| `--tcp_port` | `7777` | TCP port to listen on |

## Output

The server prints recall and search time after each query batch and appends a row
to `{benchmarks_path}/{dataset}/result/log_serve.csv` in the same format as the
benchmark mode CSV log.

Example server output:
```
[Master] TCP server listening on port 7777
[Master] Waiting for client...
[Master] Client connected. Entering serve loop.
[Master] Job: nq=96 k=100
[warmupSearchList 0, warmupSearchListSize 0]
...
recall: 0.9823  r2: 0.9901  time: 0.1247s
[Master] Results sent for nq=96
```

Example client output:
```
[Client] nq=96 d=128 k=100
[Client] Connecting to 127.0.0.1:7777...
[Client] Connected.
[Client] iter=0 sent 96 queries
[Client] iter=0 done in 0.1312s
  Query   0: [id=932085 d=54229.000] [id=934876 d=55091.000] ...
  Query   1: ...
[Client] Done.
```

## Running Benchmark Mode (unchanged)

```bash
mpirun -n 5 ./release/bin/query \
    --benchmarks_path ./benchmarks --dataset sift1m \
    --cache --group=2 --team=2 --block=4 --nprobes 100 200 300
```

## nq Divisibility Requirement

In `DIVIDE_GROUP` mode (the default), `nq` must satisfy:

```
nq % (groupCount × blockCount) == 0
```

With `--group=2 --block=4` this means `nq` must be a multiple of **8**.
Valid values near 100: 96, 104, 112, ...

If the client sends a non-conforming `nq` the search will produce wrong results
or hang. The client will print an error in a future version — for now just ensure
`--nq` is a valid multiple.

## Troubleshooting

**Server hangs at `[Master] Waiting for client...`**
Client has not connected yet. Start the client after seeing this message.

**Client hangs after `[Client] iter=0 sent N queries`**
The server did not receive the queries or workers are stuck. Check that `nq` is
divisible by `groupCount × blockCount` and that the server started successfully
(you should see `preSearch` timing printed before the waiting message).

**`No groundtruth at ...`**
Run benchmark mode once first to generate the groundtruth file.

**MPI OFI errors on Intel MPI**
Intel MPI's OFI dynamic process management (`MPI_Comm_connect`) does not work on
this machine. This implementation uses plain TCP sockets instead, which avoids
the issue entirely.