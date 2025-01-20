# HARMONY: A Scalable Distributed Vector Database for High-Throughput Approximate Nearest Neighbor Search

## Introduction
Harmony is a distributed ANNS system that employs a novel multi-granularity partition strategy, combining dimension-based and vector-based partition.

## Dataset

We have prepared some tiny datasets for testing in benchmarks folder. You can download the large datasets from the following links:

- nuswide: [GoogleDrive](https://drive.google.com/file/d/1d0w5XchVZvuRcV9sDZtbWC6TnB20tODM/view?usp=sharing)
- msong: [GoogleDrive](https://drive.google.com/file/d/1BcTuT4su77_Ue6Wi8EU340HSYoJeHwnD/view?usp=drive_link)
- sift1m: [GoogleDrive](https://drive.google.com/file/d/1BcTuT4su77_Ue6Wi8EU340HSYoJeHwnD/view?usp=drive_link)

If you build from docker, it will automatically download nuswide dataset from the link above.

### Dataset Format

You can prepare your own dataset in the following format, place it in the benchmarks folder.

```
benchmark
|-- nuswide
|   |-- origin
|   |   |-- msong_base.fvecs
|   |   |-- msong_query.fvecs
```

fvecs file format is as follows:

```
<4 bytes int representing num_dimension><num_dimension * sizeof(float) bytes raw data>
...
<4 bytes int representing num_dimension><num_dimension * sizeof(float) bytes raw data>
```

## Experimental Setup

Our server setup includes two Intel Xeon Gold 5318Y CPUs, each with 24 cores and 48 threads, totaling 96 CPU cores. The server boasts 2TB of memory and runs on CentOS Stream 8 operating system.

We also provide a dockerfile based on Ubuntu22.04 with all the dependencies installed.

## Quick Start

```bash
cmake -B release -DCMAKE_BUILD_TYPE=Release .
cmake --build release -j

mpirun -n 1 -bind-to none ./release/bin/query --benchmarks_path ./benchmarks --dataset msong  --run_faiss --verbose
mpirun -n 5 -bind-to none ./release/bin/query --benchmarks_path ./benchmarks --dataset msong  --group=2 --team=2 --block=4 
```

## How to Run

### Manual Installation

If you want to build the project on your own machine, you should install the following dependencies:

- build-essential (g++ >= 13.2.0)
- cmake
- openblas
- intel-mkl = 2024.1
- Eigen3
- MPI

### Build

```bash
cmake -B release -DCMAKE_BUILD_TYPE=Release .
cmake --build release -j

cmake -B build .
cmake --build build -j
```

We only measure pruning rates in debug or standard mode, so if you need performance-related metrics, please use the release compiled version. If you require metrics related to pruning rates, use the debug or standard compiled version.

### Run

We have prepared a fully functional script named `query` for conducting benchmark tests and other tasks. Next, we will demonstrate how to replicate our experimental results.

#### Faiss Baseline

As a baseline and to generate ground truth, we use faiss-ivfflat. You may execute run_faiss once to obtain baseline values.

```bash
mpirun -n 1 ./release/bin/query --benchmarks_path ./benchmarks --dataset msong \
  --nprobes 50 100 300 1000 --run_faiss --verbose
```

#### Harmony Index Generation

```bash
./release/bin/query --benchmarks_path ./benchmarks --dataset msong --train_only --verbose
```

#### Harmony Query Performance

Next, you can test the query performance with the following command, by default harmony use hybrid partitioning, specifying `--mode dim` tells harmony to use dimension partitioning, specifying `--mode vector` tells harmony to use vector partitioning. `--cache` is used to use the cached index. You can use `--blockSend`, `--disableOrderOpt`, `--disablePruning` to disable corresponding optimization.

```bash
mpirun -n 5 -bind-to none ./release/bin/query --benchmarks_path ./benchmarks --dataset sift1m  --cache --block=4 --group=2 --team=2  --disablePruning
mpirun -n 5 -bind-to none ./release/bin/query --benchmarks_path ./benchmarks --dataset msong  --mode dim --block=8
mpirun -n 5 -bind-to none ./release/bin/query --benchmarks_path ./benchmarks --dataset msong  --mode vector 
```

After running the above commands, you can check the results in `benchmarks/nuswide/result/log.csv`.

#### Further Usage

Finally, you can use the following command to get a more comprehensive usage guide for this script:

```bash
./release/bin/query --help

Usage: harmony [--help] [--version] [--benchmarks_path VAR] [--dataset VAR] [--input_format VAR] [--output_format VAR] [--k VAR] [--nprobes VAR...] [--opt_levels VAR...] [--train_only] [--cache] [--sub_nprobe_ratio VAR] [--metric VAR] [--run_faiss] [--loop VAR] [--nlist VAR] [--verbose] [--ratios VAR...] [--csv VAR] [--dataset_info] [--early_stop] [--block VAR] [--warmup_list_size VAR] [--warmup_list VAR] [--disablePruning] [--disableOrderOpt] [--inBalance] [--blockSend] [--fullWarmUp] [--group VAR] [--team VAR] [--mode VAR] [--HardInBalance] [--HardInBalanceRatio VAR] [--HardInBalanceTeam VAR]

Optional arguments:
  -h, --help            shows help message and exits 
  -v, --version         prints version information and exits 
  --benchmarks_path     benchmarks path [nargs=0..1] [default: "/home/xuqian/Triangle/benchmarks"]
  --dataset             dataset name [nargs=0..1] [default: "msong"]
  --input_format        format of the dataset [nargs=0..1] [default: "fvecs"]
  --output_format       format of the output [nargs=0..1] [default: "bin"]
  --k                   number of nearest neighbors [nargs=0..1] [default: 100]
  --nprobes             number of clusters to search [nargs=0..100] [default: {0}]
  --opt_levels          optimization levels [nargs=0..10] [default: {"OPT_NONE"}]
  --train_only          train only 
  --cache               use cached index 
  --sub_nprobe_ratio    ratio of the number of subNNs to the number of clusters [nargs=0..1] [default: 1]
  --metric              metric type [nargs=0..1] [default: "l2"]
  --run_faiss           run faiss 
  --loop                [nargs=0..1] [default: 1]
  --nlist               [nargs=0..1] [default: 0]
  --verbose             verbose 
  --ratios              search ratio [nargs=0..100] [default: {1}]
  --csv                 csv result file [nargs=0..1] [default: ""]
  --dataset_info        only output dataset-info to csv file 
  --early_stop          early stop 
  --block               number of blocks [nargs=0..1] [default: 0]
  --warmup_list_size    how many vectors in a list are used to warmup heap [nargs=0..1] [default: 0]
  --warmup_list         how many lists are used to warmup heap [nargs=0..1] [default: 0]
  --disablePruning      set pruning disabled 
  --disableOrderOpt     disable block search order optimization 
  --inBalance           
  --blockSend           use blocking MPI_Send instead of unblocking MPI_Isend with search phase 
  --fullWarmUp          use groundtruth to warmup heap 
  --group               The number of groups the nq query vectors need to be divided into. [nargs=0..1] [default: 1]
  --team                The number of teams the workers need to be divided into. Each team handles an individual part of base vectors [nargs=0..1] [default: 1]
  --mode                The mode of search [nargs=0..1] [default: "hybrid"]
  --HardInBalance       enable hard inBalance 
  --HardInBalanceRatio  control how inbalanced the search would be, 0.0 is perfectly balanced [nargs=0..1] [default: 1]
  --HardInBalanceTeam   number of Hard InBalance Team [nargs=0..1] [default: 0]
```


## License

This project is licensed under the MIT License.
