#!/bin/bash
# Script to run SIFT100M benchmark with proper memory management

set -e

DATASET=${DATASET:-sift100m}
NUM_PROCS=${NUM_PROCS:-5}
GROUP=${GROUP:-2}
TEAM=${TEAM:-2}
BLOCK=${BLOCK:-4}
MODE=${MODE:-group}
SKIP_TRAIN=${SKIP_TRAIN:-true}

echo "=========================================="
echo "Running SIFT100M Benchmark"
echo "=========================================="
echo "MPI Processes: $NUM_PROCS"
echo "Group: $GROUP, Team: $TEAM, Block: $BLOCK"
echo "Mode: $MODE"
echo "=========================================="

# Step 1: Check if index exists, train if needed
if [ "$SKIP_TRAIN" = "false" ] && ! docker exec harmony-mpi-head test -d "/app/benchmarks/${DATASET}/indexes"; then
  echo "Index not found. Training on single node (no MPI)..."
  docker exec harmony-mpi-head bash -c "
    export I_MPI_FABRICS=shm:tcp
    cd /app
    mpirun -np 1 ./release/bin/query \
      --benchmarks_path /app/benchmarks \
      --dataset ${DATASET} \
      --train_only \
      --verbose
  " || {
    echo "Training failed. Trying with smaller nlist..."
    docker exec harmony-mpi-head bash -c "
      export I_MPI_FABRICS=shm:tcp
      cd /app
      mpirun -np 1 ./release/bin/query \
        --benchmarks_path /app/benchmarks \
        --dataset ${DATASET} \
        --train_only \
        --nlist 5000 \
        --verbose
    "
  }
  echo "Training complete!"
else
  echo "Index found or training skipped."
fi

# Step 2: Run distributed query
echo "Starting distributed query execution..."
docker exec harmony-mpi-head bash -c "
  export I_MPI_HYDRA_BOOTSTRAP=ssh
  export FI_PROVIDER=tcp
  
  # Monitor memory and run
  mpirun -np $NUM_PROCS -machinefile /app/mpi/hostfile \
    ./release/bin/query \
      --benchmarks_path /app/benchmarks \
      --dataset $DATASET \
      --cache \
      --group=$GROUP \
      --team=$TEAM \
      --block=$BLOCK \
      --mode=$MODE \
      --nprobes=100 \
      --verbose
"

echo "=========================================="
echo "Job complete!"
echo "Results should be in: /app/benchmarks/${DATASET}/result/"
echo "=========================================="