#!/bin/bash
set -e

echo "Starting Harmony worker at rank ${OMPI_COMM_WORLD_RANK}"

# Example:
exec ./release/bin/query \
    --benchmarks_path ./benchmarks \
    --dataset ./datasets \
    "$@"
