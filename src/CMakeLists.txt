include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(SRC_FILES
    Clustering.cpp
    Index.cpp
    IVF.cpp
    node.cpp
)

add_library(Harmony STATIC ${SRC_FILES})

target_link_libraries(Harmony PUBLIC
    OpenMP::OpenMP_CXX
    mkl_intel_lp64
    mkl_sequential
    mkl_core
    MPI::MPI_CXX
)

add_executable(main main.cpp)
target_link_libraries(main Harmony faiss)

# query binary — no gRPC, no master_server needed
add_executable(query query.cpp master_server.cpp)
target_link_libraries(query Harmony faiss)

# Standalone TCP client — no MPI, no Harmony lib needed
add_executable(harmony_client client.cpp)
target_include_directories(harmony_client PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party
)

# ============================================================
# Usage:
#
# SERVE MODE (client + master + workers):
#
#   Terminal 1 (master+workers, ranks 1..N):
#   mpirun -n 5 ./release/bin/query \
#       --benchmarks_path ./benchmarks --dataset msong \
#       --cache --group=2 --team=2 --block=4 \
#       --nprobe 100
#
#   Terminal 2 (client, rank 0) — launched as a SEPARATE mpirun:
#   mpirun -n 1 ./release/bin/harmony_client \
#       --query ./benchmarks/msong/origin/msong_query.fvecs \
#       --k 10 --nq 100
#
# OR run everything together in one mpirun (6 ranks total):
#   mpirun -n 6 \
#       -host client_vm    ./release/bin/harmony_client \
#       -host master_vm:5  ./release/bin/query \
#           --benchmarks_path ./benchmarks --dataset msong \
#           --cache --group=2 --team=2 --block=4 --nprobe 100
#
# BENCHMARK MODE (original behaviour, no client rank):
#   mpirun -n 5 ./release/bin/query \
#       --benchmarks_path ./benchmarks --dataset msong \
#       --cache --block=4 --group=2 --team=2 --benchmark
# ============================================================