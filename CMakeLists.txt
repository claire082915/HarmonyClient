cmake_minimum_required(VERSION 3.21)


set(CMAKE_C_COMPILER $ENV{HOME}/gcc-13.2/bin/gcc)
set(CMAKE_CXX_COMPILER $ENV{HOME}/gcc-13.2/bin/g++)

project(Harmony)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Standard)
elseif (NOT CMAKE_BUILD_TYPE MATCHES "Debug|Release|Standard")
    message(FATAL_ERROR "Invalid build type: ${CMAKE_BUILD_TYPE}")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) 
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(OpenMP REQUIRED)
# find_package(GTest REQUIRED)
find_package(GTest QUIET)
if(GTest_FOUND)
    message(STATUS "GoogleTest found — enabling tests")
    enable_testing()
    add_subdirectory(tests)
else()
    message(WARNING "GoogleTest not found — skipping tests")
endif()
find_package(MPI REQUIRED)

if(DEFINED ENV{MKLROOT})
    set(MKL_ROOT $ENV{MKLROOT})
    message(STATUS "Using MKLROOT from environment: ${MKL_ROOT}")
else()
    message(FATAL_ERROR "MKLROOT not set. Please source oneAPI setvars.sh")
endif()

include_directories(${MKL_ROOT}/include)
link_directories(${MKL_ROOT}/lib/intel64)

set(MKL_LIBRARIES
    mkl_intel_lp64
    mkl_gnu_thread
    mkl_core
    gomp
    pthread
    m
    dl
)

option(ENABLE_PROFILING "Enable profiling" OFF)

if (ENABLE_PROFILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
endif()

# set(W_CXX_FLAGS " -Wall -Wextra")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${W_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
elseif (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "Standard")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${W_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -Ofast -mavx512f -march=native -mtune=native -DNDEBUG")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${W_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -Ofast -mavx512f -march=native -mtune=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${W_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -Ofast -march=native -mtune=native")

elseif (CMAKE_BUILD_TYPE STREQUAL "MPI")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${W_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -Ofast -mavx512f -march=native -mtune=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${W_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -Ofast -march=native -mtune=native")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${W_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -Ofast -march=znver2 -mavx2 -mfma -mtune=znver2 -DNDEBUG")
    set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_definitions(NSTATS)
endif()

include_directories(${CMAKE_SOURCE_DIR}/third_party)
include_directories(${CMAKE_SOURCE_DIR}/third_party/faiss)
include_directories($ENV{HOME}/eigen/include/eigen3)

# ✅ Disable AVX-512 for AMD EPYC
set(FAISS_ENABLE_GPU OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_AVX512 OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_AVX2 ON CACHE BOOL "" FORCE)

add_subdirectory(third_party/faiss)
add_subdirectory(src)