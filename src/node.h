#ifndef NODE_H
#define NODE_H
#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <vector>

#include "IVF.h"
#include "cassert"
#include "utils.h"

namespace tribase {

using namespace std;
class Worker {
private:
public:
    struct InitInfo {
        size_t d, block_dim, workerCount;
        size_t nlist;
        size_t blockCount, nprobe, nb;
        InitInfo(size_t d, size_t block_dim, size_t workerCount, size_t nlist, size_t blockCount, size_t nprobe,
                 size_t nb)
            : d(d),
              block_dim(block_dim),
              workerCount(workerCount),
              nlist(nlist),
              blockCount(blockCount),
              nprobe(nprobe),
              nb(nb) {}
        InitInfo() : d(0), block_dim(0), workerCount(0), nlist(0), blockCount(0), nprobe(0), nb(0) {}
        void print() {
            cout << GREEN << "InitInfo:[";
            cout << "d:" << d;
            cout << ",nlist:" << nlist;
            cout << ",nprobe:" << nprobe;
            cout << ",worker:" << workerCount;
            cout << ",block_dim:" << block_dim;
            cout << ",nb:" << nb;
            cout << "]";
            cout << endl;
        }
    };
    // 代表一次搜索请求中需要包含的信息

    size_t rank = 0;
    size_t blockSize = 0;
    size_t nq = 0;
    // std::unique_ptr<IVF[]> ivfs;      // nlist个聚类,聚类内部的向量维度是block_dim
    std::unique_ptr<size_t[]> listSizes;         // nlist个聚类的向量数
    vector<std::unique_ptr<float[]>> listCodes;  // nlist个聚类的向量数
    std::unique_ptr<float[]> querys;             // nq个查询向量, 维度是block_dim
    // std::unique_ptr<SearchBlock[]> blocks;       // 按照搜索顺序排列，第1个先搜索
    std::unique_ptr<idx_t[]> listidqueries;      // nq * nprobe 查询向量相近的聚类id
    std::unique_ptr<size_t[]> queryCompareSize;  // nq * nprobe 查询向量相近的聚类id
    // 为了计算第q个向量的distancesForQueryies的偏移量,偏移量是queryCompareSizePreSum[q]
    std::unique_ptr<size_t[]> queryCompareSizePreSum;

    // 每一块的计算结果的临时存储
    vector<std::unique_ptr<float[]>> distancesForBlocks;
    size_t blockDistancesSize;
    InitInfo info;

    void addIVFs(vector<std::unique_ptr<float[]>>& listCodesBuffer);

    void init(int rank) {
        // cout << CRAN << "start init" << rank << RESET << endl;
        this->rank = rank;

        // 1. InitInfo
        MPI_Bcast(&info, sizeof(InitInfo), MPI_BYTE, 0, MPI_COMM_WORLD);
        // info.print();


        // 2. IVF的大小，IVF的向量表示
        listSizes = std::make_unique<size_t[]>(info.nlist);
        MPI_Bcast(listSizes.get(), info.nlist * sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
        auto listCodesBuffer = vector<std::unique_ptr<float[]>>(info.nlist);
        for (size_t i = 0; i < info.nlist; i++) {
            listCodesBuffer[i] = std::make_unique<float[]>(listSizes[i] * info.d);
            MPI_Bcast(listCodesBuffer[i].get(), listSizes[i] * info.d , MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
        addIVFs(listCodesBuffer);

        // 3. nq, querys
        MPI_Bcast(&nq, sizeof(nq), MPI_BYTE, 0, MPI_COMM_WORLD);
        std::unique_ptr<float[]> querysBuffer = std::make_unique<float[]>(nq * info.d);
        MPI_Bcast(querysBuffer.get(), nq * info.d, MPI_FLOAT, 0, MPI_COMM_WORLD);
        addQuerys(querysBuffer.get(), nq);

        // 4. query最近的nprobe个聚类中心的id
        listidqueries = std::make_unique<idx_t[]>(nq * info.nprobe);  // 最近的nprobe个聚类中心的id
        MPI_Bcast(listidqueries.get(), nq * info.nprobe, MPI_INT64_T, 0, MPI_COMM_WORLD);

        // 5. queryCompareSize,queryCompareSizePreSum
        queryCompareSize = std::make_unique<size_t[]>(nq);
        MPI_Bcast(queryCompareSize.get(), nq, MPI_INT64_T, 0, MPI_COMM_WORLD);
        queryCompareSizePreSum = std::make_unique<size_t[]>(nq + 1);
        MPI_Bcast(queryCompareSizePreSum.get(), (nq + 1), MPI_INT64_T, 0, MPI_COMM_WORLD);

        // 6.其他初始化
        blockSize = nq / info.blockCount;
        blockDistancesSize = blockSize * info.nb;
        distancesForBlocks = vector<std::unique_ptr<float[]>>(blockDistancesSize);
        for (size_t i = 0; i < info.blockCount; i++) {
            distancesForBlocks[i] = std::make_unique<float[]>(blockDistancesSize);
        }
        // cout << CRAN << "finish init" << rank << RESET << endl;
    }
    void preSearchInit() {}
    // void init(size_t id, size_t d, size_t block_dim, size_t nodeCount, IVF* ivfs, size_t nlist, const float* querys,
    //           size_t querySize, size_t blockCount, size_t nprobe, size_t nb) {
    //     this->rank = id;
    //     if (nprobe > nlist) {
    //         nprobe = nlist;
    //     }
    //     this->nprobe = nprobe;
    //     this->d = d;
    //     this->block_dim = block_dim;
    //     this->blockSize = querySize / blockCount;
    //     this->blockCount = blockCount;
    //     addIVFs(ivfs, nlist);
    //     addQuerys(querys, querySize);
    //     initBlock();

    //     // 初始化distancesForBlocks
    //     //  cout << GREEN << "alloc" << blockDistancesSize  << " " << blockCount << RESET << endl;
    //     blockDistancesSize = blockSize * nb;
    //     distancesForBlocks = vector<std::unique_ptr<float[]>>(blockDistancesSize);
    //     for (size_t i = 0; i < blockCount; i++) {
    //         distancesForBlocks[i] = std::make_unique<float[]>(blockDistancesSize);
    //     }
    // }
    // void initBlock() {
    //     blocks = std::make_unique<SearchBlock[]>(blockCount);
    //     for (int i = 0; i < blockCount; i++) {
    //         blocks[i].blockId = i;  // 先按顺序排布
    //     }
    // }
    void addQuerys(const float* querys, size_t nq) {
        this->querys = std::make_unique<float[]>(nq * info.block_dim);
        copy_n_partial_vector(querys, this->querys.get(), info.d, info.block_dim, info.block_dim * (rank - 1), nq);
    }

    struct SearchResultInfo {
        // std::unique_ptr<float[]> distances;
        size_t size;
        size_t blockId;
        SearchResultInfo(size_t sz, size_t blockId) : size(sz), blockId(blockId) {}

        SearchResultInfo() : size(0), blockId(0) {}
        void print() {
            string formatted_string = std::format("SINFO:[Block({}), Size({})]", blockId, size);
            cout << BLUE << formatted_string << RESET << endl;
        }
    };
    enum SearchResultTag {
        INFO,
        DISTANCES,
    };
    SearchResultInfo resultInfo;

    void search(size_t blockId) {
        //         cout << BLUE << "alloc" << blockDistancesSize << " " << distancesForBlocks.size() << RESET << endl;

        auto clock1 = std::chrono::high_resolution_clock::now();

        size_t queryStart = blockId * blockSize;
        // size_t totalQueryCompareSize = 0;
        size_t totalQueryCompareSize =
            queryCompareSizePreSum[queryStart + blockSize] - queryCompareSizePreSum[queryStart];
        // cout << RED << "node search: blockId:" << blockId << " totalCompareSize:" << totalQueryCompareSize << RESET
        //      << endl;

        if (totalQueryCompareSize > blockDistancesSize) {
            cerr << "Error search" << endl;
            exit(1);
        }

        size_t nt = std::min(static_cast<size_t>(omp_get_max_threads()), blockSize);
        // cout << nt << endl;
#pragma omp parallel for num_threads(nt)
        for (size_t q = queryStart; q < queryStart + blockSize; q++) {
            size_t queryOffset =
                queryCompareSizePreSum[q] - queryCompareSizePreSum[queryStart];  // 第q个查询的结果应该存的地址偏移量

            size_t curDistancePosition = 0;  // 在一个查询向量的结果内
            for (size_t i = 0; i < info.nprobe; i++) {
                idx_t ivfId = listidqueries[q * info.nprobe + i];
                for (size_t v = 0; v < listSizes[ivfId]; v++) {
                    float dis = calculatedEuclideanDistance(querys.get() + q * info.block_dim,
                                                            listCodes[ivfId].get() + v * info.block_dim,
                                                            info.block_dim);
                    // cout << " " << queryOffset + curDistancePosition  << endl;
                    assert(queryOffset + curDistancePosition < totalQueryCompareSize);
                    distancesForBlocks[blockId][queryOffset + curDistancePosition] = dis;
                    curDistancePosition++;
                }
            }
            // cout << curDistancePosition << " " << queryCompareSize[q] << endl;
            assert(curDistancePosition == queryCompareSize[q]);
        }
        // assert(curDistancePosition == totalQueryCompareSize);
        // cout << BLUE;
        // cout << distances.size() << " " << id << endl;
        // for(auto vec : distances) {
        //     cout << vec.size() << ": ";
        //     for(float f : vec) {
        //         cout << f << " ";
        //     }
        //     cout << endl;
        // }
        // cout << RESET;

        // malloc0.3s, search0.9s
        auto clock2 = std::chrono::high_resolution_clock::now();
        std::cout << "node" << rank << '|' << blockId << "| search"
        << std::chrono::duration<double>(clock2 - clock1).count() << "s" << std::endl;
        resultInfo = SearchResultInfo(totalQueryCompareSize, blockId);
        auto clock3 = std::chrono::high_resolution_clock::now();
        MPI_Send(&resultInfo, sizeof(SearchResultInfo), MPI_BYTE, 0, SearchResultTag::INFO, MPI_COMM_WORLD);
        // cout << totalQueryCompareSize * sizeof(float) << endl;
        
        MPI_Send(distancesForBlocks[blockId].get(), totalQueryCompareSize, MPI_FLOAT, 0, SearchResultTag::DISTANCES,
                 MPI_COMM_WORLD);
        //         // return SearchResultInfo(move(distancesForBlocks[blockId]), totalQueryCompareSize, blockId);
        std::cout << "node" << rank << '|' << blockId << "| send"
        << std::chrono::duration<double>(clock3 - clock2).count() << "s" << std::endl;
    }
};

}  // namespace tribase
#endif