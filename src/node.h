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
        bool sync;
        InitInfo(size_t d, size_t block_dim, size_t workerCount, size_t nlist, size_t blockCount, size_t nprobe,
                 size_t nb, bool sync)
            : d(d),
              block_dim(block_dim),
              workerCount(workerCount),
              nlist(nlist),
              blockCount(blockCount),
              nprobe(nprobe),
              nb(nb),
              sync(sync)
              {}
        InitInfo() : d(0), block_dim(0), workerCount(0), nlist(0), blockCount(0), nprobe(0), nb(0), sync(true) {}
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
    std::unique_ptr<idx_t[]> blockSearchOrder;   // search block的顺序，第i个元素是第i个要进行search的blockId
    std::unique_ptr<idx_t[]> listidqueries;      // nq * nprobe 查询向量相近的聚类id
    std::unique_ptr<size_t[]> queryCompareSize;  // nq * nprobe 查询向量相近的聚类id
    // 为了计算第q个向量的distancesForQueryies的偏移量,偏移量是queryCompareSizePreSum[q]
    std::unique_ptr<size_t[]> queryCompareSizePreSum;
    std::unique_ptr<float[]> heapTops;

    // 每一块的计算结果的临时存储
    vector<std::unique_ptr<float[]>> distancesForBlocks;
    size_t blockDistancesSize;
    InitInfo info;

    void addIVFs(vector<std::unique_ptr<float[]>>& listCodesBuffer);

    MPI_Comm worker_comm;

    void init(int rank) {
        MPI_Comm_split(MPI_COMM_WORLD, 1, rank, &worker_comm);

        // Synchronize only the worker processes at the barrier

        // cout << CRAN << "start init" << rank << RESET << endl;
        this->rank = rank;

        // InitInfo
        MPI_Bcast(&info, sizeof(InitInfo), MPI_BYTE, 0, MPI_COMM_WORLD);

        // IVF的大小，IVF的向量表示
        listSizes = std::make_unique<size_t[]>(info.nlist);
        MPI_Bcast(listSizes.get(), info.nlist * sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
        auto listCodesBuffer = vector<std::unique_ptr<float[]>>(info.nlist);
        for (size_t i = 0; i < info.nlist; i++) {
            listCodesBuffer[i] = std::make_unique<float[]>(listSizes[i] * info.d);
            MPI_Bcast(listCodesBuffer[i].get(), listSizes[i] * info.d , MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
        addIVFs(listCodesBuffer);

        // Search顺序
        blockSearchOrder = std::make_unique<idx_t[]>(info.blockCount);   // search block的顺序，第i个元素是第i个要进行search的blockId
        MPI_Recv(blockSearchOrder.get(), info.blockCount, MPI_INT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // cout << "Q" << rank << endl;
        // printVector(blockSearchOrder.get(), info.blockCount, BLUE);

        // nq, querys
        MPI_Bcast(&nq, sizeof(nq), MPI_BYTE, 0, MPI_COMM_WORLD);
        std::unique_ptr<float[]> querysBuffer = std::make_unique<float[]>(nq * info.d);
        MPI_Bcast(querysBuffer.get(), nq * info.d, MPI_FLOAT, 0, MPI_COMM_WORLD);
        addQuerys(querysBuffer.get(), nq);

        // query最近的nprobe个聚类中心的id
        listidqueries = std::make_unique<idx_t[]>(nq * info.nprobe);  // 最近的nprobe个聚类中心的id
        MPI_Bcast(listidqueries.get(), nq * info.nprobe, MPI_INT64_T, 0, MPI_COMM_WORLD);

        // queryCompareSize,queryCompareSizePreSum
        queryCompareSize = std::make_unique<size_t[]>(nq);
        MPI_Bcast(queryCompareSize.get(), nq, MPI_INT64_T, 0, MPI_COMM_WORLD);
        queryCompareSizePreSum = std::make_unique<size_t[]>(nq + 1);
        MPI_Bcast(queryCompareSizePreSum.get(), (nq + 1), MPI_INT64_T, 0, MPI_COMM_WORLD);

        // 最大堆
        heapTops = std::make_unique<float[]>(nq);
        MPI_Bcast(heapTops.get(), nq, MPI_FLOAT, 0, MPI_COMM_WORLD);
        // for(int i = 0; i < nq; i++) {
        //     cout << format("Q{}, top{}", i, heapTops[i]) << endl;
        // }
        // 其他初始化
        blockSize = nq / info.blockCount;
        blockDistancesSize = blockSize * info.nb;
        distancesForBlocks = vector<std::unique_ptr<float[]>>(blockDistancesSize);
        for (size_t i = 0; i < info.blockCount; i++) {
            distancesForBlocks[i] = std::make_unique<float[]>(blockDistancesSize);
        }
        // cout << CRAN << "finish init" << rank << RESET << endl;
        // cout << RED << rank << RESET << endl;
    }
    void preSearchInit() {}
   
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

    void search() {
        if(info.sync) {
            cout << YELLOW << "Sync On" << RESET << endl;
        } else {
            // cout << YELLOW << "Sync Off" << RESET << endl;
        }
        for (size_t i = 0; i < info.blockCount; i++) {
            idx_t blockId = blockSearchOrder[i];
            searchBlock(blockId);
            if(info.sync) {
                MPI_Barrier(worker_comm);
            }
        }
    }
    void searchBlock(size_t blockId) {
        size_t skip = 0;

        auto clock1 = std::chrono::high_resolution_clock::now();

        size_t queryStart = blockId * blockSize;
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
// #pragma omp parallel for 
         for (size_t q = queryStart; q < queryStart + blockSize; q++) {
            size_t queryOffset =
                queryCompareSizePreSum[q] - queryCompareSizePreSum[queryStart];  // 第q个查询的结果应该存的地址偏移量
            for (size_t i = 0; i < queryCompareSize[q]; i++){
                if(heapTops[q] < distancesForBlocks[blockId][queryOffset + i]) {
                    skip++;
                }
            }
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

        cout << format("node:{} block:{} skip:{:<-10} {:.3f}%", rank, blockId, skip, (double)skip / totalQueryCompareSize * 100) << endl;
    }
};

}  // namespace tribase
#endif