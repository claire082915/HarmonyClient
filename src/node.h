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
#include "heap.hpp"
#include "IVFScan.hpp"
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
              nb(nb)
              {}
        InitInfo() : d(0), block_dim(0), workerCount(0), nlist(0), blockCount(0), nprobe(0), nb(0)  {}
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

    // MPI_Comm worker_comm;

    //vector的每一个元素对应一个block
    // vector<MPI_Request> infoRequests;
    vector<MPI_Request> disRequests;
    vector<MPI_Request> sendRequests;
    // vector<MPI_Status> statuses;

    std::unique_ptr<idx_t[]> sendNextWorker; //接受到blockId为i的块的时候，应该发送给rank为sendNextWorker[i]的机器, sendNextWorker[i]=0说明可以发给master
    std::unique_ptr<idx_t[]> recvPrevWorker; //blockId为i的块应该从rank为recvPrevWorker[i]的机器接收，如果recvPrevWorker[i] = 0, 说明不需要接收, 直接可以算

    MyStopWatch uniWatch;

    void init(int rank) {
        MyStopWatch watch(true);
        // MPI_Comm_split(MPI_COMM_WORLD, 1, rank, &worker_comm);

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

        //提前malloc
        int presumeNq = 10000;
        std::unique_ptr<float[]> querysBuffer = std::make_unique<float[]>(presumeNq * info.d);
        this->querys = std::make_unique<float[]>(presumeNq * info.block_dim);
        listidqueries = std::make_unique<idx_t[]>(presumeNq * info.nprobe);  // 最近的nprobe个聚类中心的id
        heapTops = std::make_unique<float[]>(presumeNq);
        queryCompareSize = std::make_unique<size_t[]>(presumeNq);
        queryCompareSizePreSum = std::make_unique<size_t[]>(presumeNq + 1);
        sendNextWorker = std::make_unique<idx_t[]>(info.blockCount);
        recvPrevWorker = std::make_unique<idx_t[]>(info.blockCount);

        // blockSize = presumeNq / info.blockCount;
        blockDistancesSize = presumeNq / info.blockCount * info.nb;
        distancesForBlocks = vector<std::unique_ptr<float[]>>(info.blockCount);
        for (size_t i = 0; i < info.blockCount; i++) {
            distancesForBlocks[i] = std::make_unique<float[]>(blockDistancesSize);
        }

        disRequests = vector<MPI_Request>(info.blockCount);
        sendRequests = vector<MPI_Request>(info.blockCount);

        MPI_Barrier(MPI_COMM_WORLD); //对应preSearch最后的barrier

        uniWatch = MyStopWatch(false, "uniWatch", MAG);
        uniWatch.print(format("node {} cross barrier", rank), false);

        // nq, querys
        MPI_Bcast(&nq, sizeof(nq), MPI_BYTE, 0, MPI_COMM_WORLD);
        if(nq > presumeNq) {
            cerr << "presumeNq is too small" << endl;
            exit(1);
        }
        uniWatch.print(format("node {} nq", rank), false);
        // std::unique_ptr<float[]> querysBuffer = std::make_unique<float[]>(nq * info.d);
        // uniWatch.print(format("node {} nq querys buffer alloc", rank), false);
        MPI_Bcast(querysBuffer.get(), nq * info.d, MPI_FLOAT, 0, MPI_COMM_WORLD);
        uniWatch.print(format("node {} querys", rank), false);
        addQuerys(querysBuffer.get(), nq);
        uniWatch.print(format("node {} addQuerys", rank), false);

        // query最近的nprobe个聚类中心的id
        // listidqueries = std::make_unique<idx_t[]>(nq * info.nprobe);  // 最近的nprobe个聚类中心的id
        MPI_Bcast(listidqueries.get(), nq * info.nprobe, MPI_INT64_T, 0, MPI_COMM_WORLD);
        uniWatch.print(format("node {} listidqueries", rank), false);

        // queryCompareSize,queryCompareSizePreSum
        queryCompareSize = std::make_unique<size_t[]>(nq);
        MPI_Bcast(queryCompareSize.get(), nq, MPI_INT64_T, 0, MPI_COMM_WORLD);
        // queryCompareSizePreSum = std::make_unique<size_t[]>(nq + 1);
        MPI_Bcast(queryCompareSizePreSum.get(), (nq + 1), MPI_INT64_T, 0, MPI_COMM_WORLD);
        uniWatch.print(format("node {} queryCompareSize", rank), false);

        // 最大堆
        // heapTops = std::make_unique<float[]>(nq);
        MPI_Bcast(heapTops.get(), nq, MPI_FLOAT, 0, MPI_COMM_WORLD);
        uniWatch.print(format("node {} heapTops", rank), false);
        // for(int i = 0; i < nq; i++) {
        //     cout << format("Q{}, top{}", i, heapTops[i]) << endl;
        // }

        //初始化sendNextWorker, recvPrevWorker
        // sendNextWorker = std::make_unique<idx_t[]>(info.blockCount);
        // recvPrevWorker = std::make_unique<idx_t[]>(info.blockCount);
        MPI_Recv(sendNextWorker.get(), info.blockCount, MPI_INT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(recvPrevWorker.get(), info.blockCount, MPI_INT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        uniWatch.print(format("node {} sendNextWorker", rank), false);

        // 其他初始化
        blockSize = nq / info.blockCount;
        blockDistancesSize = blockSize * info.nb;
        // distancesForBlocks = vector<std::unique_ptr<float[]>>(blockDistancesSize);
        // for (size_t i = 0; i < info.blockCount; i++) {
        //     distancesForBlocks[i] = std::make_unique<float[]>(blockDistancesSize);
        // }
        uniWatch.print(format("node {} distancesForBlocks", rank), false);

        // 初始化request, status, 
        // infoRequests = vector<MPI_Request>(info.blockCount);
        // disRequests = vector<MPI_Request>(info.blockCount);
        // statuses = vector<MPI_Status>(info.blockCount);
        // cout << CRAN << "finish init" << rank << RESET << endl;
        // cout << RED << rank << RESET << endl;
        watch.print(format("Node {} Init", rank));
        uniWatch.print(format("node {} finish init", rank), false);

    }
   
    void addQuerys(const float* querys, size_t nq) {
        // this->querys = std::make_unique<float[]>(nq * info.block_dim);
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
        CUTIDS,
    };
    SearchResultInfo resultInfo;

    void search(bool cut) {
        uniWatch.print(format("node {} search() start", rank), false);
        MyStopWatch totalWatch(true);
        int searchedBlockCount = 0;
        vector<bool> isBlockSearched = vector<bool>(info.blockCount);
        for (size_t blockId = 0; blockId < info.blockCount; blockId++) {
            if(recvPrevWorker[blockId] == 0) {
                searchBlock(blockId, cut);
                isBlockSearched[blockId] = true;
                searchedBlockCount++;
            } else {
                size_t sender = recvPrevWorker[blockId];
                // size
                idx_t size;
                // tag 当做blockId
                MPI_Irecv(distancesForBlocks[blockId].get(), getTotalQueryCompareSize(blockId), MPI_FLOAT, sender, blockId, MPI_COMM_WORLD, &disRequests[blockId]);
                // cout << format("node({}) waiting for block({}) from node({})", rank, blockId, sender) << endl;
            }
        }

        //不断查
        MyStopWatch watch(true);
        while(searchedBlockCount < info.blockCount)
        for (size_t blockId = 0; blockId < info.blockCount; blockId++) {
            if(isBlockSearched[blockId]) {
                continue;
            }
            int isReceived;
            MPI_Status stat;
            MPI_Test(&disRequests[blockId], &isReceived, &stat);
            if(isReceived) {
                // cout << GREEN << format("node({}) received block({}) from node({})",rank, blockId, stat.MPI_SOURCE) << RESET << endl;
                watch.print(format("node {} wait", rank));
                searchBlock(blockId, cut);
                watch.reset();
                searchedBlockCount++;
                isBlockSearched[blockId] = true;
            } else {
                // cout << format("node({}) not receving block({})",rank, blockId) << endl;
            }
        }
        totalWatch.print(format("Search Finished node({}), total skip:{:.1f}%", rank, (double)totalSkip / totalCompare * 100));


        for(int i = 0; i < sendRequests.size(); i++) {
            MPI_Wait(&sendRequests[i], MPI_STATUS_IGNORE);
        }

    }
    size_t getTotalQueryCompareSize(size_t blockId) {
        size_t queryStart = blockId * blockSize;
        size_t totalQueryCompareSize = queryCompareSizePreSum[queryStart + blockSize] - queryCompareSizePreSum[queryStart];
        if((double)queryCompareSizePreSum[queryStart + blockSize] - queryCompareSizePreSum[queryStart] > INT_MAX) {
            cerr << RED << "increase block size" << RESET << endl;
            throw std::invalid_argument("increase block size");
        }
        return totalQueryCompareSize;
    }
    idx_t totalSkip = 0;
    idx_t totalCompare = 0;
    void searchBlock(size_t blockId, bool cut) {
        MyStopWatch searchWatch(true);

        auto clock1 = std::chrono::high_resolution_clock::now();

        size_t queryStart = blockId * blockSize;
        size_t totalQueryCompareSize = getTotalQueryCompareSize(blockId);
            
        // cout << RED << rank << "node search: blockId:" << blockId << " totalCompareSize:" << totalQueryCompareSize << RESET
        //      << endl;


        if (totalQueryCompareSize > blockDistancesSize) {
            cerr << "Error search" << endl;
            exit(1);
        }

        size_t nt = std::min(static_cast<size_t>(omp_get_max_threads()), blockSize);
        // cout << "node " << rank << " nt = " << nt << endl;
        size_t skip = 0;
#pragma omp parallel for num_threads(nt) reduction(+:skip)
// #pragma omp parallel for num_threads(nt)
        for (size_t q = queryStart; q < queryStart + blockSize; q++) {
            // cout << "node " << rank << " nt = " << omp_get_num_threads() << endl;
            size_t queryOffset =
                queryCompareSizePreSum[q] - queryCompareSizePreSum[queryStart];  // 第q个查询的结果应该存的地址偏移量

            size_t curDistancePosition = 0;  // 在一个查询向量的结果内
            for (size_t i = 0; i < info.nprobe; i++) {
                idx_t ivfId = listidqueries[q * info.nprobe + i];
                for (size_t v = 0; v < listSizes[ivfId]; v++) {
                    if(cut) {
                        if(distancesForBlocks[blockId][queryOffset + curDistancePosition] == INFINITY) {
                            skip++;
                        } else {
                            float dis = calculatedEuclideanDistance(querys.get() + q * info.block_dim,
                                                                    listCodes[ivfId].get() + v * info.block_dim,
                                                                    info.block_dim);
                            // cout << " " << queryOffset + curDistancePosition  << endl;
                            
                            // assert(queryOffset + curDistancePosition < totalQueryCompareSize);
                            distancesForBlocks[blockId][queryOffset + curDistancePosition] += dis;
                            if (distancesForBlocks[blockId][queryOffset + curDistancePosition] > heapTops[q]) {
                                distancesForBlocks[blockId][queryOffset + curDistancePosition] = INFINITY;
                            }
                        }
                        curDistancePosition++;
                    } else {
                        float dis = calculatedEuclideanDistance(querys.get() + q * info.block_dim,
                                                                listCodes[ivfId].get() + v * info.block_dim,
                                                                info.block_dim);
                        // cout << " " << queryOffset + curDistancePosition  << endl;
                        assert(queryOffset + curDistancePosition < totalQueryCompareSize);
                        distancesForBlocks[blockId][queryOffset + curDistancePosition] += dis;
                        curDistancePosition++;
                    }
                    
                }
            }
            // cout << curDistancePosition << " " << queryCompareSize[q] << endl;
            assert(curDistancePosition == queryCompareSize[q]);
        }
// #pragma omp parallel for 
        //  for (size_t q = queryStart; q < queryStart + blockSize; q++) {
        //     size_t queryOffset =
        //         queryCompareSizePreSum[q] - queryCompareSizePreSum[queryStart];  // 第q个查询的结果应该存的地址偏移量
        //     for (size_t i = 0; i < queryCompareSize[q]; i++){
        //         if(heapTops[q] < distancesForBlocks[blockId][queryOffset + i]) {
        //             skip++;
        //         }
        //     }
        // }

        // malloc0.3s, search0.9s
        // auto clock2 = std::chrono::high_resolution_clock::now();
        // std::cout << "node" << rank << '|' << blockId << "| search"
        // << std::chrono::duration<double>(clock2 - clock1).count() << "s" << std::endl;
        resultInfo = SearchResultInfo(totalQueryCompareSize, blockId);
        // auto clock3 = std::chrono::high_resolution_clock::now();
        // MPI_Send(&resultInfo, sizeof(SearchResultInfo), MPI_BYTE, sendNextWorker[blockId], SearchResultTag::INFO, MPI_COMM_WORLD);
        // cout << totalQueryCompareSize * sizeof(float) << endl;
        
        // cout << format("node({}) send block({}) to node({})", rank, blockId, sendNextWorker[blockId]) << endl;

        uniWatch.print(format("ready to perform node({}) -> block({}) -> node({})", rank, blockId, sendNextWorker[blockId]), false);
        MyStopWatch watch(true);
        // MPI_Request req;
        MPI_Isend(distancesForBlocks[blockId].get(), totalQueryCompareSize, MPI_FLOAT, sendNextWorker[blockId], blockId,
                 MPI_COMM_WORLD, &sendRequests[blockId]);
        // MPI_Request_free(&req);

        watch.print(format("node({}) -> block({}) -> node({}) 传输时间", rank, blockId, sendNextWorker[blockId]));
        //  
               // return SearchResultInfo(move(distancesForBlocks[blockId]), totalQueryCompareSize, blockId);
        // std::cout << "node" << rank << '|' << blockId << "| send"
        // << std::chrono::duration<double>(clock3 - clock2).count() << "s" << std::endl;

        // cout << format("node:{} block:{} skip:{:<-10} {:.1f}%", rank, blockId, skip, (double)skip / totalQueryCompareSize * 100) << endl;
        totalSkip += skip;
        totalCompare += totalQueryCompareSize;
        searchWatch.print(format("node({}) search block({}) skip:{:.1f}%", rank, blockId, (double)skip / totalQueryCompareSize * 100));
    }
};

class BaseWorker {
private:
public:
    struct InitInfo {
        size_t d, workerCount;
        size_t nlist;
        size_t nprobe, nb;
        size_t startIVFId, ivfCount;
        InitInfo(size_t d, size_t workerCount, size_t nlist, size_t nprobe,
                 size_t nb, size_t beginIVF, size_t ivfCount)
            : d(d),
              workerCount(workerCount),
              nlist(nlist),
              nprobe(nprobe),
              nb(nb),
              startIVFId(beginIVF),
              ivfCount(ivfCount)
              {}
        InitInfo() : d(0), workerCount(0), nlist(0), nprobe(0), nb(0), startIVFId(0),
              ivfCount(0) {}
        void print() {
            cout << GREEN << "InitInfo:[";
            cout << "d:" << d;
            cout << ",nlist:" << nlist;
            cout << ",nprobe:" << nprobe;
            cout << ",worker:" << workerCount;
            cout << ",nb:" << nb;
            cout << ",startIVFId:" << startIVFId;
            cout << ",ivfCount:" << ivfCount;
            cout << "]";
            cout << endl;
        }
    };
    // 代表一次搜索请求中需要包含的信息

    size_t rank = 0;
    size_t nq = 0;
    size_t k = 0;
    std::unique_ptr<size_t[]> listSizes;         // nlist个聚类的向量数
    vector<std::unique_ptr<float[]>> listCodes;  // nlist个聚类的向量数
    vector<std::unique_ptr<size_t[]>> listIds;  // nlist个聚类的向量数
    std::unique_ptr<float[]> querys;             // nq个查询向量, 维度是block_dim
    std::unique_ptr<idx_t[]> listidqueries;      // nq * nprobe 查询向量相近的聚类id
    // std::unique_ptr<size_t[]> queryCompareSize;  
    // std::unique_ptr<size_t[]> queryCompareSizePreSum;
    std::unique_ptr<float[]> distances;
    std::unique_ptr<idx_t[]> labels;

    InitInfo info;

    MyStopWatch uniWatch;

    void init(int rank) {
        MyStopWatch watch(true);

        this->rank = rank;

        // 1.InitInfo
        MPI_Recv(&info, sizeof(InitInfo), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        info.print();
        // IVF的ID通过startIVFid给出, 顺序排列

        // 2.IVF的大小，IVF的向量表示
        listSizes = std::make_unique<size_t[]>(info.ivfCount);
        MPI_Recv(listSizes.get(), info.ivfCount * sizeof(size_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        listCodes = vector<std::unique_ptr<float[]>>(info.ivfCount); //每个聚类对应的codes
        listIds = vector<std::unique_ptr<size_t[]>>(info.ivfCount); //每个聚类对应的ids
        for (size_t i = 0; i < info.ivfCount; i++) {
            listCodes[i] = std::make_unique<float[]>(listSizes[i] * info.d);
            listIds[i] = std::make_unique<size_t[]>(listSizes[i] * info.d);
            MPI_Recv(listCodes[i].get(), listSizes[i] * info.d , MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(listIds[i].get(), listSizes[i] * sizeof(size_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        watch.print("listCodes and listIds");

        //提前malloc
        int presumeNq = 10000;
        int presumeK = 1000;
        this->querys = std::make_unique<float[]>(presumeNq * info.d);
        listidqueries = std::make_unique<idx_t[]>(presumeNq * info.nprobe);  // 最近的nprobe个聚类中心的id
        // queryCompareSize = std::make_unique<size_t[]>(presumeNq);
        // queryCompareSizePreSum = std::make_unique<size_t[]>(presumeNq + 1);

        distances = std::make_unique<float[]>(presumeNq * presumeK);
        labels = std::make_unique<idx_t[]>(presumeNq * presumeK);
        init_result(METRIC_L2, presumeNq * presumeK, distances.get(), labels.get());

        MPI_Barrier(MPI_COMM_WORLD); //对应preSearch最后的barrier

        uniWatch = MyStopWatch(true, "uniWatch", MAG);
        uniWatch.print(format("node {} cross barrier", rank), false);

        // nq, k, querys
        MPI_Bcast(&nq, sizeof(nq), MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&k, sizeof(k), MPI_BYTE, 0, MPI_COMM_WORLD);
        if(nq * k > presumeK * presumeNq) {
            cerr << "presumeNq is too small" << endl;
            exit(1);
        }
        uniWatch.print(format("node {} nq", rank), false);
        MPI_Bcast(querys.get(), nq * info.d, MPI_FLOAT, 0, MPI_COMM_WORLD);
        uniWatch.print(format("node {} querys", rank), false);

        // query最近的nprobe个聚类中心的id
        MPI_Bcast(listidqueries.get(), nq * info.nprobe, MPI_INT64_T, 0, MPI_COMM_WORLD);
        uniWatch.print(format("node {} listidqueries", rank), false);

        // queryCompareSize,queryCompareSizePreSum
        // queryCompareSize = std::make_unique<size_t[]>(nq);
        // MPI_Bcast(queryCompareSize.get(), nq, MPI_INT64_T, 0, MPI_COMM_WORLD);
        // MPI_Bcast(queryCompareSizePreSum.get(), (nq + 1), MPI_INT64_T, 0, MPI_COMM_WORLD);
        // uniWatch.print(format("node {} queryCompareSize", rank), false);

        // 最大堆

        watch.print(format("Node {} Init", rank));
        uniWatch.print(format("node {} finish init", rank), false);

    }
    float* getCodes(idx_t ivfId) {
        idx_t index = ivfId - info.startIVFId;
        return listCodes[index].get();
    }
    size_t* getIds(idx_t ivfId) {
        idx_t index = ivfId - info.startIVFId;
        return listIds[index].get();
    }
    size_t getListSize(idx_t ivfId) {
        idx_t index = ivfId - info.startIVFId;
        return listSizes[index];
    }
    bool isResponsible(idx_t ivfId) {
        return ivfId >= info.startIVFId && ivfId < info.startIVFId + info.ivfCount;
    }

    void search() {
        uniWatch.print(format("node {} start search", rank), false);
        
#pragma omp parallel for 
        for(size_t q = 0; q < nq; q++) {
            std::unique_ptr<IVFScanBase> scaner = std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_NONE, EdgeDevice::EDGEDEVIVE_DISABLED>(info.d, k));  // 在聚类中心内部搜
            scaner->set_query(querys.get() + q * info.d);
            for (size_t i = 0; i < info.nprobe; i++) {
                idx_t ivfId = listidqueries[q * info.nprobe + i];
                if(!isResponsible(ivfId)) {
                    continue;
                }
                size_t listSize = getListSize(ivfId);
                float* codes = getCodes(ivfId);
                size_t* ids = getIds(ivfId);
                scaner->lite_scan_codes(listSize, codes, ids, distances.get() + q * k, labels.get() + q * k);
            }
        }
        MPI_Send(distances.get(), k * nq, MPI_FLOAT, 0, 0, MPI_COMM_WORLD); 
        MPI_Send(labels.get(), k * nq, MPI_INT64_T, 0, 0, MPI_COMM_WORLD); 
        uniWatch.print(format("node {} start search", rank), false);
    }
    
};
}  // namespace tribase
#endif