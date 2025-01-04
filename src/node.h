#ifndef NODE_H
#define NODE_H
#include <mpi.h>
#include <numeric>
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

class Index;

using namespace std;
class Worker {
private:
public:
    struct InitInfo {
        size_t d, block_dim, workerCount;
        size_t nlist;
        size_t blockCount, nprobe, nb;
        idx_t presumeBlockDistancesSize;
        InitInfo(size_t d, size_t block_dim, size_t workerCount, size_t nlist, size_t blockCount, size_t nprobe,
                 size_t nb, idx_t presumeBlockDistancesSize)
            : d(d),
              block_dim(block_dim),
              workerCount(workerCount),
              nlist(nlist),
              blockCount(blockCount),
              nprobe(nprobe),
              nb(nb),
              presumeBlockDistancesSize(presumeBlockDistancesSize)
              {}
        InitInfo() : d(0), block_dim(0), workerCount(0), nlist(0), blockCount(0), nprobe(0), nb(0), presumeBlockDistancesSize(0)  {}
        void print() {
            cout << GREEN << "InitInfo:[";
            cout << "d:" << d;
            cout << ",nlist:" << nlist;
            cout << ",nprobe:" << nprobe;
            cout << ",worker:" << workerCount;
            cout << ",block_dim:" << block_dim;
            cout << ",nb:" << nb;
            cout << ",presume:" << presumeBlockDistancesSize / 1000000000 << "GB";
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
    std::unique_ptr<idx_t[]> queryCompareSize;  // nq * nprobe 查询向量相近的聚类id
    // 为了计算第q个向量的distancesForQueryies的偏移量,偏移量是queryCompareSizePreSum[q]
    std::unique_ptr<idx_t[]> queryCompareSizePreSum;
    std::unique_ptr<float[]> heapTops;

    // 每一块的计算结果的临时存储
    class DistanceBufferPool {
        vector<std::unique_ptr<float[]>> distancesForBlocks;
        vector<int> blockIds; //distancesForBlocks每一个的blockid是什么
        vector<bool> used; // distancesForBlocks是否正在被使用
        size_t spareBlock = 0;
        bool useDynamicAlloc = false; //如果为真则distancesForBlocks的下标不代表blockid， 否则代表
        size_t size = 0;

        struct Action {
            bool use;
            size_t blockId;
            idx_t compareSize;
            size_t sender;
            Action(size_t blockId) : blockId(blockId) {
                use = true;
            }
            Action(size_t blockId, idx_t compareSize, size_t sender) 
            : blockId(blockId), compareSize(compareSize), sender(sender) {
                use = false;
            }
        };
        queue<Action> waitList; //bool代表是不是use

        void IRecvSplit(float* buffer, size_t blockId, idx_t compareSize, size_t sender, vector<MPI_Request>& reqs) {
            if(compareSize > INT_MAX) {
                cout << YELLOW << format("WARNING: block {} compare size too big , split into {}", blockId, compareSize / INT_MAX + 1) << RESET << endl;
                idx_t sizeToSend = compareSize;
                while(sizeToSend > INT_MAX) {
                    MPI_Request req;
                    //TODO 这样不行，Irecv顺序不一样
                    MPI_Irecv(buffer + compareSize - sizeToSend, INT_MAX, MPI_FLOAT, sender, blockId, MPI_COMM_WORLD, &req);
                    reqs.push_back(req);
                    sizeToSend -= INT_MAX;
                }
                MPI_Irecv(buffer + compareSize - sizeToSend, sizeToSend, MPI_FLOAT, sender, blockId, MPI_COMM_WORLD, &reqs[0]);
            } else {
                MPI_Irecv(buffer, compareSize, MPI_FLOAT, sender, blockId, MPI_COMM_WORLD, &reqs[0]);
            }
        }
        Worker* worker;
    public:

        DistanceBufferPool(const InitInfo& info, Worker* worker) : worker(worker) {
            distancesForBlocks = vector<std::unique_ptr<float[]>>(info.blockCount);
            waitList = queue<Action>();
            try {
                for (size_t i = 0; i < info.blockCount; i++) {
                    distancesForBlocks[size] = std::make_unique<float[]>(info.presumeBlockDistancesSize);
                    size++;
                    // cout << format("buffer {}", size) << endl;
                }
                cout << YELLOW << format("use Normal Alloc") << RESET << endl;
            } catch (const std::bad_alloc& e) { 
                useDynamicAlloc = true;
                blockIds = vector<int>(size, -1);
                used = vector<bool>(size, false);
                spareBlock = size;
                cout << YELLOW << format("use Dynamic Alloc") << RESET << endl;
            }
            // for (size_t i = 0; i < info.blockCount / 2; i++) {
            //     distancesForBlocks[size] = std::make_unique<float[]>(info.presumeBlockDistancesSize);
            //     size++;
            //     // cout << format("buffer {}", size) << endl;
            // }
            // useDynamicAlloc = true;
            cout << format("buffer {}", size) << endl;
        }
        float* getBuffer(size_t blockId) {
            if(!useDynamicAlloc) {
                return distancesForBlocks[blockId].get();
            }
            for (int i = 0; i < size; i++) {
                if(used[i] && (blockIds[i] == blockId)) {
                    return distancesForBlocks[i].get();
                }
            }
            cerr << RED << format("getBuffer {} not found", blockId) << RESET << endl;
            return nullptr;
        }

        bool IRecv(size_t blockId, idx_t compareSize, size_t sender, vector<MPI_Request>& reqs) {
            if(!useDynamicAlloc) {
                IRecvSplit(distancesForBlocks[blockId].get(), blockId, compareSize, sender, reqs);
                return true;
            } else {
                if(spareBlock <= 0) {
                    waitList.push(Action(blockId, compareSize, sender));
                    return false;
                } else {
                    for (int i = 0; i < size; i++) {
                        if(used[i] == false) {
                            used[i] = true;
                            blockIds[i] = blockId;
                            // MPI_Irecv(distancesForBlocks[i].get(), compareSize, MPI_FLOAT, sender, blockId, MPI_COMM_WORLD, &reqs[0]);
                            IRecvSplit(distancesForBlocks[i].get(), blockId, compareSize, sender, reqs);
                            cout << format("Irecv {}", blockId) << endl;
                            spareBlock--;
                            return true;
                        }
                    }
                }
                return false;
                // cerr << ""
            }
        }
        bool use(size_t blockId) {
            if(!useDynamicAlloc) {
                return true;
            } else {
                if(spareBlock <= 0) {
                    waitList.push(Action(blockId));
                    return false;
                } 
                for (int i = 0; i < size; i++) {
                    if(used[i] == false) {
                        used[i] = true;
                        blockIds[i] = blockId;
                        cout << format("use {}", blockId) << endl;
                        spareBlock--;
                        return true;
                    }
                }
                return false;
            }
        }
        bool releaseBuffer(size_t blockId) {
            if(!useDynamicAlloc) {
                return true;
            } else {
                for (int i = 0; i < size; i++) {
                    if(used[i] && (blockIds[i] == blockId)) {
                        used[i] = false;
                        blockIds[i] = -1;
                        cout << format("release {}", blockId) << endl;
                        spareBlock++;
                        if(!waitList.empty()) {
                            auto action = waitList.front();
                            if(action.use) {
                                use(action.blockId);
                            } else {
                                IRecv(action.blockId, action.compareSize, action.sender, worker->disRequests[action.blockId]);
                            }
                            waitList.pop();
                        }
                        return true;
                    }
                }
                return false;
            }
        }
    };
    std::unique_ptr<DistanceBufferPool> distanceBufferPool;

    // idx_t presumeBlockDistancesSize;
    InitInfo info;

    void addIVFs(vector<std::unique_ptr<float[]>>& listCodesBuffer);

    // MPI_Comm worker_comm;

    //vector的每一个元素对应一个block
    // vector<MPI_Request> infoRequests;
    vector<vector<MPI_Request>> disRequests;
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
        info.print();

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
        queryCompareSize = std::make_unique<idx_t[]>(presumeNq);
        queryCompareSizePreSum = std::make_unique<idx_t[]>(presumeNq + 1);
        sendNextWorker = std::make_unique<idx_t[]>(info.blockCount);
        recvPrevWorker = std::make_unique<idx_t[]>(info.blockCount);

        // blockSize = presumeNq / info.blockCount;
        // presumeBlockDistancesSize = presumeNq / info.blockCount * info.nb * 2;
        // cout << presumeBlockDistancesSize << "blockD" << endl;
        distanceBufferPool = std::make_unique<DistanceBufferPool>(info, this);
        watch.print(format("node {} distancesForBlocks", rank));

        disRequests = vector<vector<MPI_Request>>(info.blockCount);
        for(int i = 0; i < disRequests.size(); i++) {
            disRequests[i] = vector<MPI_Request>(1);
        }
        sendRequests = vector<MPI_Request>(info.blockCount);

        //初始化sendNextWorker, recvPrevWorker
        // sendNextWorker = std::make_unique<idx_t[]>(info.blockCount);
        // recvPrevWorker = std::make_unique<idx_t[]>(info.blockCount);
        MPI_Recv(sendNextWorker.get(), info.blockCount, MPI_INT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(recvPrevWorker.get(), info.blockCount, MPI_INT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

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


        // 其他初始化
        blockSize = nq / info.blockCount;
        

        // 初始化request, status, 
        
        watch.print(format("Node {} Init", rank));
        uniWatch.print(format("node {} finish init", rank), false);

    }
   
    void addQuerys(const float* querys, size_t nq) {
        // this->querys = std::make_unique<float[]>(nq * info.block_dim);
        copy_n_partial_vector(querys, this->querys.get(), info.d, info.block_dim, info.block_dim * (rank - 1), nq);
    }

    void search(bool cut) {
        uniWatch.print(format("node {} search() start", rank), false);
        MyStopWatch totalWatch(true);
        int searchedBlockCount = 0;
        vector<bool> isBlockSearched = vector<bool>(info.blockCount);

        //先给可以算的分配内存
        for (size_t blockId = 0; blockId < info.blockCount; blockId++) {
            if(recvPrevWorker[blockId] == 0) {
                //直接标记使用
                distanceBufferPool->use(blockId);
            }
        }
        for (size_t blockId = 0; blockId < info.blockCount; blockId++) {
            if(recvPrevWorker[blockId] != 0) {
                size_t sender = recvPrevWorker[blockId];
                // tag 当做blockId
                // MPI_Irecv(distancesForBlocks[blockId].get(), getTotalQueryCompareSize(blockId), MPI_FLOAT, sender, blockId, MPI_COMM_WORLD, &disRequests[blockId]);
                // MPI_Irecv(distanceBufferPool->getBuffer(blockId), getTotalQueryCompareSize(blockId), MPI_FLOAT, sender, blockId, MPI_COMM_WORLD, &disRequests[blockId]);
                bool suc = distanceBufferPool->IRecv(blockId, getTotalQueryCompareSize(blockId), sender, disRequests[blockId]);
                if(!suc) {
                    break;
                }
                // cout << format("node({}) waiting for block({}) from node({})", rank, blockId, sender) << endl;
            } 
        }

        //不断查
        MyStopWatch watch(false);
        while(searchedBlockCount < info.blockCount)
        for (size_t blockId = 0; blockId < info.blockCount; blockId++) {
            if(isBlockSearched[blockId]) {
                continue;
            }
            if(recvPrevWorker[blockId] == 0) {
                // searchBlock(blockId, cut);
                searchBlock(blockId, cut);
                isBlockSearched[blockId] = true;
                searchedBlockCount++;
            } else {
                int isReceived;
                MPI_Status stat;
                bool testFail = false;
                for(auto& req : disRequests[blockId]) {
                    MPI_Test(&req, &isReceived, &stat);
                    if(!isReceived) {
                        testFail = true;
                        break;
                    }
                }
                if(!testFail) {
                    // cout << GREEN << format("node({}) received block({}) from node({})",rank, blockId, stat.MPI_SOURCE) << RESET << endl;
                    watch.print(format("node {} wait", rank));
                    searchBlock(blockId, cut);
                    watch.reset();
                    searchedBlockCount++;
                    isBlockSearched[blockId] = true;
                }
            }
        }
        totalWatch.print(format("Search Finished node({}), total skip:{:.1f}%", rank, (double)totalSkip / totalCompare * 100));

        MPI_Waitall(sendRequests.size(), sendRequests.data(), MPI_STATUS_IGNORE);

    }
    idx_t getTotalQueryCompareSize(size_t blockId) {
        size_t queryStart = blockId * blockSize;
        idx_t totalQueryCompareSize = queryCompareSizePreSum[queryStart + blockSize] - queryCompareSizePreSum[queryStart];
        // if((double)queryCompareSizePreSum[queryStart + blockSize] - queryCompareSizePreSum[queryStart] > INT_MAX) {
        //     cerr << RED << "increase block size" << RESET << endl;
        //     throw std::invalid_argument("increase block size");
        // }
        return totalQueryCompareSize;
    }
    idx_t totalSkip = 0;
    idx_t totalCompare = 0;
    void searchBlock(size_t blockId, bool cut) {
        MyStopWatch searchWatch(false);

        float* distanceBuffer = distanceBufferPool->getBuffer(blockId);

        size_t queryStart = blockId * blockSize;
        idx_t totalQueryCompareSize = getTotalQueryCompareSize(blockId);
            
        // cout << RED << rank << "node search: blockId:" << blockId << " totalCompareSize:" << totalQueryCompareSize << RESET
        //      << endl;


        if (totalQueryCompareSize > info.presumeBlockDistancesSize) {
            cerr << "Error blockDistancesSize too small" << endl;
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
                        // if(distancesForBlocks[blockId][queryOffset + curDistancePosition] == INFINITY) {
                        if(distanceBuffer[queryOffset + curDistancePosition] == INFINITY) {
                            skip++;
                        } else {
                            float dis = calculatedEuclideanDistance(querys.get() + q * info.block_dim,
                                                                    listCodes[ivfId].get() + v * info.block_dim,
                                                                    info.block_dim);
                            // cout << " " << queryOffset + curDistancePosition  << endl;
                            
                            // assert(queryOffset + curDistancePosition < totalQueryCompareSize);
                            distanceBuffer[queryOffset + curDistancePosition] += dis;
                            if (distanceBuffer[queryOffset + curDistancePosition] > heapTops[q]) {
                                distanceBuffer[queryOffset + curDistancePosition] = INFINITY;
                            }
                        }
                        curDistancePosition++;
                    } else {
                        float dis = calculatedEuclideanDistance(querys.get() + q * info.block_dim,
                                                                listCodes[ivfId].get() + v * info.block_dim,
                                                                info.block_dim);
                        // cout << " " << queryOffset + curDistancePosition  << endl;
                        assert(queryOffset + curDistancePosition < totalQueryCompareSize);
                        distanceBuffer[queryOffset + curDistancePosition] += dis;
                        curDistancePosition++;
                    }
                    
                }
            }
            // cout << curDistancePosition << " " << queryCompareSize[q] << endl;
            assert(curDistancePosition == queryCompareSize[q]);
        }

        uniWatch.print(format("ready to perform node({}) -> block({}) -> node({})", rank, blockId, sendNextWorker[blockId]), false);
        MyStopWatch watch(false);
        if(totalQueryCompareSize > INT_MAX) {
            cout << "> INT_MAX" << endl;
            idx_t sizeToSend = totalQueryCompareSize;
            while(sizeToSend > INT_MAX) {
                MPI_Request request;
                MPI_Isend(distanceBuffer + totalQueryCompareSize - sizeToSend, INT_MAX, MPI_FLOAT, sendNextWorker[blockId], blockId, MPI_COMM_WORLD, &request);
                sendRequests.push_back(request);
                sizeToSend -= INT_MAX;
            }
            MPI_Isend(distanceBuffer + totalQueryCompareSize - sizeToSend, sizeToSend, MPI_FLOAT, sendNextWorker[blockId], blockId, MPI_COMM_WORLD, &sendRequests[blockId]);
        } else {
            MPI_Isend(distanceBuffer, totalQueryCompareSize, MPI_FLOAT, sendNextWorker[blockId], blockId, MPI_COMM_WORLD, &sendRequests[blockId]);
        }

        watch.print(format("node({}) -> block({}) -> node({}) 传输时间", rank, blockId, sendNextWorker[blockId]));

        totalSkip += skip;
        totalCompare += totalQueryCompareSize;

        // distanceBufferPool->releaseBuffer(blockId); 不能直接release, 要检查buffer是不是已经发送完毕了

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
    // vector<std::unique_ptr<float[]>> listCodes;  // nlist个聚类的向量数
    // vector<std::unique_ptr<size_t[]>> listIds;  // nlist个聚类的向量数
    std::unique_ptr<float[]> querys;             // nq个查询向量, 维度是block_dim
    std::unique_ptr<idx_t[]> listidqueries;      // nq * nprobe 查询向量相近的聚类id
    // std::unique_ptr<size_t[]> queryCompareSize;  
    // std::unique_ptr<size_t[]> queryCompareSizePreSum;
    std::unique_ptr<float[]> distances;
    std::unique_ptr<idx_t[]> labels;

    InitInfo info;

    MyStopWatch uniWatch;

    std::unique_ptr<Index> index;
    // Index* index;

    // void init(int rank, tribase::Index* index) {
    void init(int rank);
    
    
    // void single_thread_search_simple(size_t n, const float* queries, size_t k, float* distances, idx_t* labels) {

    //     std::unique_ptr<IVFScanBase> scaner = std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_NONE, EdgeDevice::EDGEDEVIVE_DISABLED>(info.d, k)); 

    //     //下面四个向量都和i绑定，也就是和每一个查询绑定
    //     float* disi = distances; //结果，查询向量最近的k个向量的距离
    //     idx_t* idxi = labels; //结果，查询向量最近的k个向量的id
    //     idx_t* listids = listidqueries.get();             // 单个查询对应的IVF聚类中心id
    //     MyStopWatch w;
    //     for (size_t i = 0; i < n; i++) {
    //         //每一个i对应一个查询
    //         scaner->set_query(queries + i * info.d);

    //         for (size_t j = 0; j < info.nprobe; j++) {
    //             //在第j个聚类中搜索所有点

    //             idx_t ivfId = listids[j];
    //             if(!(ivfId >= info.startIVFId && ivfId < info.startIVFId + info.ivfCount)) {
    //                 continue;
    //             }
    //             idx_t index = ivfId - info.startIVFId;
    //             size_t listSize = listSizes[index];
    //             float* codes = listCodes[index].get();
    //             size_t* ids = listIds[index].get();


    //             // MyStopWatch wa;
    //             scaner->lite_scan_codes(listSize, codes, ids, disi, idxi);
    //             // if(i == 3) {
    //                 // cout << RED << listSize << RESET << endl;
    //                 // wa.print(format("q {} ivf {}", i, j), false);
    //             // }
    //         }
    //         disi += k;
    //         idxi += k;
    //         // w.print(format("q {}", i));
    //     }


    //     // std::unique_ptr<IVFScanBase> scaner = std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_NONE, EdgeDevice::EDGEDEVIVE_DISABLED>(info.d, k)); 

    //     // //下面四个向量都和i绑定，也就是和每一个查询绑定
    //     // float* disi = distances; //结果，查询向量最近的k个向量的距离
    //     // idx_t* idxi = labels; //结果，查询向量最近的k个向量的id
    //     // idx_t* listids = listidqueries.get();//单个查询对应的聚类中心id

    //     // for (size_t i = 0; i < n; i++) {
    //     //     //每一个i对应一个查询
    //     //     scaner->set_query(queries + i * d);
    //     //     //获取最近的nprobe个聚类中心
    //     //     for (size_t j = 0; j < nprobe; j++) {
    //     //         //在第j个聚类中搜索所有点
    //     //         //list代表聚类
    //     //         // IVF& list = lists[listids[j]];
    //     //         idx_t ivfId = listids[j];
    //     //         idx_t index = ivfId - info.startIVFId;
    //     //         size_t listSize = listSizes[index];
    //     //         float* codes = listCodes[index].get();
    //     //         size_t* ids = listIds[index].get();

    //     //         //查询点到中心的距离
    //     //         float centroid2query = centroids2query[j];
    //     //         //聚类中的点的数量
    //     //         size_t list_size = list.get_list_size();

    //     //         size_t scan_begin = 0;
    //     //         size_t scan_end = list_size;

    //     //         scaner->scan_codes(scan_begin, scan_end, list_size, list.get_candidate_codes(), list.get_candidate_id(),
    //     //                         list.get_candidate_norms(), centroid2query, list.get_candidate2centroid(),
    //     //                         list.get_sqrt_candidate2centroid(), sub_k, list.get_sub_nearest_IP_id(),
    //     //                         list.get_sub_nearest_IP_dis(), list.get_sub_farest_IP_id(),
    //     //                         list.get_sub_farest_IP_dis(), list.get_sub_nearest_L2_id(),
    //     //                         list.get_sub_nearest_L2_dis(), nullptr, disi, idxi, stats,
    //     //                         centroid_codes.get() + listids[j] * d);
    //     //     }
    //     //     sort_result(METRIC_L2, k, disi, idxi);
    //     //     disi += k;
    //     //     idxi += k;
    //     //     listids += nprobe;
    //     // }
    // }

    void search();
//     void search() {
//         uniWatch.print(format("node {} start search", rank), false);

//         size_t nt = std::min(static_cast<size_t>(omp_get_max_threads()), nq);
//         size_t batch_size = nq / nt;
//         size_t extra = nq % nt;
//         cout << nt << endl;
// #pragma omp parallel for num_threads(nt)
//         for (size_t i = 0; i < nt; i++) {
//             size_t start, end;
//             if (i < extra) {
//                 start = i * (batch_size + 1);
//                 end = start + batch_size + 1;
//             } else {
//                 start = i * batch_size + extra;
//                 end = start + batch_size;
//             }
//             if (start < end) {
//                 // end - start是查询向量的数量，queries + start * d是查询向量的起始位置，distance + start * k
//                 // 是结果存放的起始位置

//                 // cout << format("thread {} {}", omp_get_thread_num(), start) << endl;
//                 MyStopWatch w;
//                 single_thread_search_simple(end - start, querys.get() + start * info.d, k, distances.get() + start * k, labels.get() + start * k);
//                 // w.print(format("single thread {}", i));
//             }
//         }
// // #pragma omp parallel for schedule(dynamic)
// //         for(size_t q = 0; q < nq; q++) {
// //             // size_t nt = omp_get_num_threads();
// //             // cout << nt << endl;
// //             // auto scaner = scaners[omp_get_thread_num()].get();
// //             // scaner->set_query(querys.get() + q * info.d);
// //             // float* simi = distances.get() + q * k;
// //             // idx_t* idxi = labels.get() + q * k;
// //             // float* query = querys.get() + q * info.d;
// //             for (size_t i = 0; i < info.nprobe; i++) {
// //                 idx_t ivfId = listidqueries[q * info.nprobe + i];
// //                 // if(!(ivfId >= info.startIVFId && ivfId < info.startIVFId + info.ivfCount)) {
// //                 //     continue;
// //                 // }
// //                 idx_t index = ivfId - info.startIVFId;
// //                 size_t listSize = listSizes[index];
// //                 // float* codes = listCodes[index].get();
// //                 // size_t* ids = listIds[index].get();
// //                 // scaner->lite_scan_codes(listSize, codes, ids, distances.get() + q * k, labels.get() + q * k);
// //                 for (size_t j = 0; j < listSize; j++) {
// //                     //和每一个待比较向量进行比较，每一个i和一个待比较向量绑定
// //                     // const float* candicate = codes + j * info.d;
// //                     // float dis = calculatedEuclideanDistance(query, candicate, info.d);
// //                     // if (dis < simi[0]) {
// //                     //     //比堆顶
// //                     //     heap_replace_top<METRIC_L2>(k, simi, idxi, dis, ids[j]);
// //                     //     // simi[0] = dis;
// //                     // }
// //                         //比堆顶
// //                     heap_replace_top<METRIC_L2>(k, distances.get(), labels.get(), 0, 0);
// //                         // simi[0] = dis;
// //                 }
// //             }
// //         }
//         uniWatch.print(format("node {} search", rank), false);
//         MPI_Send(distances.get(), k * nq, MPI_FLOAT, 0, 0, MPI_COMM_WORLD); 
//         MPI_Send(labels.get(), k * nq, MPI_INT64_T, 0, 0, MPI_COMM_WORLD); 
//         uniWatch.print(format("node {} send", rank), false);
//     }
    
};
}  // namespace tribase
#endif