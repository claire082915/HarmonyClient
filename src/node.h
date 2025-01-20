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
#include "order.h"

namespace harmony {

class Index;

using namespace std;

const int presumeNq = 10000;

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
    // Represents the information that needs to be included in a search request

    size_t rank = 0;
    size_t blockSize = 0;
    size_t nq = 0;
    size_t k = 0;

    std::unique_ptr<size_t[]> listSizes;         // Number of vectors in each of the nlist clusters

    std::unique_ptr<float[]> querys;             // nq query vectors, each with block_dim dimensions

    std::unique_ptr<idx_t[]> blockSearchOrder;   // Order of blocks to search, element i is the blockId to search at the i-th step

    std::unique_ptr<idx_t[]> listidqueries;      // nq * nprobe query vectors' corresponding cluster IDs

    std::unique_ptr<idx_t[]> queryCompareSize;   // Number of base vectors to compare for each query vector

    // The prefix sum of queryCompareSize is stored in queryCompareSizePreSum to calculate offset for each query
    std::unique_ptr<idx_t[]> queryCompareSizePreSum;

    vector<std::unique_ptr<float[]>> distanceHeap;
    vector<std::unique_ptr<idx_t[]>> idHeap;

    vector<std::unique_ptr<float[]>> distancesForBlocks;

    size_t blockDistancesSize;

    double waitTime = 0;
    double searchTime = 0;

    // size_t presumeNq = 370, presumeK = 100;

    size_t presumeK = 100;

    InitInfo info;

    void addIVFs(vector<std::unique_ptr<float[]>>& listCodesBuffer);


    // Each entry in the vector corresponds to a block
    // vector<MPI_Request> infoRequests;
    vector<vector<MPI_Request>> disRequests;
    vector<MPI_Request> sendRequests;
    vector<MPI_Request> sendDistanceRequests;
    vector<MPI_Request> sendIdRequests;
        
    // vector<MPI_Status> statuses;

    std::unique_ptr<idx_t[]> sendNextWorker; // When blockId i is received, send it to the worker with rank sendNextWorker[i]; sendNextWorker[i] = 0 means send it to the master
    std::unique_ptr<idx_t[]> recvPrevWorker; // BlockId i should be received from the worker with rank recvPrevWorker[i]; recvPrevWorker[i] = 0 means no need to receive, and calculation can be done directly

    MyStopWatch uniWatch;

    std::unique_ptr<Index> index;

    bool blockSend = false;
    bool cut = false;
    std::unique_ptr<float[]> heapTops;
    vector<double> skipRates;

    void init(int rank, bool blockSend);

    bool shouldSendHeap(size_t blockId) {
        return  sendNextWorker[blockId] == 0;
    }
    
    void addQuerys(const float* querys, size_t nq) {
        copy_n_partial_vector(querys, this->querys.get(), info.d, info.block_dim, info.block_dim * (rank - 1), nq);
    }

    void search(bool cut);

    idx_t getTotalQueryCompareSize(size_t blockId) {
        size_t queryStart = blockId * blockSize;
        idx_t totalQueryCompareSize = queryCompareSizePreSum[queryStart + blockSize] - queryCompareSizePreSum[queryStart];
        return totalQueryCompareSize;
    }

    void postSearch() {
        if(cut) {
            MPI_Send(skipRates.data(), info.blockCount, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD); 
        }
    }

    idx_t totalSkip = 0;

    idx_t totalCompare = 0;

    void searchBlock(size_t blockId, bool cut);
    
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
    // Represents the information needed in a single search request

    size_t rank = 0;
    size_t nq = 0;
    size_t k = 0;
    std::unique_ptr<size_t[]> listSizes;         // Number of vectors in each of the nlist clusters
    vector<std::unique_ptr<float[]>> listCodes;  // Vectors in each of the nlist clusters
    vector<std::unique_ptr<size_t[]>> listIds;  // IDs of the vectors in each of the nlist clusters
    std::unique_ptr<float[]> querys;             // nq query vectors, each with a dimension of block_dim
    std::unique_ptr<idx_t[]> listidqueries;      // nq * nprobe cluster IDs corresponding to similar query vectors
    
    std::unique_ptr<float[]> distances;
    std::unique_ptr<idx_t[]> labels;

    InitInfo info;

    MyStopWatch uniWatch;

    std::unique_ptr<Index> index;
    // Index* index;

    double waitTime = 0, searchTime = 0;

    // void init(int rank, tribase::Index* index) {
    void init(int rank);
    
    void single_thread_search_simple(size_t n, const float* queries, size_t k, float* distances, idx_t* labels, idx_t* listidqueries);

    void single_thread_search_fast(size_t n, const float* queries, size_t k, float* distances, idx_t* labels, idx_t* listidqueries);

    void search();
   
    
};


class GroupWorker {
private:
public:
    struct InitInfo {
        const size_t d, block_dim, workerCount;
        const size_t nlist;
        const size_t blockCount, nprobe, nb;
        const idx_t presumeBlockDistancesSize;
        const size_t groupCount, teamCount, teamSize;
        const size_t startIVFId, ivfCount;
        const size_t teamId, rankInsideTeam;

        // Constructor with default values for all fields
        InitInfo(size_t d = 0, size_t block_dim = 0, size_t workerCount = 0, size_t nlist = 0, size_t blockCount = 0,
                size_t nprobe = 0, size_t nb = 0, idx_t presumeBlockDistancesSize = 0, size_t groupCount = 0,
                size_t teamCount = 0, size_t teamSize = 0, size_t startIVFId = 0, size_t ivfCount = 0, 
                size_t teamId = 0, size_t rankInsideTeam = 0)
            : d(d), block_dim(block_dim), workerCount(workerCount), nlist(nlist), blockCount(blockCount), 
            nprobe(nprobe), nb(nb), presumeBlockDistancesSize(presumeBlockDistancesSize), groupCount(groupCount), 
            teamCount(teamCount), teamSize(teamSize), startIVFId(startIVFId), ivfCount(ivfCount), 
            teamId(teamId), rankInsideTeam(rankInsideTeam) {}

        // Print method
        void print() const {
            cout << GREEN << "InitInfo: [";
            cout << "d:" << d;
            cout << ", nlist:" << nlist;
            cout << ", nprobe:" << nprobe;
            cout << ", worker:" << workerCount;
            cout << ", block_dim:" << block_dim;
            cout << ", nb:" << nb;
            cout << ", presume:" << presumeBlockDistancesSize / 1000000000 << "GB"; // Presumed size in GB
            cout << ", groupCount:" << groupCount;
            cout << ", teamCount:" << teamCount;
            cout << ", teamSize:" << teamSize;
            cout << ", startIVFId:" << startIVFId;
            cout << ", ivfCount:" << ivfCount;
            cout << ", teamId:" << teamId;
            cout << ", rankInsideTeam:" << rankInsideTeam;
            cout << "]";
            cout << "\033[0m" << endl;  // Reset color after printing
        }
    };
    
    struct Group {
    };
    
    static int getTag(size_t groupId, size_t blockId, size_t blockCount) {
        return groupId * blockCount + blockId;
    }

    static int getDistanceHeapTag(size_t groupId) {
        return groupId * 2;
    }

    static int getIdHeapTag(size_t groupId) {
        return groupId * 2 + 1;
    }

    size_t rank = 0;
    size_t blockSize = 0;
    size_t groupSize = 0;
    size_t nq = 0, k = 0;

    std::unique_ptr<size_t[]> listSizes;         // Number of vectors in nlist clusters
    std::unique_ptr<float[]> querys;             // nq query vectors, dimension is block_dim
    std::unique_ptr<idx_t[]> listidqueries;      // nq * nprobe query vectors' corresponding cluster IDs
    std::unique_ptr<idx_t[]> queryCompareSize;   // nq * nprobe query vectors' corresponding cluster IDs
    std::unique_ptr<idx_t[]> queryCompareSizePreSum; // Used to calculate the offset of distancesForQueries for the q-th vector

    vector<std::unique_ptr<float[]>> distanceHeap; // Each pointer in the vector represents the heap of all blocks in a group
    vector<std::unique_ptr<idx_t[]>> idHeap;  // Heap of IDs for each group

    vector<vector<std::unique_ptr<float[]>>> distancesForBlocks; // Distances for blocks

    size_t blockDistancesSize;
    double waitTime = 0, searchTime = 0;
    size_t presumeK = 100;

    InitInfo info;

    void addIVFs(vector<std::unique_ptr<float[]>>& listCodesBuffer);

    vector<vector<MPI_Request>> disRequests; // Each element in the vector corresponds to a block
    vector<MPI_Request> sendRequests;
    vector<MPI_Request> sendDistanceRequests;
    vector<MPI_Request> sendIdRequests;
    
    SearchOrder groupSearchOrder, blockSearchOrder;

    MyStopWatch uniWatch;

    std::unique_ptr<Index> index;

    bool blockSend = false;
    bool cut = false;

    std::unique_ptr<float[]> heapTops;

    vector<double> skipRates;

    void init(int rank, bool blockSend);

    // Method to check if the heap for a block should be sent
    bool shouldSendHeap(size_t blockId) {
        return blockSearchOrder.sendNextWorker[info.rankInsideTeam][blockId] == 0;
    }

    void receiveQuery();
   
    void addQuerys(const float* querys, size_t nq) {
        // Copy query vectors to class member
        copy_n_partial_vector(querys, this->querys.get(), info.d, info.block_dim, info.block_dim * (info.rankInsideTeam - 1), nq);
    }

    void search(bool cut);
    
    // Get the total comparison size for the given block and group
    idx_t getBlockQueryCompareSize(size_t groupId, size_t blockId) {
        size_t queryStart = getQueryOffset(groupId, blockId);
        idx_t totalQueryCompareSize = queryCompareSizePreSum[queryStart + blockSize] - queryCompareSizePreSum[queryStart];
        return totalQueryCompareSize;
    }

    idx_t getQueryOffset(size_t groupId, size_t blockId) {
        return groupId * groupSize + blockId * blockSize;
    }

    void postSearch() {
        // Send skip rates if cut is enabled
        if(cut) {
            MPI_Send(skipRates.data(), info.blockCount, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD); 
        }
    }

    idx_t totalSkip = 0;

    idx_t totalCompare = 0;

private:

    // Search methods for group and block
    void searchGroup(idx_t groupId);
    void searchBlock(size_t blockId, size_t groupId);

    void reset() {
        disRequests = vector<vector<MPI_Request>>(info.blockCount);
        for(int i = 0; i < disRequests.size(); i++) {
            disRequests[i] = vector<MPI_Request>(1);
        }
    }

    size_t getSender(idx_t groupId, idx_t blockId);
    size_t getReceiver(idx_t groupId, idx_t blockId);

    void single_thread_searchBlock(size_t n, size_t blockId, size_t groupId, float* queries, float* distanceBuffer, float* simi, idx_t* idxi, idx_t* listidqueries);

};




}  // namespace tribase
#endif