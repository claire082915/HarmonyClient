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

    // std::unique_ptr<float[]> heapTops;
    vector<std::unique_ptr<float[]>> distancesForBlocks;
    size_t blockDistancesSize;

    double waitTime = 0;
    double searchTime = 0;

    // size_t presumeNq = 370, presumeK = 100;
    size_t presumeK = 100;

      // Temporary storage for each block's computation result
    class DistanceBufferPool {
        vector<std::unique_ptr<float[]>> distancesForBlocks; // Actual storage buffer
        vector<int> blockIds; // The blockId corresponding to each entry in distancesForBlocks
        vector<bool> used; // Whether each distancesForBlocks entry is currently in use
        size_t spareBlock = 0; // Number of available buffers
        bool useDynamicAlloc = false; // If true, the index of distancesForBlocks does not represent blockId; otherwise, it does
        int MAXCHUNKSIZE;

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
        queue<Action> waitList; // Whether the action is use or receive

        int generateChunkTag(size_t blockId, int chunkIndex) {
            // Use a large enough multiplier to avoid collision of chunkIndex within a block
            // Also ensure it doesn't exceed MPI_TAG_UB (usually minimum 32767)
            const int chunkMultiplier = 10000;
            int tag = static_cast<int>(blockId) * chunkMultiplier + chunkIndex;
            if (chunkMultiplier < chunkIndex) {
                cerr << "Error: chunkMultiplier < chunkIndex" << chunkMultiplier<< " < " << chunkIndex << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Optional: check if tag exceeds MPI_TAG_UB (max supported tag)
            int tag_ub;
            int found;
            MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_ub, &found);
            if (found && tag > tag_ub) {
                cerr << "Error: MPI tag exceeds MPI_TAG_UB: " << tag << " > " << tag_ub << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            return tag;
        }

        void IRecvSplit(float* buffer, size_t blockId, idx_t compareSize, size_t sender, vector<MPI_Request>& reqs) {
            if(reqs.size() != 0) {
                cerr << "what the fuck" << endl;
            }
            if (compareSize > MAXCHUNKSIZE) {
                cout << YELLOW << format("WARNING: block {} compare size {} too big, split into {}", blockId, compareSize, (compareSize + MAXCHUNKSIZE - 1) / MAXCHUNKSIZE) << RESET << endl;

                idx_t sizeToReceive = compareSize;
                int chunkIndex = 0;
                int offset = 0;

                while (sizeToReceive > MAXCHUNKSIZE) {
                    MPI_Request req;
                    int tag = generateChunkTag(blockId, chunkIndex);

                    MPI_Irecv(buffer + offset, MAXCHUNKSIZE, MPI_FLOAT, sender, tag, MPI_COMM_WORLD, &req);
                    // cout << format("IRecvSplit worker {} tag {}", worker->rank, tag) << endl; 
                    reqs.push_back(req);

                    sizeToReceive -= MAXCHUNKSIZE;
                    offset += MAXCHUNKSIZE;
                    chunkIndex++;
                }

                // Last chunk
                MPI_Request req;
                int tag = generateChunkTag(blockId, chunkIndex);
                MPI_Irecv(buffer + offset, sizeToReceive, MPI_FLOAT, sender, tag, MPI_COMM_WORLD, &req);
                // cout << format("IRecvSplit worker {} last tag {}", worker->rank, tag) << endl; 
                reqs.push_back(req);

            } else {
                MPI_Request req;
                int tag = generateChunkTag(blockId, 0);
                MPI_Irecv(buffer, compareSize, MPI_FLOAT, sender, tag, MPI_COMM_WORLD, &req);
                reqs.push_back(req);
            }
        }

        Worker* worker;

    public:
        DistanceBufferPool(const InitInfo& info, Worker* worker) : worker(worker) {
            MAXCHUNKSIZE = INT_MAX;
            distancesForBlocks = vector<std::unique_ptr<float[]>>();
            waitList = queue<Action>();
            try {
                for (size_t i = 0; i < info.blockCount; i++) {
                    distancesForBlocks.push_back(std::make_unique<float[]>(info.presumeBlockDistancesSize));
                }
                cout << YELLOW << format("use Normal Alloc") << RESET << endl;
            } catch (const std::bad_alloc& e) { 
                useDynamicAlloc = true;
                blockIds = vector<int>(distancesForBlocks.size(), -1);
                used = vector<bool>(distancesForBlocks.size(), false);
                spareBlock = distancesForBlocks.size();
                cout << YELLOW << format("use Dynamic Alloc") << RESET << endl;
            }
            cout << format("buffer size {}", distancesForBlocks.size()) << endl;
        }

        float* getBuffer(size_t blockId) { // Get the buffer for a specific blockId
            if(!useDynamicAlloc) {
                return distancesForBlocks[blockId].get();
            }
            for (int i = 0; i < distancesForBlocks.size(); i++) {
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
                    for (int i = 0; i < distancesForBlocks.size(); i++) {
                        if(used[i] == false) {
                            used[i] = true;
                            blockIds[i] = blockId;
                            IRecvSplit(distancesForBlocks[i].get(), blockId, compareSize, sender, reqs);
                            cout << format("Irecv {}", blockId) << endl;
                            spareBlock--;
                            return true;
                        }
                    }
                }
                return false;
            }
        }

        void ISendSplit(const float* buffer, size_t blockId, idx_t compareSize, size_t receiver, std::vector<MPI_Request>& reqs) {
            cout << YELLOW << format("WARNING: block {} ISendSplit, split into {}", blockId, (compareSize + MAXCHUNKSIZE - 1) / MAXCHUNKSIZE) << RESET << endl;
            idx_t sizeLeft = compareSize;
            idx_t offset = 0;
            int chunkIndex = 0;

            while (sizeLeft > 0) {
                int chunkSize = static_cast<int>(std::min<idx_t>(sizeLeft, MAXCHUNKSIZE));
                int tag = generateChunkTag(blockId, chunkIndex);
                MPI_Request req;

                MPI_Isend(buffer + offset, chunkSize, MPI_FLOAT, receiver, tag, MPI_COMM_WORLD, &req);
                // cout << format("ISendSplit worker {} tag {}", worker->rank, tag) << endl; 
                reqs.push_back(req);

                offset += chunkSize;
                sizeLeft -= chunkSize;
                chunkIndex++;
            }
        }

        bool use(size_t blockId) { // Request allocation of buffer for blockId, return whether successful
            if(!useDynamicAlloc) {
                return true;
            } else {
                if(spareBlock <= 0) {
                    waitList.push(Action(blockId));
                    return false;
                } 
                for (int i = 0; i < distancesForBlocks.size(); i++) {
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

        bool releaseBuffer(size_t blockId) { // Release the buffer for a given blockId
            if(!useDynamicAlloc) {
                return true;
            } else {
                for (int i = 0; i < distancesForBlocks.size(); i++) {
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

    // Each entry in the vector corresponds to a block
    // vector<MPI_Request> infoRequests;
    vector<vector<MPI_Request>> disRequests;
    vector<vector<MPI_Request>> sendRequests;
    // vector<MPI_Request> sendRequests;
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
        // this->querys = std::make_unique<float[]>(nq * info.block_dim);
        copy_n_partial_vector(querys, this->querys.get(), info.d, info.block_dim, info.block_dim * (rank - 1), nq);
    }

    void search(bool cut);

    idx_t getTotalQueryCompareSize(size_t blockId) {
        size_t queryStart = blockId * blockSize;
        idx_t totalQueryCompareSize = queryCompareSizePreSum[queryStart + blockSize] - queryCompareSizePreSum[queryStart];
        // if(totalQueryCompareSize > INT_MAX) {
        //     // cerr << RED << "increase block size" << RESET << endl;
        //     cerr << YELLOW << "warning : totalQueryCompareSize > INT_MAX" << RESET << endl;
        //     // cout << totalQueryCompareSize << endl;
        //     // throw std::invalid_argument("increase block size");
        // }
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
    std::unique_ptr<float[]> heapTops;

    InitInfo info;

    MyStopWatch uniWatch;

    std::unique_ptr<Index> index;
    // Index* index;

    double waitTime = 0, searchTime = 0;

    bool cut = false;

    // void init(int rank, tribase::Index* index) {
    void init(int rank);
    
    
    void single_thread_search_simple(size_t n, const float* queries, size_t k, float* distances, idx_t* labels, idx_t* listidqueries, float* heapTops);
    void single_thread_search_fast(size_t n, const float* queries, size_t k, float* distances, idx_t* labels, idx_t* listidqueries);
    
    void search(bool cut);

    
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

        // Default constructor with all values set to zero
        // InitInfo()
        //     : d(0), block_dim(0), workerCount(0), nlist(0), blockCount(0), nprobe(0), nb(0),
        //     presumeBlockDistancesSize(0), groupCount(0), teamCount(0), teamSize(0), 
        //     startIVFId(0), ivfCount(0), teamId(0), rankInsideTeam(0) {}

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

    vector<vector<std::unique_ptr<float[]>>> distancesForBlocks;
    size_t blockDistancesSize;
    double waitTime = 0, searchTime = 0;
    size_t presumeK = 100;
    // size_t presumeNq = 370, presumeK = 100;
    

    // idx_t presumeBlockDistancesSize;
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
    bool minorCut = false;

    std::unique_ptr<float[]> heapTops;

    vector<double> skipRates;


    void init(int rank, bool blockSend);
    bool shouldSendHeap(size_t blockId) {
        return blockSearchOrder.sendNextWorker[info.rankInsideTeam][blockId] == 0;
    }
    void receiveQuery();
   
    void addQuerys(const float* querys, size_t nq) {
        // this->querys = std::make_unique<float[]>(nq * info.block_dim);
        copy_n_partial_vector(querys, this->querys.get(), info.d, info.block_dim, info.block_dim * (info.rankInsideTeam - 1), nq);
    }

    void search(bool cut, bool minorCut);
    

    idx_t getBlockQueryCompareSize(size_t groupId, size_t blockId) {
        size_t queryStart = getQueryOffset(groupId, blockId);
        idx_t totalQueryCompareSize = queryCompareSizePreSum[queryStart + blockSize] - queryCompareSizePreSum[queryStart];
        // if((double)queryCompareSizePreSum[queryStart + blockSize] - queryCompareSizePreSum[queryStart] > INT_MAX) {
        //     cerr << RED << "increase block size" << RESET << endl;
        //     throw std::invalid_argument("increase block size");
        // }
        return totalQueryCompareSize;
    }
    idx_t getQueryOffset(size_t groupId, size_t blockId) {
        return groupId * groupSize + blockId * blockSize;
    }

    void postSearch() {
        if(cut) {
            MPI_Send(skipRates.data(), info.blockCount, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD); 
        }
    }

    idx_t totalSkip = 0;
    idx_t totalCompare = 0;

private:
    void searchGroup(idx_t groupId);
    void searchBlock(size_t blockId, size_t groupId);
    void reset() {
        // for (size_t i = 0; i < info.blockCount; i++) {
        //     init_result(METRIC_L2, blockSize * k, distanceHeap[i].get(), idHeap[i].get());
        // }
        //distancefornblocks
        disRequests = vector<vector<MPI_Request>>(info.blockCount);
        for(int i = 0; i < disRequests.size(); i++) {
            disRequests[i] = vector<MPI_Request>(1);
        }
        // waitTime = searchTime = 0;
    }
    size_t getSender(idx_t groupId, idx_t blockId);
    size_t getReceiver(idx_t groupId, idx_t blockId);
    void single_thread_searchBlock(size_t n, size_t blockId, size_t groupId, float* queries, float* distanceBuffer, float* simi, idx_t* idxi, idx_t* listidqueries);
};



}  // namespace tribase
#endif