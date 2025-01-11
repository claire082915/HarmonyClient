#ifndef ORDER_H
#define ORDER_H
#include <numeric>
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <vector>
#include "utils.h"

namespace tribase {

using namespace std;
class SearchOrder {
public:
    std::vector<std::vector<idx_t>> workerSearchOrder; //每一个worker要顺序计算哪些blockid
    std::vector<std::vector<idx_t>> searchedWorkerOrder; //每一个块要陆续被哪些worker算
    vector<vector<idx_t>> sendNextWorker, recvPrevWorker;
    size_t workerCount, blockCount;
    //worker的rank从1开始，block的id从0开始
    SearchOrder() {}
    SearchOrder(size_t workerCount, size_t blockCount, bool orderOptimize) : workerCount(workerCount), blockCount(blockCount){

        workerSearchOrder = vector<vector<idx_t>>(workerCount + 1);
        for (size_t i = 1; i <= workerCount; i++) {
            workerSearchOrder[i] = vector<idx_t>(blockCount);
        }

        if(orderOptimize) {
            size_t gap = (blockCount + workerCount - 1) / workerCount;
            for (size_t i = 1; i <= workerCount; i++) {
                for (size_t block = 0; block < blockCount; block++) {
                    size_t order = (block + (gap * (i - 1))) % blockCount;
                    workerSearchOrder[i][order] = block;
                }
            }
        } else {
            for (size_t i = 1; i <= workerCount; i++) {
                for (size_t order = i - 1; order < i - 1 + blockCount; order++) {
                    workerSearchOrder[i][order % blockCount] = order - (i - 1);
                }
            } 
        }

        //对于每一个块，其搜索的顺序，即一系列rank
        searchedWorkerOrder = vector<vector<idx_t>>(blockCount);
        for (size_t i = 0; i < blockCount; i++) {
            searchedWorkerOrder[i] = vector<idx_t>();
        }
        for (size_t order = 0; order < blockCount; order++) {
            for(size_t rank = 1; rank <= workerCount; rank++) {
                size_t block = workerSearchOrder[rank][order]; 
                searchedWorkerOrder[block].push_back(rank);
            }
        }

        //每一个worker，应该将某个block传递给下一个worker的rank
        sendNextWorker = vector<vector<idx_t>>(workerCount + 1);
        recvPrevWorker = vector<vector<idx_t>>(workerCount + 1);
        for (size_t i = 1; i <= workerCount; i++) {
            sendNextWorker[i] = vector<idx_t>(blockCount, 0);
            recvPrevWorker[i] = vector<idx_t>(blockCount, 0);
        }
        for (size_t block = 0; block < blockCount; block++) {
            for (size_t rankOrder = 0; rankOrder < workerCount - 1; rankOrder++) {
                size_t senderRank = searchedWorkerOrder[block][rankOrder];
                size_t recvRank = searchedWorkerOrder[block][rankOrder + 1];
                sendNextWorker[senderRank][block] = recvRank;  
                recvPrevWorker[recvRank][block] = senderRank;  
            }
        }
    }

    size_t getRecvPrevWorker(size_t workerRank, size_t blockId) {
        return recvPrevWorker[workerRank][blockId];
    }
    size_t getSendNextWorker(size_t workerRank, size_t blockId) {
        return sendNextWorker[workerRank][blockId];
    }
    

    void print() {
        cout << "worker计算block的顺序" << endl;
        for (size_t i = 1; i <= workerCount; i++) {
            printVector(workerSearchOrder[i], BLUE);
        }
        cout << "block依次经过的worker" << endl;
        for (size_t i = 0; i < blockCount; i++) {
            printVector(searchedWorkerOrder[i], GREEN);
        }
        cout << "每个worker接收到block之后要发给谁" << endl;
        for(size_t rank = 1; rank <= workerCount; rank++) {
            printVector(sendNextWorker[rank], BLUE);
        }
        cout << "每个worker接收到block时的发送者应该是谁" << endl;
        for(size_t rank = 1; rank <= workerCount; rank++) {
            printVector(recvPrevWorker[rank], BLUE);
        }
    }
};

}  // namespace tribase
#endif