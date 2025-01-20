#ifndef ORDER_H
#define ORDER_H
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <vector>

#include "utils.h"

namespace harmony {

// A class that represents the search order for a certain number of workers and a certain number of blocks
class SearchOrder {

public:

    // workerSearchOrder[i] represents a vector of block indices
    // Each worker has a list of blocks to process in a specific sequence
    std::vector<std::vector<idx_t>> workerSearchOrder;    // The order of blocks each worker processes sequentially
    
    // searchedWorkerOrder stores the sequence of workers processing each block
    std::vector<std::vector<idx_t>> searchedWorkerOrder;  // The order of workers that process each block sequentially

    // sendNextWorker[i][j] indicates the next worker to which block j is sent by worker i
    std::vector<std::vector<idx_t>> sendNextWorker;
    
    // recvPrevWorker[i][j] indicates the previous worker from which block j is received by worker i
    std::vector<std::vector<idx_t>> recvPrevWorker;

    size_t workerCount;  // Total number of workers
    size_t blockCount;   // Total number of blocks

    SearchOrder() {}

    // worker rank starts at 1, block ID starts at 0
    // orderOptimize enables optimization about search order
    SearchOrder(size_t workerCount, size_t blockCount, bool orderOptimize)
        : workerCount(workerCount), blockCount(blockCount) {
        
        // Initialize workerSearchOrder with an extra index for easier 1-based indexing
        workerSearchOrder = std::vector<std::vector<idx_t>>(workerCount + 1);
        for (size_t i = 1; i <= workerCount; i++) {
            workerSearchOrder[i] = std::vector<idx_t>(blockCount);
        }

        // Compute the block processing order for each worker
        if (orderOptimize) {
            size_t gap = (blockCount + workerCount - 1) / workerCount;  // Calculate gap for order optimization
            for (size_t i = 1; i <= workerCount; i++) {
                for (size_t block = 0; block < blockCount; block++) {
                    size_t order = (block + (gap * (i - 1))) % blockCount;  // Compute the optimized order
                    workerSearchOrder[i][order] = block;  // Assign block to worker's order
                }
            }
        } else {
            // Assign block processing order without optimization
            for (size_t i = 1; i <= workerCount; i++) {
                for (size_t order = i - 1; order < i - 1 + blockCount; order++) {
                    workerSearchOrder[i][order % blockCount] = order - (i - 1);  // Cycle through block IDs
                }
            }
        }

        // Initialize searchedWorkerOrder, representing the sequence of workers for each block
        searchedWorkerOrder = std::vector<std::vector<idx_t>>(blockCount);
        for (size_t i = 0; i < blockCount; i++) {
            searchedWorkerOrder[i] = std::vector<idx_t>();  // Start with an empty vector for each block
        }

        // Populate searchedWorkerOrder based on workerSearchOrder
        for (size_t order = 0; order < blockCount; order++) {
            for (size_t rank = 1; rank <= workerCount; rank++) {
                size_t block = workerSearchOrder[rank][order];  // Get the block ID in the current order
                searchedWorkerOrder[block].push_back(rank);  // Add the rank to the block's worker sequence
            }
        }

        // Initialize sendNextWorker and recvPrevWorker for communication between workers
        sendNextWorker = std::vector<std::vector<idx_t>>(workerCount + 1);
        recvPrevWorker = std::vector<std::vector<idx_t>>(workerCount + 1);
        for (size_t i = 1; i <= workerCount; i++) {
            sendNextWorker[i] = std::vector<idx_t>(blockCount, 0);  // Default all to 0
            recvPrevWorker[i] = std::vector<idx_t>(blockCount, 0);  // Default all to 0
        }

        // Determine communication links between workers for each block
        for (size_t block = 0; block < blockCount; block++) {
            for (size_t rankOrder = 0; rankOrder < workerCount - 1; rankOrder++) {
                size_t senderRank = searchedWorkerOrder[block][rankOrder];  // Worker sending the block
                size_t recvRank = searchedWorkerOrder[block][rankOrder + 1];  // Worker receiving the block
                sendNextWorker[senderRank][block] = recvRank;  // Assign the next worker
                recvPrevWorker[recvRank][block] = senderRank;  // Assign the previous worker
            }
        }
    }

    // Retrieve the worker rank from which a block is received
    size_t getRecvPrevWorker(size_t workerRank, size_t blockId) {
        return recvPrevWorker[workerRank][blockId];
    }

    // Retrieve the worker rank to which a block should be sent
    size_t getSendNextWorker(size_t workerRank, size_t blockId) {
        return sendNextWorker[workerRank][blockId];
    }

    // Print the search order and communication information
    void print() {
        std::cout << "Order in which each worker processes blocks" << std::endl;
        for (size_t i = 1; i <= workerCount; i++) {
            printVector(workerSearchOrder[i], BLUE);  // Print block sequence for each worker
        }

        std::cout << "Order in which each block is processed by workers" << std::endl;
        for (size_t i = 0; i < blockCount; i++) {
            printVector(searchedWorkerOrder[i], GREEN);  // Print worker sequence for each block
        }

        std::cout << "For each worker, the worker to send the block to next" << std::endl;
        for (size_t rank = 1; rank <= workerCount; rank++) {
            printVector(sendNextWorker[rank], BLUE);  // Print next worker for each block
        }

        std::cout << "For each worker, the sender from which a block is received" << std::endl;
        for (size_t rank = 1; rank <= workerCount; rank++) {
            printVector(recvPrevWorker[rank], BLUE);  // Print previous worker for each block
        }
    }
};

}  // namespace harmony
#endif
