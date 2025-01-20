#include "node.h"

#include "Index.h"
namespace harmony {


void Worker::init(int rank, bool blockSend) {
    // Timer object to measure initialization times.
    MyStopWatch watch(false);

    // Store the rank and blockSend flag for the worker instance.
    this->rank = rank;
    this->blockSend = blockSend;

    // Initialize configuration info by broadcasting InitInfo structure from the root (rank 0) process.
    MPI_Bcast(&info, sizeof(InitInfo), MPI_BYTE, 0, MPI_COMM_WORLD);
    // Uncomment the following line to print the info for debugging.
    // info.print();

    // Initialize the index with parameters from InitInfo.
    index = std::make_unique<Index>(info.d, info.nlist, info.nprobe);

    // Allocate memory for storing the sizes of inverted file lists (IVF) and broadcast the sizes.
    listSizes = std::make_unique<size_t[]>(info.nlist);
    MPI_Bcast(listSizes.get(), info.nlist * sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Initialize IVF list structures in parallel using OpenMP.
#pragma omp parallel for
    for (size_t i = 0; i < info.nlist; i++) {
        index->lists[i].reset(listSizes[i], info.block_dim, 0);  // Reset lists with the appropriate sizes.
    }
    watch.print("lists");

    // Broadcast candidate IDs for each list.
    for (size_t i = 0; i < info.nlist; i++) {
        MPI_Bcast(index->lists[i].candidate_id.get(), listSizes[i] * sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    // Allocate buffers for candidate codes in each list.
    auto listCodesBuffer = std::vector<std::unique_ptr<float[]>>(info.nlist);
    for (size_t i = 0; i < info.nlist; i++) {
        listCodesBuffer[i] = std::make_unique<float[]>(listSizes[i] * info.d);
        MPI_Bcast(listCodesBuffer[i].get(), listSizes[i] * info.d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    // Copy partial vectors from buffers to the index's candidate code arrays.
    for (size_t i = 0; i < info.nlist; i++) {
        copy_n_partial_vector(listCodesBuffer[i].get(), index->lists[i].candidate_codes.get(), 
                              info.d, info.block_dim, (rank - 1) * info.block_dim, listSizes[i]);
    }

    // Broadcast centroid codes and IDs to all processes.
    MPI_Bcast(index->centroid_codes.get(), info.nlist * info.d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(index->centroid_ids.get(), info.nlist, MPI_INT64_T, 0, MPI_COMM_WORLD);

    watch.print("listCodes and listIds");

    // Allocate memory and receive search block order from the root process.
    blockSearchOrder = std::make_unique<idx_t[]>(info.blockCount);
    MPI_Recv(blockSearchOrder.get(), info.blockCount, MPI_INT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Allocate memory for queries and associated data structures.
    index->originalQuery = std::make_unique<float[]>(presumeNq * info.d);  // Store original query vectors.
    this->querys = std::make_unique<float[]>(presumeNq * info.block_dim);  // Store partial query vectors.
    listidqueries = std::make_unique<idx_t[]>(presumeNq * info.nprobe);    // IDs of the nearest nprobe clusters.
    heapTops = std::make_unique<float[]>(presumeNq);                       // Heap tops for each query.
    queryCompareSize = std::make_unique<idx_t[]>(presumeNq);               // Sizes for query comparisons.
    queryCompareSizePreSum = std::make_unique<idx_t[]>(presumeNq + 1);     // Prefix sums of comparison sizes.
    sendNextWorker = std::make_unique<idx_t[]>(info.blockCount);           // IDs for the next worker to send data.
    recvPrevWorker = std::make_unique<idx_t[]>(info.blockCount);           // IDs for the previous worker to receive from.

    // Allocate heaps for distances and IDs used during search.
    distanceHeap = std::vector<std::unique_ptr<float[]>>(info.blockCount);
    idHeap = std::vector<std::unique_ptr<idx_t[]>>(info.blockCount);
    for (size_t i = 0; i < info.blockCount; i++) {
        distanceHeap[i] = std::make_unique<float[]>(presumeK * presumeNq);
        idHeap[i] = std::make_unique<idx_t[]>(presumeK * presumeNq);
        init_result(METRIC_L2, presumeNq * presumeK, distanceHeap[i].get(), idHeap[i].get());
    }

    // Allocate buffers for distances for each block.
    blockDistancesSize = 2 * presumeNq / info.blockCount * info.nb;
    distancesForBlocks = std::vector<std::unique_ptr<float[]>>(info.blockCount);
    for (size_t i = 0; i < info.blockCount; i++) {
        distancesForBlocks[i] = std::make_unique<float[]>(blockDistancesSize);
    }

    // Initialize MPI request arrays for communication.
    disRequests = std::vector<std::vector<MPI_Request>>(info.blockCount);
    for (size_t i = 0; i < disRequests.size(); i++) {
        disRequests[i] = std::vector<MPI_Request>(1);
    }
    sendRequests = std::vector<MPI_Request>(info.blockCount);
    sendDistanceRequests = std::vector<MPI_Request>(info.blockCount);
    sendIdRequests = std::vector<MPI_Request>(info.blockCount);

    // Initialize skip rates for each block.
    skipRates = std::vector<double>(info.blockCount, 0.0);

    // Receive the worker IDs for sending and receiving data.
    MPI_Recv(sendNextWorker.get(), info.blockCount, MPI_INT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(recvPrevWorker.get(), info.blockCount, MPI_INT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void Worker::search(bool cut) {
    this->cut = cut;

    MPI_Barrier(MPI_COMM_WORLD);  // Corresponds to the barrier at the end of preSearch

    uniWatch = MyStopWatch(false, "uniWatch", MAG);

    uniWatch.print(format("node {} search() start", rank), false);
    // uniWatch.print(format("node {} cross barrier", rank), false);

    // nq, queries
    MPI_Bcast(&nq, sizeof(nq), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, sizeof(k), MPI_BYTE, 0, MPI_COMM_WORLD);
    if ((double)nq * k > presumeK * presumeNq) {
        cerr << "presumeNq * k is too small" << endl;
        exit(1);
    }
    if (nq > presumeNq) {
        cerr << "presumeNq is too small" << endl;
        exit(1);
    }
    uniWatch.print(format("node {} nq k {} ", rank, k), false);
    // MPI_Bcast(querysBuffer.get(), nq * info.d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(index->originalQuery.get(), nq * info.d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    uniWatch.print(format("node {} querys", rank), false);
    addQuerys(index->originalQuery.get(), nq);
    // addQuerys(querysBuffer.get(), nq);
    uniWatch.print(format("node {} addQuerys", rank), false);

    // Get the nearest nprobe centroids for each query
    // listidqueries = std::make_unique<idx_t[]>(nq * info.nprobe);  // IDs of the nearest nprobe centroids
    MPI_Bcast(listidqueries.get(), nq * info.nprobe, MPI_INT64_T, 0, MPI_COMM_WORLD);
    uniWatch.print(format("node {} listidqueries", rank), false);

    // queryCompareSize, queryCompareSizePreSum
    MPI_Bcast(queryCompareSize.get(), nq, MPI_INT64_T, 0, MPI_COMM_WORLD);
    // queryCompareSizePreSum = std::make_unique<size_t[]>(nq + 1);
    MPI_Bcast(queryCompareSizePreSum.get(), (nq + 1), MPI_INT64_T, 0, MPI_COMM_WORLD);
    uniWatch.print(format("node {} queryCompareSize", rank), false);

    // Maximum heap
    MPI_Bcast(heapTops.get(), nq, MPI_FLOAT, 0, MPI_COMM_WORLD);
    uniWatch.print(format("node {} heapTops", rank), false);

    // Other initializations
    blockSize = nq / info.blockCount;

    // watch.print(format("Node {} Init", rank));
    uniWatch.print(format("node {} finish Receiving", rank), false);

    MyStopWatch totalWatch(true, "total search watch");

    int searchedBlockCount = 0;
    vector<bool> isBlockSearched = vector<bool>(info.blockCount);

    // Allocate memory for the blocks that can be computed
    for (size_t blockId = 0; blockId < info.blockCount; blockId++) {
        if (recvPrevWorker[blockId] == 0) {
            // Directly mark as used
            //  distanceBufferPool->use(blockId);
        }
    }
    // Receive blocks from other workers if necessary
    for (size_t blockId = 0; blockId < info.blockCount; blockId++) {
        if (recvPrevWorker[blockId] != 0) {
            size_t sender = recvPrevWorker[blockId];
            // tag as blockId
            MPI_Irecv(distancesForBlocks[blockId].get(), getTotalQueryCompareSize(blockId), MPI_FLOAT, sender, blockId,
                      MPI_COMM_WORLD, &disRequests[blockId][0]);
            // MPI_Irecv(distanceBufferPool->getBuffer(blockId), getTotalQueryCompareSize(blockId), MPI_FLOAT, sender,
            // blockId, MPI_COMM_WORLD, &disRequests[blockId]); bool suc = distanceBufferPool->IRecv(blockId,
            // getTotalQueryCompareSize(blockId), sender, disRequests[blockId]); if(!suc) {
            //     break;
            // }
            // cout << format("node({}) waiting for block({}) from node({})", rank, blockId, sender) << endl;
        }
    }

    // Continue searching while blocks are not finished
    MyStopWatch watch(true);
    idx_t failCount = 0;
    while (searchedBlockCount < info.blockCount)
        for (size_t blockId = 0; blockId < info.blockCount; blockId++) {
            if (isBlockSearched[blockId]) {
                continue;
            }
            if (recvPrevWorker[blockId] == 0) {
                // if(rank == 1) {

                // searchBlock(blockId, cut);
                searchBlock(blockId, cut);
                isBlockSearched[blockId] = true;
                searchedBlockCount++;

                watch.reset();

                // vector<size_t> unFinished;
                // vector<size_t> finished;
                // for(int i = 0; i < info.blockCount; i++) {
                //     if(isBlockSearched[i]) {
                //         int flag;
                //         MPI_Test(&sendRequests[i], &flag, MPI_STATUS_IGNORE);
                //         if(flag == false) {
                //             unFinished.push_back(i);
                //         } else {
                //             finished.push_back(i);
                //         }
                //     }
                // }
                // printVector(unFinished, RED, format("node{} unfinished", rank));
                // printVector(finished, GREEN, format("node{} finished", rank));

                if (blockSend && !shouldSendHeap(blockId)) {
                    MPI_Wait(&sendRequests[blockId], MPI_STATUS_IGNORE);
                    waitTime += watch.watch.elapsedSeconds();
                    watch.print(format("node({}) block {} MPI_Wait", rank, blockId));
                }

                // }

            } else {
                int isReceived;
                MPI_Status stat;
                bool testFail = false;
                for (auto& req : disRequests[blockId]) {
                    MPI_Test(&req, &isReceived, &stat);
                    if (!isReceived) {
                        testFail = true;
                        break;
                    }
                }
                if (!testFail) {
                    // cout << GREEN << format("node({}) received block({}) from node({})",rank, blockId,
                    // stat.MPI_SOURCE) << RESET << endl;
                    waitTime += watch.watch.elapsedSeconds();
                    watch.print(format("node {} waiting , now received block {} fail {}", rank, blockId, failCount));
                    uniWatch.print(format("node {} received block {} Test failed count {}", rank, blockId, failCount),
                                   false);

                    searchBlock(blockId, cut);

                    searchedBlockCount++;
                    isBlockSearched[blockId] = true;
                    failCount = 0;

                    watch.reset();
                    if (blockSend && !shouldSendHeap(blockId)) {
                        MPI_Wait(&sendRequests[blockId], MPI_STATUS_IGNORE);
                        waitTime += watch.watch.elapsedSeconds();
                        watch.print(format("node({}) block {} MPI_Wait", rank, blockId));
                    }
                } else {
                    failCount++;
                    usleep(100);
                }
            }
        }
    double totalTime = totalWatch.watch.elapsedSeconds();
    double otherTime = totalTime - waitTime - searchTime;
    totalWatch.print(format(
        "Search Finished node({}), total skip:{:.1f}%, waitTime {} {:.2f}% searchTime {} {:.2f}% otherTime {} {:.2f}%",
        rank, (double)totalSkip / totalCompare * 100, waitTime, 100 * waitTime / totalTime, searchTime,
        100 * searchTime / totalTime, otherTime, 100 * otherTime / totalTime));

    if (!blockSend) {
        for (size_t blockId = 0; blockId < info.blockCount; blockId++) {
            if (!shouldSendHeap(blockId)) {
                MPI_Wait(&sendRequests[blockId], MPI_STATUS_IGNORE);
            }
        }
    }
    // MPI_Waitall(sendRequests.size(), sendRequests.data(), MPI_STATUS_IGNORE);
    // MPI_Waitall(sendDistanceRequests.size(), sendDistanceRequests.data(), MPI_STATUS_IGNORE);
    // MPI_Waitall(sendIdRequests.size(), sendIdRequests.data(), MPI_STATUS_IGNORE);
}

void Worker::searchBlock(size_t blockId, bool cut) {
    uniWatch.print(format("searchBlock started: node({}) block({})", rank, blockId), false);
    MyStopWatch searchWatch(true, "searchBlock");

    // Get the distance buffer for the current block
    float* distanceBuffer = distancesForBlocks[blockId].get();

    // Calculate the start index for queries and total queries to compare
    size_t queryStart = blockId * blockSize;
    idx_t totalQueryCompareSize = getTotalQueryCompareSize(blockId);

    // Check if the total number of comparisons exceeds the allocated buffer size
    if (totalQueryCompareSize > blockDistancesSize) {
        cerr << "Error: blockDistancesSize too small" << endl;
        exit(1);
    }

    size_t skip = 0;
    size_t nt = std::min(static_cast<size_t>(omp_get_max_threads()), blockSize);
    // Parallelize the search process over queries
#pragma omp parallel for num_threads(nt) reduction(+ : skip)
    for (size_t q = queryStart; q < queryStart + blockSize; q++) {
        // Calculate the offset for storing the results of query q
        size_t queryOffset = queryCompareSizePreSum[q] - queryCompareSizePreSum[queryStart];  

        float* simi = distanceHeap[blockId].get() + k * (q - queryStart);  // Array to store similarities
        idx_t* idxi = idHeap[blockId].get() + k * (q - queryStart);  // Array to store candidate indices

        size_t curDistancePosition = 0;  // Track position within the query result
        for (size_t i = 0; i < info.nprobe; i++) {
            idx_t ivfId = listidqueries[q * info.nprobe + i];  // ID of the list of candidates to search
            for (size_t v = 0; v < listSizes[ivfId]; v++) {
                if (cut) {  // If 'cut' is true, skip comparisons with infinity distance
                    if (distanceBuffer[queryOffset + curDistancePosition] == INFINITY) {
                        skip++;  // Increment skip counter if the distance is INFINITY
                    } else {
                        // Calculate Euclidean distance between query and candidate
                        float dis = calculatedEuclideanDistance(
                            querys.get() + q * info.block_dim,
                            index->lists[ivfId].get_candidate_codes() + v * info.block_dim, info.block_dim);

                        // Add the calculated distance to the distance buffer
                        distanceBuffer[queryOffset + curDistancePosition] += dis;

                        // If the updated distance is greater than the current heap top, mark it as invalid (INFINITY)
                        if (distanceBuffer[queryOffset + curDistancePosition] > heapTops[q]) {
                            distanceBuffer[queryOffset + curDistancePosition] = INFINITY;
                        } else {
                            // If the heap should be updated, replace the top element of the heap
                            if (shouldSendHeap(blockId)) {
                                if (distanceBuffer[queryOffset + curDistancePosition] < simi[0]) {
                                    heap_replace_top<METRIC_L2>(k, simi, idxi,
                                                                distanceBuffer[queryOffset + curDistancePosition],
                                                                index->lists[ivfId].get_candidate_id()[v]);
                                }
                            }
                        }
                    }
                } else {
                    // Calculate Euclidean distance when 'cut' is not applied
                    float dis = calculatedEuclideanDistance(
                        querys.get() + q * info.block_dim,
                        index->lists[ivfId].get_candidate_codes() + v * info.block_dim, info.block_dim);

                    // Add the calculated distance to the distance buffer
                    distanceBuffer[queryOffset + curDistancePosition] += dis;

                    // If the heap should be updated, replace the top element of the heap
                    if (shouldSendHeap(blockId)) {
                        if (distanceBuffer[queryOffset + curDistancePosition] < simi[0]) {
                            heap_replace_top<METRIC_L2>(k, simi, idxi,
                                                        distanceBuffer[queryOffset + curDistancePosition],
                                                        index->lists[ivfId].get_candidate_id()[v]);
                        }
                    }
                }
                curDistancePosition++;  // Move to the next position in the result
            }
        }
        // Ensure the number of distance positions matches the expected number
        assert(curDistancePosition == queryCompareSize[q]);

        // Sort the results for the query
        sort_result(METRIC_L2, k, simi, idxi);
    }

    // Update global statistics: total skipped queries and comparison sizes
    totalSkip += skip;
    skipRates[blockId] = (double)skip / totalQueryCompareSize * 100;
    totalCompare += totalQueryCompareSize;

    // Log the search time for this block
    searchTime += searchWatch.watch.elapsedSeconds();
    searchWatch.print(format("node({}) search main loop block({}) skip:{:.1f}%", rank, blockId,
                             (double)skip / totalQueryCompareSize * 100));

    uniWatch.print(format("worker({}) -> block({}) -> worker({}) transfer started", rank, blockId, sendNextWorker[blockId]),
                   false);
    MyStopWatch watch(true);

    // Send heap data if required
    if (shouldSendHeap(blockId)) {
        MPI_Send(distanceHeap[blockId].get(), blockSize * k, MPI_FLOAT, sendNextWorker[blockId], blockId,
                 MPI_COMM_WORLD);
        MPI_Send(idHeap[blockId].get(), blockSize * k, MPI_INT64_T, sendNextWorker[blockId], blockId, MPI_COMM_WORLD);
    } else {
        // Send the distance buffer to the next worker if no heap data needs to be sent
        MPI_Isend(distanceBuffer, totalQueryCompareSize, MPI_FLOAT, sendNextWorker[blockId], blockId, MPI_COMM_WORLD,
                  &sendRequests[blockId]);
    }

    // Log the transmission time
    waitTime += watch.watch.elapsedSeconds();
    watch.print(format("worker({}) -> block({}) -> worker({}) transfer time", rank, blockId, sendNextWorker[blockId]));

    // Release the buffer if it has been fully transmitted (not implemented here)
    // distanceBufferPool->releaseBuffer(blockId);  // This can't be done directly as we need to check if the buffer has been fully sent.

    // Optionally, print the final search completion message (currently commented out)
    // searchWatch.print(format("node({}) search finished block({}) skip:{:.1f}%", rank, blockId, (double)skip /
    // totalQueryCompareSize * 100));

    // uniWatch.print(format("finish node({}) search block({}) skip:{:.1f}%", rank, blockId, (double)skip /
    // totalQueryCompareSize * 100), false);
}

void BaseWorker::init(int rank) {
    MyStopWatch watch(false);

    this->rank = rank;
    // this->index = index;

    // 1. InitInfo
    MPI_Recv(&info, sizeof(InitInfo), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // info.print();
    // IVF IDs are provided by startIVFid, arranged in order

    index = std::make_unique<Index>(info.d, info.nlist, info.nprobe);

    watch.print("index");

    // 2. IVF sizes and vector representation of IVF
    listSizes = std::make_unique<size_t[]>(info.ivfCount);
    MPI_Recv(listSizes.get(), info.ivfCount * sizeof(size_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

#pragma omp parallel for
    for (size_t i = 0; i < info.ivfCount; i++) {
        index->lists[i].reset(listSizes[i], info.d, 0);
    }
    watch.print("lists");

    size_t totalNb = 0;
    for (int i = 0; i < info.ivfCount; i++) {
        totalNb += listSizes[i];
    }

    listCodes = vector<std::unique_ptr<float[]>>(info.ivfCount);  // Codes for each cluster
    listIds = vector<std::unique_ptr<size_t[]>>(info.ivfCount);   // IDs for each cluster

    for (size_t i = 0; i < info.ivfCount; i++) {
        listCodes[i] = std::make_unique<float[]>(listSizes[i] * info.d);
        listIds[i] = std::make_unique<size_t[]>(listSizes[i]);
        MPI_Recv(listCodes[i].get(), listSizes[i] * info.d, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        copy_n(listCodes[i].get(), listSizes[i] * info.d, index->lists[i].candidate_codes.get());
        MPI_Recv(listIds[i].get(), listSizes[i] * sizeof(size_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        copy_n(listIds[i].get(), listSizes[i], index->lists[i].candidate_id.get());
        // MPI_Recv(curCodes, listSizes[i] * info.d , MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // MPI_Recv(curIds, listSizes[i] * sizeof(size_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // curCodes += listSizes[i] * info.d;
        // curIds += listSizes[i];
    }
    // for (size_t i = 0; i < info.ivfCount; i++) {
    //     MPI_Recv(index->lists[i].candidate_codes.get(), listSizes[i] * info.d , MPI_FLOAT, 0, 0, MPI_COMM_WORLD,
    //     MPI_STATUS_IGNORE); MPI_Recv(index->lists[i].candidate_id.get(), listSizes[i] * sizeof(size_t), MPI_BYTE, 0,
    //     0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // }
    watch.print("listCodes and listIds");

    MPI_Bcast(index->centroid_codes.get(), info.nlist * info.d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(index->centroid_ids.get(), info.nlist, MPI_INT64_T, 0, MPI_COMM_WORLD);

    // Allocate memory in advance
    // int presumeNq = 1000;
    int presumeK = 100;
    this->querys = std::make_unique<float[]>(presumeNq * info.d);
    listidqueries = std::make_unique<idx_t[]>(presumeNq * info.nprobe);  // IDs of the nearest nprobe clusters
    // queryCompareSize = std::make_unique<size_t[]>(presumeNq);
    // queryCompareSizePreSum = std::make_unique<size_t[]>(presumeNq + 1);

    distances = std::make_unique<float[]>(presumeNq * presumeK);
    labels = std::make_unique<idx_t[]>(presumeNq * presumeK);
    init_result(METRIC_L2, presumeNq * presumeK, distances.get(), labels.get());

    MPI_Barrier(MPI_COMM_WORLD);  // Corresponding to the barrier in preSearch

    uniWatch = MyStopWatch(true, "uniWatch", MAG);
    uniWatch.print(format("node {} cross barrier", rank), false);

    // nq, k, querys
    MPI_Bcast(&nq, sizeof(nq), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, sizeof(k), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (nq * k > presumeK * presumeNq) {
        cerr << "presumeNq is too small" << endl;
        exit(1);
    }
    uniWatch.print(format("node {} nq", rank), false);
    MPI_Bcast(querys.get(), nq * info.d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    uniWatch.print(format("node {} querys", rank), false);

    // IDs of the nearest nprobe clusters for the queries
    MPI_Bcast(listidqueries.get(), nq * info.nprobe, MPI_INT64_T, 0, MPI_COMM_WORLD);
    uniWatch.print(format("node {} listidqueries", rank), false);

    // queryCompareSize, queryCompareSizePreSum
    // queryCompareSize = std::make_unique<size_t[]>(nq);
    // MPI_Bcast(queryCompareSize.get(), nq, MPI_INT64_T, 0, MPI_COMM_WORLD);
    // MPI_Bcast(queryCompareSizePreSum.get(), (nq + 1), MPI_INT64_T, 0, MPI_COMM_WORLD);
    // uniWatch.print(format("node {} queryCompareSize", rank), false);

    // Max heap

    watch.print(format("Node {} Init", rank));
    uniWatch.print(format("node {} finish init", rank), false);
}

void BaseWorker::search() {
    MyStopWatch watch(true, "search watch");
    MyStopWatch totalWatch(true, "total watch");
    uniWatch.print(format("node {} index search start", rank), false);

    uniWatch.print(format("node {} start search", rank), false);

    size_t nt = std::min(static_cast<size_t>(omp_get_max_threads()), nq);
    size_t batch_size = nq / nt;
    size_t extra = nq % nt;
    // cout << nt << endl;
#pragma omp parallel for num_threads(nt)
    for (size_t i = 0; i < nt; i++) {
        size_t start, end;
        if (i < extra) {
            start = i * (batch_size + 1);
            end = start + batch_size + 1;
        } else {
            start = i * batch_size + extra;
            end = start + batch_size;
        }
        if (start < end) {
            // end - start is the number of query vectors, queries + start * d is the starting position of query vectors, distance + start * k
            // is the starting position of the result

            single_thread_search_simple(end - start, querys.get() + start * info.d, k, distances.get() + start * k,
                                        labels.get() + start * k, listidqueries.get() + start * info.nprobe);
        }
    }

    uniWatch.print(format("node {} finish search", rank), false);
    watch.print(format("node {} search finished", rank));
    totalWatch.print(format("node {} total", rank));
}

void BaseWorker::single_thread_search_fast(size_t n, const float* queries, size_t k, float* distances, idx_t* labels,
                                           idx_t* listidqueries) {
    std::unique_ptr<IVFScanBase> scaner_quantizer =
        index->get_scanner(index->metric, OPT_NONE, info.nprobe);  // Search for the nearest cluster centers

    std::unique_ptr<float[]> centroid2queries =
        std::make_unique<float[]>(n * info.nprobe);  // Distances from n query vectors to nprobe cluster centers
    // std::unique_ptr<idx_t[]> listidqueries = std::make_unique<idx_t[]>(n * info.nprobe);  // IDs of the nearest nprobe cluster centers
    // init_result(index->metric, n * info.nprobe, centroid2queries.get(), listidqueries);  // Priority queue to store the nprobe nearest cluster centers to n query vectors
    // copy_n(this->listidqueries.get(), n * info.nprobe, listidqueries.get());

    // The following four vectors are bound to i, which corresponds to each query
    float* simi = distances;             // Results: distances of the nearest k vectors to each query vector
    idx_t* idxi = labels;                // Results: IDs of the nearest k vectors to each query vector
    float* centroids2query = centroid2queries.get();  // Distance from a single query to the centroid
    idx_t* listids = listidqueries;                 // The IVF cluster center IDs corresponding to a single query

    int calculatedCount = 0;

    for (size_t i = 0; i < n; i++) {
        // For each query i
        for (size_t j = 0; j < info.nprobe; j++) {
            // Search all points in the j-th cluster
            idx_t ivfId = listids[j];
            if (!(ivfId >= info.startIVFId && ivfId < info.startIVFId + info.ivfCount)) {
                continue;
            }
            IVF& list = index->lists[listids[j] - info.startIVFId];

            // Distance from the query point to the center
            float centroid2query = centroids2query[j];
            // Number of points in the cluster
            size_t list_size = list.get_list_size();

            for (size_t v = 0; v < list_size; v++) {
                const float* candicate = list.get_candidate_codes() + v * index->d;
                float dis = 0;
                dis = calculatedEuclideanDistance(queries + i * index->d, candicate, index->d);
                if (dis < simi[0]) {
                    // If the distance is less than the top of the heap, replace the top
                    heap_replace_top<METRIC_L2>(k, simi, idxi, dis, list.get_candidate_id()[v]);
                }
            }
        }
        // Sort the results (optional, commented out in this case)
        // sort_result(index->metric, k, simi, idxi);

        simi += k;
        idxi += k;
        centroids2query += info.nprobe;
        listids += info.nprobe;
    }
}

void BaseWorker::single_thread_search_simple(size_t n, const float* queries, size_t k, float* distances, idx_t* labels,
                                             idx_t* listidqueries) {


    // The following four vectors are bound to i, which corresponds to each query
    float* simi = distances;         // Results: distances of the nearest k vectors to each query vector
    idx_t* idxi = labels;            // Results: IDs of the nearest k vectors to each query vector
    idx_t* listids = listidqueries;  // IVF cluster center IDs corresponding to a single query

    // MyStopWatch w;
    for (size_t i = 0; i < n; i++) {
        // For each query i
        for (size_t j = 0; j < info.nprobe; j++) {
            // Search all points in the j-th cluster
            idx_t ivfId = listids[j];
            if (!(ivfId >= info.startIVFId && ivfId < info.startIVFId + info.ivfCount)) {
                continue;
            }
            IVF& list = index->lists[ivfId - info.startIVFId];
            size_t listSize = list.get_list_size();
            float* codes = list.candidate_codes.get();
            size_t* ids = list.candidate_id.get();

            for (size_t v = 0; v < list.get_list_size(); v++) {
                const float* candicate = list.get_candidate_codes() + v * index->d;
                float dis = 0;
                dis = calculatedEuclideanDistance(queries + i * index->d, candicate, index->d);
                if (dis < simi[0]) {
                    // If the distance is less than the top of the heap, replace the top
                    heap_replace_top<METRIC_L2>(k, simi, idxi, dis, list.get_candidate_id()[v]);
                }
            }
        }
        simi += k;
        idxi += k;
        listids += info.nprobe;

    }
}

void GroupWorker::init(int rank, bool blockSend) {
    MyStopWatch watch(false);

    this->rank = rank;
    this->blockSend = blockSend;

    // InitInfo: Receive initial configuration
    MPI_Recv(&info, sizeof(InitInfo), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    info.print();

    index = std::make_unique<Index>(info.d, info.nlist, info.nprobe);

    // IVF size and vector representation
    listSizes = std::make_unique<size_t[]>(info.nlist);
    MPI_Recv(listSizes.get(), info.nlist * sizeof(size_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

#pragma omp parallel for
    for (size_t i = info.startIVFId; i < info.startIVFId + info.ivfCount; i++) {
        index->lists[i].reset(listSizes[i], info.block_dim, 0);
    }
    watch.print("lists");

    for (size_t i = info.startIVFId; i < info.startIVFId + info.ivfCount; i++) {
        MPI_Recv(index->lists[i].candidate_id.get(), listSizes[i] * sizeof(size_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
    // Receiving candidate vectors and IDs
    auto listCodesBuffer = vector<std::unique_ptr<float[]>>(info.nlist);
    for (size_t i = info.startIVFId; i < info.startIVFId + info.ivfCount; i++) {
        listCodesBuffer[i] = std::make_unique<float[]>(listSizes[i] * info.d);
        MPI_Recv(listCodesBuffer[i].get(), listSizes[i] * info.d, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    for (size_t i = info.startIVFId; i < info.startIVFId + info.ivfCount; i++) {
        copy_n_partial_vector(listCodesBuffer[i].get(), index->lists[i].candidate_codes.get(), info.d, info.block_dim,
                              (info.rankInsideTeam - 1) * info.block_dim, listSizes[i]);
    }

    groupSearchOrder = SearchOrder(info.teamCount, info.groupCount, true);
    blockSearchOrder = SearchOrder(info.teamSize, info.blockCount, true);

    // Allocating memory
    index->originalQuery = std::make_unique<float[]>(presumeNq * info.d);
    this->querys = std::make_unique<float[]>(presumeNq * info.block_dim);
    listidqueries = std::make_unique<idx_t[]>(presumeNq * info.nprobe);  // IDs of the nearest nprobe cluster centers
    heapTops = std::make_unique<float[]>(presumeNq);
    queryCompareSize = std::make_unique<idx_t[]>(presumeNq);
    queryCompareSizePreSum = std::make_unique<idx_t[]>(presumeNq + 1);

    distanceHeap = vector<std::unique_ptr<float[]>>(info.groupCount);
    idHeap = vector<std::unique_ptr<idx_t[]>>(info.groupCount);
    for (size_t i = 0; i < info.groupCount; i++) {
        distanceHeap[i] = std::make_unique<float[]>(presumeK * presumeNq);
        idHeap[i] = std::make_unique<idx_t[]>(presumeK * presumeNq);
        init_result(METRIC_L2, presumeNq * presumeK, distanceHeap[i].get(), idHeap[i].get());
    }

    blockDistancesSize =
        2 * presumeNq / info.blockCount / info.groupCount * info.nb / info.teamSize * info.nprobe / info.nlist;
    distancesForBlocks = vector<vector<std::unique_ptr<float[]>>>(info.groupCount);
    for (size_t i = 0; i < distancesForBlocks.size(); i++) {
        distancesForBlocks[i] = vector<std::unique_ptr<float[]>>(info.blockCount);
        for (size_t j = 0; j < distancesForBlocks[i].size(); j++) {
            distancesForBlocks[i][j] = std::make_unique<float[]>(blockDistancesSize);
        }
    }

    disRequests = vector<vector<MPI_Request>>(info.blockCount);
    for (int i = 0; i < disRequests.size(); i++) {
        disRequests[i] = vector<MPI_Request>(1);
    }
    sendRequests = vector<MPI_Request>(info.blockCount);
    sendDistanceRequests = vector<MPI_Request>(info.blockCount);
    sendIdRequests = vector<MPI_Request>(info.blockCount);

    skipRates = vector<double>(info.blockCount, 0);

    // cout << "finish init" << endl;
}


void GroupWorker::receiveQuery() {
    MPI_Barrier(MPI_COMM_WORLD);  // Corresponds to the barrier at the end of preSearch

    uniWatch = MyStopWatch(true, "uniWatch", MAG);
    uniWatch.print(format("node {} cross barrier", rank), false);
    // nq, querys
    MPI_Bcast(&nq, sizeof(nq), MPI_BYTE, 0, MPI_COMM_WORLD);  // Broadcast nq (number of queries)
    MPI_Bcast(&k, sizeof(k), MPI_BYTE, 0, MPI_COMM_WORLD);    // Broadcast k (number of nearest neighbors)
    
    // Validate parameters
    if ((double)nq * k > presumeK * presumeNq) {
        cerr << "presumeNq * k is too small" << endl;
        exit(1);
    }
    if (nq > presumeNq) {
        cerr << "presumeNq is too small" << endl;
        exit(1);
    }
    // uniWatch.print(format("node {} nq k {} ", rank, k), false);

    MPI_Bcast(index->originalQuery.get(), nq * info.d, MPI_FLOAT, 0, MPI_COMM_WORLD);  // Broadcast query data
    addQuerys(index->originalQuery.get(), nq);  // Add queries to the local list

    // uniWatch.print(format("node {} querys", rank), false);

    MPI_Bcast(listidqueries.get(), nq * info.nprobe, MPI_INT64_T, 0, MPI_COMM_WORLD);  // Broadcast query list IDs

    // uniWatch.print(format("node {} listidqueries", rank), false);

    // queryCompareSize, queryCompareSizePreSum
    MPI_Recv(queryCompareSize.get(), nq, MPI_INT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  // Receive query comparison sizes
    MPI_Recv(queryCompareSizePreSum.get(), (nq + 1), MPI_INT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  // Receive prefix sum of query comparison sizes

    groupSize = nq / info.groupCount;  // Divide the number of queries into groups
    blockSize = groupSize / info.blockCount;  // Divide each group into blocks

    // cout << "finish receive query" << endl;
    uniWatch.print(format("node {} finish Receving", rank), false);  // Finished receiving queries
}

void GroupWorker::search(bool cut) {
    this->cut = cut;

    MyStopWatch totalWatch(true, "Total Watch");

    // Initialize max heap
    MPI_Bcast(heapTops.get(), nq, MPI_FLOAT, 0, MPI_COMM_WORLD);  // Broadcast the initial heap tops
    uniWatch.print(format("node {} received initial heapTops", rank), false);

    // Process each group in parallel
    for (int i = 0; i < info.groupCount; i++) {
        size_t groupId = groupSearchOrder.workerSearchOrder[info.teamId][i];
        // If an updated heapTop needs to be received
        size_t sender = groupSearchOrder.recvPrevWorker[info.teamId][groupId];
        if (sender != 0) {
            MyStopWatch waitWatch(true, "Wait time between master and worker");
            MPI_Recv(heapTops.get() + groupId * groupSize, groupSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);  // Receive updated heapTops for this group
            MPI_Recv(distanceHeap[groupId].get(), groupSize * k, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  // Receive distance heap
            MPI_Recv(idHeap[groupId].get(), groupSize * k, MPI_INT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  // Receive ID heap

            waitTime += waitWatch.watch.elapsedSeconds();
            waitWatch.print(format("node {} received group {} Heap", rank, groupId));  // Record time for receiving heap
            uniWatch.print(format("node {} received group {}", rank, groupId), false);
        }

        searchGroup(groupId);  // Search within the group

        reset();  // Reset state after group search
    }

    // totalWatch.print(format("node {} finished all group searches", rank));
    double totalTime = totalWatch.watch.elapsedSeconds();
    double otherTime = totalTime - waitTime - searchTime;
    totalWatch.print(format(
        "node({}) finished all group searches, waitTime {:.3f}s {:.2f}% searchTime {:.3f}s {:.2f}% otherTime {:.3f}s {:.2f}%",
        rank, waitTime, 100 * waitTime / totalTime, searchTime, 100 * searchTime / totalTime, otherTime,
        100 * otherTime / totalTime));
}

size_t GroupWorker::getSender(idx_t groupId, idx_t blockId) {
    if (blockSearchOrder.getRecvPrevWorker(info.rankInsideTeam, blockId) == 0) {
        return 0;
    }
    return blockSearchOrder.getRecvPrevWorker(info.rankInsideTeam, blockId) + (info.teamId - 1) * info.teamSize;
}

size_t GroupWorker::getReceiver(idx_t groupId, idx_t blockId) {
    if (blockSearchOrder.getSendNextWorker(info.rankInsideTeam, blockId) == 0) {
        return 0;
    }
    return blockSearchOrder.getSendNextWorker(info.rankInsideTeam, blockId) + (info.teamId - 1) * info.teamSize;
}

void GroupWorker::searchGroup(idx_t groupId) {
    // cout << format("node {} groupid {}", rank, groupId) << endl;
    uniWatch.print(format("node {} starting searchGroup {}", rank, groupId), false);
    MyStopWatch groupWatch(true, "Group search watch");

    int searchedBlockCount = 0;
    vector<bool> isBlockSearched = vector<bool>(info.blockCount);

    for (size_t blockId = 0; blockId < info.blockCount; blockId++) {
        size_t sender = getSender(groupId, blockId);
        if (sender != 0) {
            MPI_Irecv(distancesForBlocks[groupId][blockId].get(), getBlockQueryCompareSize(groupId, blockId), MPI_FLOAT,
                      sender, getTag(groupId, blockId, info.blockCount), MPI_COMM_WORLD, &disRequests[blockId][0]);  // Non-blocking receive for distance data
        }
    }

    // Continuously search while receiving data
    MyStopWatch watch(true);
    idx_t failCount = 0;
    while (searchedBlockCount < info.blockCount)
        for (size_t blockId = 0; blockId < info.blockCount; blockId++) {
            if (isBlockSearched[blockId]) {
                continue;
            }
            size_t sender = getSender(groupId, blockId);
            if (sender == 0) {
                searchBlock(blockId, groupId);  // Perform search on the block
                isBlockSearched[blockId] = true;
                searchedBlockCount++;

                watch.reset();
                if (blockSend && !shouldSendHeap(blockId)) {
                    MPI_Wait(&sendRequests[blockId], MPI_STATUS_IGNORE);  // Wait for send to complete if necessary
                    waitTime += watch.watch.elapsedSeconds();
                    watch.print(format("node({}) block {} MPI_Wait", rank, blockId));
                }

            } else {
                int isReceived;
                MPI_Status stat;
                bool testFail = false;
                for (auto& req : disRequests[blockId]) {
                    MPI_Test(&req, &isReceived, &stat);  // Test if the data is received
                    if (!isReceived) {
                        testFail = true;
                        break;
                    }
                }
                if (!testFail) {
                    // cout << GREEN << format("node({}) received block({}) from node({})", rank, blockId,
                    // stat.MPI_SOURCE) << RESET << endl;
                    waitTime += watch.watch.elapsedSeconds();
                    watch.print(format("node {} waiting, now received block {} fail {}", rank, blockId, failCount));
                    uniWatch.print(format("node {} received block {} Test failed count {}", rank, blockId, failCount),
                                   false);

                    searchBlock(blockId, groupId);  // Perform search on the received block

                    searchedBlockCount++;
                    isBlockSearched[blockId] = true;
                    failCount = 0;

                    watch.reset();
                    if (blockSend && !shouldSendHeap(blockId)) {
                        MPI_Wait(&sendRequests[blockId], MPI_STATUS_IGNORE);  // Wait for send completion
                        waitTime += watch.watch.elapsedSeconds();
                        watch.print(format("node({}) block {} MPI_Wait", rank, blockId));
                    }
                } else {
                    failCount++;
                    usleep(100);  // Wait a bit before retrying
                }
            }
        }
    // double totalTime = groupWatch.watch.elapsedSeconds();
    // double otherTime = totalTime - waitTime - searchTime;
    // groupWatch.print(format("node({}) finished search for Group{}, pruning rate:{:.1f}%, waitTime {:.3f}s {:.2f}% searchTime {:.3f}s {:.2f}% otherTime {:.3f}s {:.2f}%",
    //     rank, groupId, (double)totalSkip / totalCompare * 100, waitTime, 100 * waitTime / totalTime, searchTime, 100
    //     * searchTime / totalTime, otherTime, 100 * otherTime / totalTime));

    if (!blockSend) {
        for (size_t blockId = 0; blockId < info.blockCount; blockId++) {
            if (!shouldSendHeap(blockId)) {
                MPI_Wait(&sendRequests[blockId], MPI_STATUS_IGNORE);  // Wait for send completion if necessary
            }
        }
    }
}


void GroupWorker::single_thread_searchBlock(size_t n, size_t blockId, size_t groupId, float* queries,
                                            float* distanceBuffer, float* simi, idx_t* idxi, idx_t* listidqueries) {
    size_t curDistancePosition = 0;  // Position within the result of a query vector
    size_t skip = 0;
    for (size_t q = 0; q < n; q++) {
        for (size_t i = 0; i < info.nprobe; i++) {
            idx_t ivfId = listidqueries[q * info.nprobe + i];

            if (ivfId >= info.startIVFId && ivfId < info.startIVFId + info.ivfCount) {
                IVF& list = index->lists[ivfId];

                for (size_t v = 0; v < listSizes[ivfId]; v++) {
                    const float* codes = list.get_candidate_codes() + v * info.block_dim;

                    if (cut) {
                        // Skip calculation if the distance is infinity
                        if (distanceBuffer[curDistancePosition] == INFINITY) {
                            skip++;
                        } else {
                            float dis =
                                calculatedEuclideanDistance(querys.get() + q * info.block_dim, codes, info.block_dim);

                            distanceBuffer[curDistancePosition] += dis;

                            if (distanceBuffer[curDistancePosition] > heapTops[q]) {
                                distanceBuffer[curDistancePosition] = INFINITY;
                            } else {
                                if (shouldSendHeap(blockId)) {
                                    if (distanceBuffer[curDistancePosition] < simi[0]) {
                                        // If the distance is smaller than the heap top, replace the heap top
                                        heap_replace_top<METRIC_L2>(k, simi, idxi, distanceBuffer[curDistancePosition],
                                                                    index->lists[ivfId].get_candidate_id()[v]);
                                    }
                                }
                            }
                        }
                    } else {
                        float dis = calculatedEuclideanDistance(
                            querys.get() + q * info.block_dim,
                            index->lists[ivfId].get_candidate_codes() + v * info.block_dim, info.block_dim);
                        
                        if (shouldSendHeap(blockId)) {
                            if (dis < simi[0]) {
                                // If the distance is smaller than the heap top, replace the heap top
                                heap_replace_top<METRIC_L2>(k, simi, idxi, dis,
                                                            index->lists[ivfId].get_candidate_id()[v]);
                            }
                        }
                    }
                    curDistancePosition++;
                }
            }
        }
    }
}

void GroupWorker::searchBlock(size_t blockId, size_t groupId) {
    // Print that the search for a block has started
    // uniWatch.print(format("node({}) start searching block({}), group({})", rank, blockId, groupId), false);

    // Get the distance buffer for the block
    float* distanceBuffer = distancesForBlocks[groupId][blockId].get();

    size_t queryStart = getQueryOffset(groupId, blockId);
    idx_t totalQueryCompareSize = getBlockQueryCompareSize(groupId, blockId);

    // Check if the total query compare size exceeds the buffer size
    if (totalQueryCompareSize > blockDistancesSize) {
        cerr << "Error: blockDistancesSize is too small" << endl;
        exit(1);
    }

    MyStopWatch searchWatch(false, "searchBlock");

    // Parallel search using OpenMP
    size_t skip = 0;

    size_t nt = std::min(static_cast<size_t>(omp_get_max_threads()), blockSize);
#pragma omp parallel for reduction(+ : skip)
    for (size_t q = queryStart; q < queryStart + blockSize; q++) {
        size_t queryOffset =
            queryCompareSizePreSum[q] - queryCompareSizePreSum[queryStart];  // Offset for storing the result of query q

        float* query = querys.get() + q * info.block_dim;
        float* simi = distanceHeap[groupId].get() + k * (q - queryStart + blockId * blockSize);
        idx_t* idxi = idHeap[groupId].get() + k * (q - queryStart + blockId * blockSize);
        size_t curDistancePosition = 0;  // Position within the result of a query vector

        for (size_t i = 0; i < info.nprobe; i++) {
            idx_t ivfId = listidqueries[q * info.nprobe + i];

            if (ivfId >= info.startIVFId && ivfId < info.startIVFId + info.ivfCount) {
                IVF& list = index->lists[ivfId];

                for (size_t v = 0; v < listSizes[ivfId]; v++) {
                    const float* codes = list.get_candidate_codes() + v * info.block_dim;

                    if (cut) {
                        // Skip if the distance is infinity
                        if (distanceBuffer[queryOffset + curDistancePosition] == INFINITY) {
                            skip++;
                        } else {
                            float dis =
                                calculatedEuclideanDistance(querys.get() + q * info.block_dim, codes, info.block_dim);

                            distanceBuffer[queryOffset + curDistancePosition] += dis;

                            if (distanceBuffer[queryOffset + curDistancePosition] > heapTops[q]) {
                                distanceBuffer[queryOffset + curDistancePosition] = INFINITY;
                            } else {
                                if (shouldSendHeap(blockId)) {
                                    if (distanceBuffer[queryOffset + curDistancePosition] < simi[0]) {
                                        // If the distance is smaller than the heap top, replace the heap top
                                        heap_replace_top<METRIC_L2>(k, simi, idxi,
                                                                    distanceBuffer[queryOffset + curDistancePosition],
                                                                    index->lists[ivfId].get_candidate_id()[v]);
                                    }
                                }
                            }
                        }
                    } else {
                        float dis = calculatedEuclideanDistance(
                            query, index->lists[ivfId].get_candidate_codes() + v * info.block_dim, info.block_dim);
                        
                        distanceBuffer[queryOffset + curDistancePosition] += dis;
                        if (shouldSendHeap(blockId)) {
                            dis = distanceBuffer[queryOffset + curDistancePosition];
                            if (dis < simi[0]) {
                                // If the distance is smaller than the heap top, replace the heap top
                                heap_replace_top<METRIC_L2>(k, simi, idxi, dis,
                                                            index->lists[ivfId].get_candidate_id()[v]);
                            }
                        }
                    }
                    curDistancePosition++;
                }
            }
        }
    }

    searchTime += searchWatch.watch.elapsedSeconds();
    // Print the search time for this block
    searchWatch.print(format("node({}) search main loop block({}) group({}) skip:{:.1f}%", rank, blockId, groupId,
                             (double)skip / totalQueryCompareSize * 100));

    totalSkip += skip;
    skipRates[blockId] = (double)skip / totalQueryCompareSize * 100;
    totalCompare += totalQueryCompareSize;

    // Print that the data transmission for this block is starting
    uniWatch.print(
        format("worker({}) -> block({}) -> worker({}) transmission started", rank, blockId, getReceiver(groupId, blockId)), false);
    
    MyStopWatch watch(true);
    int tag = GroupWorker::getTag(groupId, blockId, info.blockCount);
    
    if (shouldSendHeap(blockId)) {
        // Send the heap data if needed
        MPI_Send(distanceHeap[groupId].get() + blockId * blockSize * k, blockSize * k, MPI_FLOAT, 0, tag,
                 MPI_COMM_WORLD);
        MPI_Send(idHeap[groupId].get() + blockId * blockSize * k, blockSize * k, MPI_INT64_T, 0, tag, MPI_COMM_WORLD);
    } else {
        // Otherwise, send the distance buffer
        MPI_Isend(distanceBuffer, totalQueryCompareSize, MPI_FLOAT, getReceiver(groupId, blockId), tag, MPI_COMM_WORLD,
                  &sendRequests[blockId]);
    }
    
    waitTime += watch.watch.elapsedSeconds();
    watch.print(format("worker({}) -> block({}) -> worker({}) transmission time", rank, blockId, getReceiver(groupId, blockId)));

    // Distance buffer should not be released until transmission is complete
    // distanceBufferPool->releaseBuffer(blockId);
}

}  // namespace harmony