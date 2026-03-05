#include "node.h"

#include "Index.h"
namespace harmony {


void Worker::init(int rank, bool blockSend) {
    // Timer object to measure initialization times.
    MyStopWatch watch(false);

    this->rank = rank;
    this->blockSend = blockSend;

    // Initialize configuration info by broadcasting InitInfo structure from the root (rank 0) process.
    MPI_Bcast(&info, sizeof(InitInfo), MPI_BYTE, 0, MPI_COMM_WORLD);
    // info.print();

    // Initialize the index with parameters from InitInfo.
    index = std::make_unique<Index>(info.d, info.nlist, info.nprobe);

    // Allocate memory for storing the sizes of inverted file lists (IVF) and broadcast the sizes.
    listSizes = std::make_unique<size_t[]>(info.nlist);
    MPI_Bcast(listSizes.get(), info.nlist * sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Initialize IVF list structures in parallel using OpenMP.
#pragma omp parallel for
    for (size_t i = 0; i < info.nlist; i++) {
        index->lists[i].reset(listSizes[i], info.block_dim, 0);
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
        MPI_Bcast(listCodesBuffer[i].get(), listSizes[i] * info.d , MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    // listCodes = vector<std::unique_ptr<float[]>>(info.nlist);
    for (size_t i = 0; i < info.nlist; i++) {
        copy_n_partial_vector(listCodesBuffer[i].get(), index->lists[i].candidate_codes.get(), info.d, info.block_dim,
                              (rank - 1) * info.block_dim, listSizes[i]);
    }

    MPI_Bcast(index->centroid_codes.get(), info.nlist * info.d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(index->centroid_ids.get(), info.nlist , MPI_INT64_T, 0, MPI_COMM_WORLD);

    watch.print("listCodes and listIds");

    blockSearchOrder = std::make_unique<idx_t[]>(info.blockCount);   // search block的顺序，第i个元素是第i个要进行search的blockId
    MPI_Recv(blockSearchOrder.get(), info.blockCount, MPI_INT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // printVector(blockSearchOrder.get(), info.blockCount, BLUE);

    // std::unique_ptr<float[]> querysBuffer = std::make_unique<float[]>(presumeNq * info.d);
    index->originalQuery = std::make_unique<float[]>(presumeNq * info.d);
    this->querys = std::make_unique<float[]>(presumeNq * info.block_dim);
    listidqueries = std::make_unique<idx_t[]>(presumeNq * info.nprobe);  // 最近的nprobe个聚类中心的id
    // index->heapTops = std::make_unique<float[]>(presumeNq);
    heapTops = std::make_unique<float[]>(presumeNq);
    queryCompareSize = std::make_unique<idx_t[]>(presumeNq);
    queryCompareSizePreSum = std::make_unique<idx_t[]>(presumeNq + 1);
    sendNextWorker = std::make_unique<idx_t[]>(info.blockCount);
    recvPrevWorker = std::make_unique<idx_t[]>(info.blockCount);
    // distanceHeap = std::make_unique<float[]>(presumeNq * presumeK);
    // idHeap = std::make_unique<idx_t[]>(presumeNq * presumeK);
    distanceHeap = vector<std::unique_ptr<float[]>>(info.blockCount);
    idHeap = vector<std::unique_ptr<idx_t[]>>(info.blockCount);
    for (size_t i = 0; i < info.blockCount; i++) {
        distanceHeap[i] = std::make_unique<float[]>(presumeK * presumeNq);
        idHeap[i] = std::make_unique<idx_t[]>(presumeK * presumeNq);
        init_result(METRIC_L2, presumeNq * presumeK, distanceHeap[i].get(), idHeap[i].get());
    }
    // init_result(METRIC_L2, presumeNq * presumeK, distanceHeap.get(), idHeap.get());
    
    blockDistancesSize = 2 * presumeNq / info.blockCount * info.nb;
    
    // distancesForBlocks = vector<std::unique_ptr<float[]>>(info.blockCount);
    // for (size_t i = 0; i < info.blockCount; i++) {
    //     distancesForBlocks[i] = std::make_unique<float[]>(blockDistancesSize);
    // }
    // blockSize = presumeNq / info.blockCount;
    // presumeBlockDistancesSize = presumeNq / info.blockCount * info.nb * 2;
    // cout << presumeBlockDistancesSize << "blockD" << endl;
    distanceBufferPool = std::make_unique<DistanceBufferPool>(info, this);
    // watch.print(format("node {} distancesForBlocks", rank));

    disRequests.clear();
    sendRequests.clear();
    disRequests.resize(info.blockCount);
    sendRequests.resize(info.blockCount);
    // disRequests = vector<vector<MPI_Request>>(info.blockCount);
    // sendRequests = vector<vector<MPI_Request>>(info.blockCount);
    // for(int i = 0; i < disRequests.size(); i++) {
    //     disRequests[i] = vector<MPI_Request>();
    //     sendRequests[i] = vector<MPI_Request>();
    // }
    // sendRequests = vector<MPI_Request>(info.blockCount);
    sendDistanceRequests = vector<MPI_Request>(info.blockCount);
    sendIdRequests = vector<MPI_Request>(info.blockCount);

    skipRates = vector<double>(info.blockCount, 0);

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

    // nq, querys
    MPI_Bcast(&nq, sizeof(nq), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, sizeof(k), MPI_BYTE, 0, MPI_COMM_WORLD);
    if((double)nq * k > presumeK * presumeNq) {
        cerr << "presumeNq * k is too small" << endl;
        exit(1);
    }
    if(nq > presumeNq) {
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

    // queryCompareSize,queryCompareSizePreSum
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
    uniWatch.print(format("node {} finish Receving", rank), false);


    MyStopWatch totalWatch(true, "total search watch");
    
    int searchedBlockCount = 0;
    vector<bool> isBlockSearched = vector<bool>(info.blockCount);

    // Allocate memory for the blocks that can be computed
    for (size_t blockId = 0; blockId < info.blockCount; blockId++) {
        if(recvPrevWorker[blockId] == 0) {
            // Directly mark as used
            distanceBufferPool->use(blockId);
        }
    }
    for (size_t blockId = 0; blockId < info.blockCount; blockId++) {
        if(recvPrevWorker[blockId] != 0) {
            size_t sender = recvPrevWorker[blockId];
            // tag as blockId
            // MPI_Irecv(distancesForBlocks[blockId].get(), getTotalQueryCompareSize(blockId), MPI_FLOAT, sender, blockId, MPI_COMM_WORLD, &disRequests[blockId][0]);
            // MPI_Irecv(distanceBufferPool->getBuffer(blockId), getTotalQueryCompareSize(blockId), MPI_FLOAT, sender, blockId, MPI_COMM_WORLD, &disRequests[blockId]);
            bool suc = distanceBufferPool->IRecv(blockId, getTotalQueryCompareSize(blockId), sender, disRequests[blockId]);
            if(!suc) {
                break;
            }
            // cout << format("node({}) waiting for block({}) from node({})", rank, blockId, sender) << endl;
        } 
    }

    // Continue searching while blocks are not finished
    MyStopWatch watch(true);
    idx_t failCount = 0;
    while(searchedBlockCount < info.blockCount)
    for (size_t blockId = 0; blockId < info.blockCount; blockId++) {
        if(isBlockSearched[blockId]) {
            continue;
        }
        if(recvPrevWorker[blockId] == 0) {
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

            if(blockSend && !shouldSendHeap(blockId)) {
                for(auto& req : sendRequests[blockId]) {
                    MPI_Wait(&req, MPI_STATUS_IGNORE);
                }
                // MPI_Wait(&sendRequests[blockId], MPI_STATUS_IGNORE);
                waitTime += watch.watch.elapsedSeconds();
                watch.print(format("node({}) block {} MPI_Wait", rank, blockId));
            }

            // }

        } else {
            bool testFail = false;
            for(auto& req : disRequests[blockId]) {
                int isReceived;
                MPI_Status stat;
                MPI_Test(&req, &isReceived, &stat);
                if(!isReceived) {
                    testFail = true;
                    break;
                }
            }
            if(!testFail) {
                // cout << GREEN << format("node({}) received block({}) from node({})",rank, blockId, stat.MPI_SOURCE) << RESET << endl;
                waitTime += watch.watch.elapsedSeconds();
                watch.print(format("node {} waiting , now received block {} fail {}", rank, blockId, failCount));
                uniWatch.print(format("node {} received block {} Test failed count {}", rank, blockId, failCount),
                                   false);

                searchBlock(blockId, cut);

                searchedBlockCount++;
                isBlockSearched[blockId] = true;
                failCount = 0;

                watch.reset();
                if(blockSend && !shouldSendHeap(blockId)) {
                    for(auto& req : sendRequests[blockId]) {
                        MPI_Wait(&req, MPI_STATUS_IGNORE);
                    }
                    // MPI_Wait(&sendRequests[blockId], MPI_STATUS_IGNORE);
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
    totalWatch.print(format("Search Finished node({}), total skip:{:.1f}%, waitTime {} {:.2f}% searchTime {} {:.2f}% otherTime {} {:.2f}%", 
        rank, (double)totalSkip / totalCompare * 100, waitTime, 100 * waitTime / totalTime, searchTime, 100 * searchTime / totalTime, otherTime, 100 * otherTime / totalTime));

    if(!blockSend) {
        for(size_t blockId = 0; blockId < info.blockCount; blockId++) {
            if(!shouldSendHeap(blockId)) {
                cout << "sendRequests[blockId].size" << sendRequests[blockId].size() << endl;
                for(auto& req : sendRequests[blockId]) {
                    MPI_Wait(&req, MPI_STATUS_IGNORE);
                }
                // MPI_Wait(&sendRequests[blockId], MPI_STATUS_IGNORE);
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


    float* distanceBuffer = distanceBufferPool->getBuffer(blockId);
    // float* distanceBuffer = distancesForBlocks[blockId].get();

    size_t queryStart = blockId * blockSize;
    idx_t totalQueryCompareSize = getTotalQueryCompareSize(blockId);
        
    // cout << RED << rank << "node search: blockId:" << blockId << " totalCompareSize:" << totalQueryCompareSize << RESET
    //      << endl;


    if (totalQueryCompareSize > blockDistancesSize) {
        cerr << "Error blockDistancesSize too small" << endl;
        exit(1);
    }

    size_t skip = 0;
    // index->search(blockSize, querys.get() + queryStart * info.block_dim, 0, distanceBuffer, NULL, 1, &param);

    

    size_t nt = std::min(static_cast<size_t>(omp_get_max_threads()), blockSize);
//     // cout << "node " << rank << " nt = " << nt << endl;
#pragma omp parallel for num_threads(nt) reduction(+:skip)
// #pragma omp parallel for num_threads(nt)
    for (size_t q = queryStart; q < queryStart + blockSize; q++) {
        // cout << "node " << rank << " nt = " << omp_get_num_threads() << endl;
        size_t queryOffset =
            queryCompareSizePreSum[q] - queryCompareSizePreSum[queryStart];  

        float* simi = distanceHeap[blockId].get() + k * (q - queryStart);
        idx_t* idxi = idHeap[blockId].get() + k * (q - queryStart);

        size_t curDistancePosition = 0;  
        for (size_t i = 0; i < info.nprobe; i++) {
            idx_t ivfId = listidqueries[q * info.nprobe + i];
            for (size_t v = 0; v < listSizes[ivfId]; v++) {
                if(cut) {
                    // if(distancesForBlocks[blockId][queryOffset + curDistancePosition] == INFINITY) {
                    if(distanceBuffer[queryOffset + curDistancePosition] == INFINITY) {
                        skip++;
                    } else {
                        float dis = calculatedEuclideanDistance(querys.get() + q * info.block_dim,
                                                                index->lists[ivfId].get_candidate_codes() + v * info.block_dim,
                                                                info.block_dim);
                        // cout << " " << queryOffset + curDistancePosition  << endl;
                        
                        // assert(queryOffset + curDistancePosition < totalQueryCompareSize);
                        distanceBuffer[queryOffset + curDistancePosition] += dis;
                        if (distanceBuffer[queryOffset + curDistancePosition] > heapTops[q]) {
                            distanceBuffer[queryOffset + curDistancePosition] = INFINITY;
                        } else {
                            if(shouldSendHeap(blockId)) {
                                if (distanceBuffer[queryOffset + curDistancePosition] < simi[0]) {
                                    heap_replace_top<METRIC_L2>(k, simi, idxi, distanceBuffer[queryOffset + curDistancePosition], index->lists[ivfId].get_candidate_id()[v]);
                                }
                            } 
                        }
                    }
                } else {
                    float dis = calculatedEuclideanDistance(querys.get() + q * info.block_dim,
                                                            index->lists[ivfId].get_candidate_codes() + v * info.block_dim,
                                                            info.block_dim);
                    // cout << " " << queryOffset + curDistancePosition  << endl;
                    assert(queryOffset + curDistancePosition < totalQueryCompareSize);
                    distanceBuffer[queryOffset + curDistancePosition] += dis;
                    if(shouldSendHeap(blockId)) {
                        if (distanceBuffer[queryOffset + curDistancePosition] < simi[0]) {
                            // cout << format("q {} replace {} {} top {}", q, distanceBuffer[queryOffset + curDistancePosition], index->lists[ivfId].get_candidate_id()[v], simi[0]) << endl;
                            heap_replace_top<METRIC_L2>(k, simi, idxi, distanceBuffer[queryOffset + curDistancePosition], index->lists[ivfId].get_candidate_id()[v]);
                        }
                    }
                }
                curDistancePosition++;
                
            }
        }
        // cout << curDistancePosition << " " << queryCompareSize[q] << endl;
        assert(curDistancePosition == queryCompareSize[q]);
        sort_result(METRIC_L2, k, simi, idxi);
    }

    totalSkip += skip;
    skipRates[blockId] = (double)skip / totalQueryCompareSize * 100;
    totalCompare += totalQueryCompareSize;

    searchTime += searchWatch.watch.elapsedSeconds();
    searchWatch.print(format("node({}) search main loop block({}) skip:{:.1f}%", rank, blockId, (double)skip / totalQueryCompareSize * 100));

    uniWatch.print(format("worker({}) -> block({}) -> worker({}) transfer started", rank, blockId, sendNextWorker[blockId]), false);
    MyStopWatch watch(true);
    if(shouldSendHeap(blockId)) {
        // cout << "send Heap" << blockId << endl;
        // printVector(idHeap[blockId].get(), blockSize * k, RED);
        // MPI_Isend(distanceHeap[blockId].get(), blockSize * k, MPI_FLOAT, sendNextWorker[blockId], 0, MPI_COMM_WORLD, &sendDistanceRequests[blockId]);
        // MPI_Isend(idHeap[blockId].get(), blockSize * k, MPI_INT64_T, sendNextWorker[blockId], 1, MPI_COMM_WORLD, &sendIdRequests[blockId]);
        MPI_Send(distanceHeap[blockId].get(), blockSize * k, MPI_FLOAT, sendNextWorker[blockId], blockId, MPI_COMM_WORLD);
        MPI_Send(idHeap[blockId].get(), blockSize * k, MPI_INT64_T, sendNextWorker[blockId], blockId, MPI_COMM_WORLD);


        // init_result(METRIC_L2, blockSize * k, distanceHeap.get(), idHeap.get());
    } else {


    // uniWatch.print(format("ready to perform node({}) -> block({}) -> node({})", rank, blockId, sendNextWorker[blockId]), false);
    // if(totalQueryCompareSize > INT_MAX) {
    //     cout << "> INT_MAX" << endl;
    //     idx_t sizeToSend = totalQueryCompareSize;
    //     while(sizeToSend > INT_MAX) {
    //         MPI_Request request;
    //         MPI_Isend(distanceBuffer + totalQueryCompareSize - sizeToSend, INT_MAX, MPI_FLOAT, sendNextWorker[blockId], blockId, MPI_COMM_WORLD, &request);
    //         sendRequests.push_back(request);
    //         sizeToSend -= INT_MAX;
    //     }
    //     MPI_Isend(distanceBuffer + totalQueryCompareSize - sizeToSend, sizeToSend, MPI_FLOAT, sendNextWorker[blockId], blockId, MPI_COMM_WORLD, &sendRequests[blockId]);
    // } else {
        // MPI_Isend(distanceBuffer, totalQueryCompareSize, MPI_FLOAT, sendNextWorker[blockId], blockId, MPI_COMM_WORLD, &sendRequests[blockId]);
        distanceBufferPool->ISendSplit(distanceBuffer, blockId, totalQueryCompareSize, sendNextWorker[blockId], sendRequests[blockId]);
    // }
    }
    waitTime += watch.watch.elapsedSeconds();
    watch.print(format("worker({}) -> block({}) -> worker({}) transfer time", rank, blockId, sendNextWorker[blockId]));


    // searchWatch.print(format("node({}) search finish block({}) skip:{:.1f}%", rank, blockId, (double)skip / totalQueryCompareSize * 100));

   

    // uniWatch.print(format("finish node({}) search block({}) skip:{:.1f}%", rank, blockId, (double)skip / totalQueryCompareSize * 100), false);

}

void BaseWorker::init(int rank) {
        MyStopWatch watch(true);

        this->rank = rank;
        // this->index = index;

        // 1.InitInfo
        MPI_Recv(&info, sizeof(InitInfo), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // info.print();
        // IVF IDs are provided by startIVFid, arranged in order

        index = std::make_unique<Index>(info.d, info.nlist, info.nprobe);

        watch.print("index");

        // 2. IVF sizes and vector representation of IVF
        listSizes = std::make_unique<size_t[]>(info.ivfCount);


        listCodes = vector<std::unique_ptr<float[]>>(info.ivfCount); 
        listIds = vector<std::unique_ptr<size_t[]>>(info.ivfCount); 

        watch.print("malloc");

        MPI_Recv(listSizes.get(), info.ivfCount * sizeof(size_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        watch.print("listSize");

#pragma omp parallel for
        for (size_t i = 0; i < info.ivfCount; i++) {
            index->lists[i].reset(listSizes[i], info.d, 0);
        }

        for (size_t i = 0; i < info.ivfCount; i++) {
            listCodes[i] = std::make_unique<float[]>(listSizes[i] * info.d);
            listIds[i] = std::make_unique<size_t[]>(listSizes[i]);
        }
        watch.print("lists");

        size_t totalNb = 0;
        for(int i = 0; i < info.ivfCount; i++) {
            totalNb += listSizes[i];
        }


        for (size_t i = 0; i < info.ivfCount; i++) {
            MPI_Recv(listCodes[i].get(), listSizes[i] * info.d , MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            copy_n(listCodes[i].get(), listSizes[i] * info.d, index->lists[i].candidate_codes.get());
            MPI_Recv(listIds[i].get(), listSizes[i] * sizeof(size_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            copy_n(listIds[i].get(), listSizes[i], index->lists[i].candidate_id.get());
            // MPI_Recv(curCodes, listSizes[i] * info.d , MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // MPI_Recv(curIds, listSizes[i] * sizeof(size_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // curCodes += listSizes[i] * info.d;
            // curIds += listSizes[i];
        }
        // for (size_t i = 0; i < info.ivfCount; i++) {
        //     MPI_Recv(index->lists[i].candidate_codes.get(), listSizes[i] * info.d , MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //     MPI_Recv(index->lists[i].candidate_id.get(), listSizes[i] * sizeof(size_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // }
        watch.print("listCodes and listIds");

        MPI_Bcast(index->centroid_codes.get(), info.nlist * info.d, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(index->centroid_ids.get(), info.nlist , MPI_INT64_T, 0, MPI_COMM_WORLD);

        // int presumeNq = 1000;
        int presumeK = 100;
        this->querys = std::make_unique<float[]>(presumeNq * info.d);
        listidqueries = std::make_unique<idx_t[]>(presumeNq * info.nprobe);  
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
        if(nq * k > presumeK * presumeNq) {
            cerr << "presumeNq is too small" << endl;
            exit(1);
        }
        uniWatch.print(format("node {} nq", rank), false);
        MPI_Bcast(querys.get(), nq * info.d, MPI_FLOAT, 0, MPI_COMM_WORLD);
        uniWatch.print(format("node {} querys", rank), false);

        // IDs of the nearest nprobe clusters for the queries
        MPI_Bcast(listidqueries.get(), nq * info.nprobe, MPI_INT64_T, 0, MPI_COMM_WORLD);
        uniWatch.print(format("node {} listidqueries", rank), false);

        heapTops = std::make_unique<float[]>(presumeNq);
        MPI_Bcast(heapTops.get(), nq, MPI_FLOAT, 0, MPI_COMM_WORLD);
        uniWatch.print(format("node {} heapTops", rank), false);

        // queryCompareSize,queryCompareSizePreSum
        // queryCompareSize = std::make_unique<size_t[]>(nq);
        // MPI_Bcast(queryCompareSize.get(), nq, MPI_INT64_T, 0, MPI_COMM_WORLD);
        // MPI_Bcast(queryCompareSizePreSum.get(), (nq + 1), MPI_INT64_T, 0, MPI_COMM_WORLD);
        // uniWatch.print(format("node {} queryCompareSize", rank), false);

        // Max heap

        watch.print(format("Node {} Init", rank));
        uniWatch.print(format("node {} finish init", rank), false);

    }

void BaseWorker::search(bool cut) {
    this->cut = cut;
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


                // cout << format("thread {} {}", omp_get_thread_num(), start) << endl;
                // MyStopWatch w;
                single_thread_search_simple(end - start, querys.get() + start * info.d, k, distances.get() + start * k, labels.get() + start * k, listidqueries.get() + start * info.nprobe, heapTops.get() + start);
                // single_thread_search_fast(end - start, querys.get() + start * info.d, k, distances.get() + start * k, labels.get() + start * k, listidqueries.get() + start * info.nprobe);
                // w.print(format("single thread {}", i));
            }
        }


    searchTime += watch.watch.elapsedSeconds();
    watch.print(format("node {} search", rank));

    uniWatch.print(format("node {} search", rank), false);
    MPI_Send(distances.get(), k * nq, MPI_FLOAT, 0, 0, MPI_COMM_WORLD); 
    MPI_Send(labels.get(), k * nq, MPI_INT64_T, 0, 0, MPI_COMM_WORLD); 
    uniWatch.print(format("node {} send", rank), false);

    waitTime += watch.watch.elapsedSeconds();
    watch.print(format("node {} send", rank));

    double totalTime = totalWatch.watch.elapsedSeconds();
    totalWatch.print(format("Search Finished node({}), waitTime {} {:.2f}% searchTime {} {:.2f}%",
            rank, waitTime, 100 * waitTime / totalTime, searchTime, 100 * searchTime / totalTime));
            
    int count = 0;
    for(int i = 0; i < nq * info.nprobe; i++) {
        auto ivfId = listidqueries[i];
        if((ivfId >= info.startIVFId && ivfId < info.startIVFId + info.ivfCount)) {
            count++;
        }
    }
    cout << format("node {} counted {}%", rank, 100.0 * count / nq / info.nprobe) << endl;
}

void BaseWorker::single_thread_search_simple(size_t n, const float* queries, size_t k, float* distances, idx_t* labels, idx_t* listidqueries, float* heapTops) {
    float* simi = distances; 
    idx_t* idxi = labels; 
    idx_t* listids = listidqueries;             
    size_t cutCount = 0, totalCount = 0;
    // MyStopWatch w;
    for (size_t i = 0; i < n; i++) {
        // scaner->set_query(queries + i * info.d);
        // printVector(listids, info.nprobe, GREEN);
        for (size_t j = 0; j < info.nprobe; j++) {

            idx_t ivfId = listids[j];
            if(!(ivfId >= info.startIVFId && ivfId < info.startIVFId + info.ivfCount)) {
                continue;
            }
            // idx_t index = ivfId - info.startIVFId;
            IVF& list = index->lists[ivfId - info.startIVFId];
            // size_t listSize = listSize[index];
            // float* codes = listCodes[index].get();
            // size_t* ids = listIds[index].get();
            size_t listSize = list.get_list_size();
            totalCount += listSize;
            float* codes = list.candidate_codes.get();
            size_t* ids = list.candidate_id.get();

            // MyStopWatch wa;
            // scaner->lite_scan_codes(listSize, codes, ids, simi, idxi);

            if(cut) {
                for (size_t v = 0; v < list.get_list_size(); v++) {
                    const float* candicate = list.get_candidate_codes() + v * index->d;
                    float dis = 0;
                    // dis = calculatedEuclideanDistance(queries + i * index->d, candicate, index->d);
                    int numCheck = 4;
                    size_t checkDim = index->d / numCheck; 
                    const float* query = queries + i * index->d;
                    for(int check = 0; check < numCheck; check++) {
                        if(check == numCheck - 1) {
                            dis += calculatedEuclideanDistance(query + checkDim * check, candicate + checkDim * check, index->d - check * checkDim);
                        } else {
                            dis += calculatedEuclideanDistance(query + checkDim * check, candicate + checkDim * check, checkDim);
                            if(dis > heapTops[i]) {
                                dis = INFINITY;
                                cutCount += numCheck - check - 1;
                                break;
                            }
                        }
                    }
                    if (dis < simi[0]) {
                        heap_replace_top<METRIC_L2>(k, simi, idxi, dis, list.get_candidate_id()[v]);
                    }
                }
            } else {
                for (size_t v = 0; v < list.get_list_size(); v++) {
                    const float* candicate = list.get_candidate_codes() + v * index->d;
                    float dis = 0;
                    dis = calculatedEuclideanDistance(queries + i * index->d, candicate, index->d);
                    if (dis < simi[0]) {
                        heap_replace_top<METRIC_L2>(k, simi, idxi, dis, list.get_candidate_id()[v]);
                    }
                }
            }

           
        }
        simi += k;
        idxi += k;
        listids += info.nprobe;
        
        // w.print(format("q {}", i));
    }
    // cout << RED << format("{}%", double(cutCount) * 25 / totalCount)<< RESET << endl;
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
        MPI_Recv(index->lists[i].candidate_id.get(), listSizes[i] * sizeof(size_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    // for (size_t i = info.startIVFId; i < info.startIVFId + info.ivfCount; i++) {
    //     MPI_Recv(index->lists[i].candidate_codes.get(), listSizes[i] * info.block_dim , MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // }

    auto listCodesBuffer = vector<std::unique_ptr<float[]>>(info.nlist);
    for (size_t i = info.startIVFId; i < info.startIVFId + info.ivfCount; i++) {
        listCodesBuffer[i] = std::make_unique<float[]>(listSizes[i] * info.d);
        MPI_Recv(listCodesBuffer[i].get(), listSizes[i] * info.d , MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
    
    // blockDistancesSize = presumeNq / info.blockCount / info.groupCount * (info.nb / info.nlist) * info.nprobe * 2;
    blockDistancesSize = 2 * presumeNq / info.blockCount / info.groupCount * info.nb * info.nprobe / info.nlist;
    // blockDistancesSize = 2 * presumeNq / info.blockCount / info.groupCount * info.nb / info.teamSize * info.nprobe / info.nlist;
    distancesForBlocks = vector<vector<std::unique_ptr<float[]>>>(info.groupCount);
    for (size_t i = 0; i < distancesForBlocks.size(); i++) {
        distancesForBlocks[i] = vector<std::unique_ptr<float[]>>(info.blockCount);
        for (size_t j = 0; j < distancesForBlocks[i].size(); j++) {
            distancesForBlocks[i][j] = std::make_unique<float[]>(blockDistancesSize);
        }
    }


    disRequests = vector<vector<MPI_Request>>(info.blockCount);
    for(int i = 0; i < disRequests.size(); i++) {
        disRequests[i] = vector<MPI_Request>(1);
    }
    sendRequests = vector<MPI_Request>(info.blockCount);
    sendDistanceRequests = vector<MPI_Request>(info.blockCount);
    sendIdRequests = vector<MPI_Request>(info.blockCount);

    skipRates = vector<double>(info.blockCount, 0);

    // cout << "finish init" << endl;


}

void GroupWorker::receiveQuery() {

    MPI_Barrier(MPI_COMM_WORLD); // Corresponds to the barrier at the end of preSearch

    uniWatch = MyStopWatch(false, "uniWatch", MAG);
    uniWatch.print(format("node {} cross barrier", rank), false);
    // nq, querys
    MPI_Bcast(&nq, sizeof(nq), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, sizeof(k), MPI_BYTE, 0, MPI_COMM_WORLD);
    if((double)nq * k > presumeK * presumeNq) {
        cerr << "presumeNq * k is too small" << endl;
        exit(1);
    }
    if(nq > presumeNq) {
        cerr << "presumeNq is too small" << endl;
        exit(1);
    }
    // uniWatch.print(format("node {} nq k {} ", rank, k), false);

    MPI_Bcast(index->originalQuery.get(), nq * info.d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    addQuerys(index->originalQuery.get(), nq);

    // uniWatch.print(format("node {} querys", rank), false);

    MPI_Bcast(listidqueries.get(), nq * info.nprobe, MPI_INT64_T, 0, MPI_COMM_WORLD);

    // uniWatch.print(format("node {} listidqueries", rank), false);

    // queryCompareSize,queryCompareSizePreSum
    MPI_Recv(queryCompareSize.get(), nq, MPI_INT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // queryCompareSizePreSum = std::make_unique<size_t[]>(nq + 1);
    MPI_Recv(queryCompareSizePreSum.get(), (nq + 1), MPI_INT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // uniWatch.print(format("node {} queryCompareSize", rank), false);
    // printVector(queryCompareSize.get(), nq, BLUE);
    // printVector(queryCompareSizePreSum.get(), nq + 1, GREEN);

    groupSize = nq / info.groupCount;        
    blockSize = groupSize / info.blockCount;

    // cout << "finish receive query" << endl;
    uniWatch.print(format("node {} finish Receving", rank), false);

}
void GroupWorker::search(bool cut, bool minorCut) {

    this->cut = cut;
    this->minorCut = minorCut;

    MyStopWatch totalWatch(true, "Total Watch");

    // Initialize max heap
    MPI_Bcast(heapTops.get(), nq, MPI_FLOAT, 0, MPI_COMM_WORLD);
    uniWatch.print(format("node {} received initial heapTops", rank), false);

    for(int i = 0; i < info.groupCount; i++) {
        size_t groupId = groupSearchOrder.workerSearchOrder[info.teamId][i];
        // If an updated heapTop needs to be received
        size_t sender = groupSearchOrder.recvPrevWorker[info.teamId][groupId];
        if(sender != 0) {
            MyStopWatch waitWatch(true, "Wait time between master and worker");
            MPI_Recv(heapTops.get() + groupId * groupSize, groupSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // MPI_Recv(listSizes.get(), info.nlist * sizeof(size_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(distanceHeap[groupId].get(), groupSize * k, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(idHeap[groupId].get()      , groupSize * k, MPI_INT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            waitTime += waitWatch.watch.elapsedSeconds();
            waitWatch.print(format("node {} received group {} Heap", rank, groupId));  // Record time for receiving heap
            uniWatch.print(format("node {} received group {}", rank, groupId), false);
            
        }

        searchGroup(groupId);

        reset();
    }

    // totalWatch.print(format("node {} finished all group searches", rank));
    double totalTime = totalWatch.watch.elapsedSeconds();
    double otherTime = totalTime - waitTime - searchTime;
    totalWatch.print(format("node({}) finished all group searches, waitTime {:.3f}s {:.2f}% searchTime {:.3f}s {:.2f}% otherTime {:.3f}s {:.2f}%", 
        rank, waitTime, 100 * waitTime / totalTime, searchTime, 100 * searchTime / totalTime, otherTime, 100 * otherTime / totalTime));

    
}
size_t GroupWorker::getSender(idx_t groupId, idx_t blockId) {
    if(blockSearchOrder.getRecvPrevWorker(info.rankInsideTeam, blockId) == 0) {
        return 0;
    }
    return blockSearchOrder.getRecvPrevWorker(info.rankInsideTeam, blockId) + (info.teamId - 1) * info.teamSize;
}
size_t GroupWorker::getReceiver(idx_t groupId, idx_t blockId) {
    if(blockSearchOrder.getSendNextWorker(info.rankInsideTeam, blockId) == 0) {
        return 0;
    }
    return blockSearchOrder.getSendNextWorker(info.rankInsideTeam, blockId) + (info.teamId - 1) * info.teamSize;
}

void GroupWorker::searchGroup(idx_t groupId) 
{
    // cout << format("node {} groupid {}", rank, groupId) << endl;
    uniWatch.print(format("node {} starting searchGroup {}", rank, groupId), false);
    MyStopWatch groupWatch(true, "Group search watch");
   
    int searchedBlockCount = 0;
    vector<bool> isBlockSearched = vector<bool>(info.blockCount);


    //TODO FIX DISQUEST
    for (size_t blockId = 0; blockId < info.blockCount; blockId++) {
        size_t sender = getSender(groupId, blockId);
        if(sender != 0) {
            MPI_Irecv(distancesForBlocks[groupId][blockId].get(), getBlockQueryCompareSize(groupId, blockId), MPI_FLOAT, sender, getTag(groupId, blockId, info.blockCount), MPI_COMM_WORLD, &disRequests[blockId][0]);
            // MPI_Irecv(distanceBufferPool->getBuffer(blockId), getTotalQueryCompareSize(blockId), MPI_FLOAT, sender, blockId, MPI_COMM_WORLD, &disRequests[blockId]);
            // bool suc = distanceBufferPool->IRecv(blockId, getTotalQueryCompareSize(blockId), sender, disRequests[blockId]);
            // if(!suc) {
            //     break;
            // }
            // cout << format("node({}) waiting for block({}) from node({})", rank, blockId, sender) << endl;
        } 
    }

    MyStopWatch watch(true);
    idx_t failCount = 0;
    while(searchedBlockCount < info.blockCount)
    for (size_t blockId = 0; blockId < info.blockCount; blockId++) {
        if(isBlockSearched[blockId]) {
            continue;
        }
        size_t sender = getSender(groupId, blockId);
        if(sender == 0) {

            searchBlock(blockId, groupId);
            isBlockSearched[blockId] = true;
            searchedBlockCount++;

            watch.reset();
            if(blockSend && !shouldSendHeap(blockId)) {
                MPI_Wait(&sendRequests[blockId], MPI_STATUS_IGNORE);
                waitTime += watch.watch.elapsedSeconds();
                watch.print(format("node({}) block {} MPI_Wait", rank, blockId));
            }

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
                waitTime += watch.watch.elapsedSeconds();
                watch.print(format("node {} waiting , now received block {} fail {}", rank, blockId, failCount));
                uniWatch.print(format("node {} received block {} Test failed count {}", rank, blockId, failCount), false);

                searchBlock(blockId, groupId);

                searchedBlockCount++;
                isBlockSearched[blockId] = true;
                failCount = 0;

                watch.reset();
                if(blockSend && !shouldSendHeap(blockId)) {
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
    // double totalTime = groupWatch.watch.elapsedSeconds();
    // double otherTime = totalTime - waitTime - searchTime;
    // groupWatch.print(format("node({}) finished search for Group{}, pruning rate:{:.1f}%, waitTime {:.3f}s {:.2f}% searchTime {:.3f}s {:.2f}% otherTime {:.3f}s {:.2f}%",
    //     rank, groupId, (double)totalSkip / totalCompare * 100, waitTime, 100 * waitTime / totalTime, searchTime, 100
    //     * searchTime / totalTime, otherTime, 100 * otherTime / totalTime));

    if(!blockSend) {
        for(size_t blockId = 0; blockId < info.blockCount; blockId++) {
            if(!shouldSendHeap(blockId)) {
                MPI_Wait(&sendRequests[blockId], MPI_STATUS_IGNORE);
            }
        }
    } 
}

void GroupWorker::single_thread_searchBlock(size_t n, size_t blockId, size_t groupId, float* queries, float* distanceBuffer, float* simi, idx_t* idxi, idx_t* listidqueries) {

    size_t curDistancePosition = 0;  // Position within the result of a query vector
    size_t skip = 0;
    for (size_t q = 0; q < n; q++) {
        for (size_t i = 0; i < info.nprobe; i++) {
            idx_t ivfId = listidqueries[q * info.nprobe + i];

            if(ivfId >= info.startIVFId && ivfId < info.startIVFId + info.ivfCount) {

                IVF& list = index->lists[ivfId];

                for (size_t v = 0; v < listSizes[ivfId]; v++) {

                    const float* codes = list.get_candidate_codes() + v * info.block_dim;

                    if(cut) {
                        // if(distancesForBlocks[blockId][queryOffset + curDistancePosition] == INFINITY) {
                        if(distanceBuffer[curDistancePosition] == INFINITY) {
                            skip++;
                        } else {
                            float dis = calculatedEuclideanDistance(querys.get() + q * info.block_dim,
                                                                    codes,
                                                                    info.block_dim);

                            distanceBuffer[curDistancePosition] += dis;

                            if (distanceBuffer[curDistancePosition] > heapTops[q]) {
                                distanceBuffer[curDistancePosition] = INFINITY;
                            } else {
                                if(shouldSendHeap(blockId)) {
                                    if (distanceBuffer[curDistancePosition] < simi[0]) {
                                        heap_replace_top<METRIC_L2>(k, simi, idxi, distanceBuffer[curDistancePosition], index->lists[ivfId].get_candidate_id()[v]);
                                    }
                                } 
                            }
                        }
                    } else {
                        float dis = calculatedEuclideanDistance(querys.get() + q * info.block_dim,
                                                                index->lists[ivfId].get_candidate_codes() + v * info.block_dim,
                                                                info.block_dim);
                        // cout << " " << queryOffset + curDistancePosition  << endl;
                        // assert(queryOffset + curDistancePosition < totalQueryCompareSize);
                        // distanceBuffer[queryOffset + curDistancePosition] += dis;
                        // if(info.blockCount != 1) {
                        //     dis += distanceBuffer[curDistancePosition];
                        // }
                        if(shouldSendHeap(blockId)) {
                            if (dis < simi[0]) {
                                heap_replace_top<METRIC_L2>(k, simi, idxi, dis, index->lists[ivfId].get_candidate_id()[v]);
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


    // float* distanceBuffer = distanceBufferPool->getBuffer(blockId);
    float* distanceBuffer = distancesForBlocks[groupId][blockId].get();

    size_t queryStart = getQueryOffset(groupId, blockId);
    idx_t totalQueryCompareSize = getBlockQueryCompareSize(groupId,blockId);
        
    // cout << RED << rank << "node search: blockId:" << blockId << " totalCompareSize:" << totalQueryCompareSize << RESET
    //      << endl;


    if (totalQueryCompareSize > blockDistancesSize) {
        
        cerr << format("Error blockDistancesSize too small {} < {}", blockDistancesSize , totalQueryCompareSize) << endl;
        exit(1);
    }

    MyStopWatch searchWatch(false, "searchBlock");

    size_t skip = 0;

    size_t nt = std::min(static_cast<size_t>(omp_get_max_threads()), blockSize);
#pragma omp parallel for reduction(+:skip)
    for (size_t q = queryStart; q < queryStart + blockSize; q++) {
        size_t queryOffset =
            queryCompareSizePreSum[q] - queryCompareSizePreSum[queryStart];  // Offset for storing the result of query q

        float* query = querys.get() + q * info.block_dim;
        float* simi = distanceHeap[groupId].get() + k * (q - queryStart + blockId * blockSize);
        idx_t* idxi = idHeap[groupId].get()       + k * (q - queryStart + blockId * blockSize);
        size_t curDistancePosition = 0;  /// Position within the result of a query vector

        for (size_t i = 0; i < info.nprobe; i++) {

            idx_t ivfId = listidqueries[q * info.nprobe + i];

            if(ivfId >= info.startIVFId && ivfId < info.startIVFId + info.ivfCount) {

                IVF& list = index->lists[ivfId];

                for (size_t v = 0; v < listSizes[ivfId]; v++) {

                    const float* codes = list.get_candidate_codes() + v * info.block_dim;

                    if(cut) {
                        // if(distancesForBlocks[blockId][queryOffset + curDistancePosition] == INFINITY) {
                        if(distanceBuffer[queryOffset + curDistancePosition] == INFINITY) {
                            skip++;
                        } else {

                            if(minorCut) {

                                float dis = distanceBuffer[queryOffset + curDistancePosition];
                                auto candicate = codes;
                                auto query = querys.get() + q * info.block_dim;
                                int numCheck = 4;
                                size_t checkDim = info.block_dim / numCheck; 

                                for(int check = 0; check < numCheck; check++) {
                                    if(check == numCheck - 1) {
                                        dis += calculatedEuclideanDistance(query + checkDim * check, candicate + checkDim * check, info.block_dim- check * checkDim);
                                    } else {
                                        dis += calculatedEuclideanDistance(query + checkDim * check, candicate + checkDim * check, checkDim);
                                        if(dis > heapTops[q]) {
                                            dis = INFINITY;
                                            // cutCount += numCheck - check - 1;
                                            break;
                                        }
                                    }
                                }

                                distanceBuffer[queryOffset + curDistancePosition] = dis;

                                if (distanceBuffer[queryOffset + curDistancePosition] > heapTops[q]) {
                                    distanceBuffer[queryOffset + curDistancePosition] = INFINITY;
                                } else {
                                    if(shouldSendHeap(blockId)) {
                                        if (distanceBuffer[queryOffset + curDistancePosition] < simi[0]) {
                                            heap_replace_top<METRIC_L2>(k, simi, idxi, distanceBuffer[queryOffset + curDistancePosition], index->lists[ivfId].get_candidate_id()[v]);
                                        }
                                    } 
                                }
                            } else {
                                float dis = calculatedEuclideanDistance(querys.get() + q * info.block_dim,
                                                                        codes,
                                                                        info.block_dim);

                                distanceBuffer[queryOffset + curDistancePosition] += dis;

                                if (distanceBuffer[queryOffset + curDistancePosition] > heapTops[q]) {
                                    distanceBuffer[queryOffset + curDistancePosition] = INFINITY;
                                } else {
                                    if(shouldSendHeap(blockId)) {
                                        if (distanceBuffer[queryOffset + curDistancePosition] < simi[0]) {
                                            heap_replace_top<METRIC_L2>(k, simi, idxi, distanceBuffer[queryOffset + curDistancePosition], index->lists[ivfId].get_candidate_id()[v]);
                                        }
                                    } 
                                }
                            }
                            
                            // float dis = calculatedEuclideanDistance(querys.get() + q * info.block_dim,
                            //                                         codes,
                            //                                         info.block_dim);

                            // distanceBuffer[queryOffset + curDistancePosition] += dis;

                            // if (distanceBuffer[queryOffset + curDistancePosition] > heapTops[q]) {
                            //     distanceBuffer[queryOffset + curDistancePosition] = INFINITY;
                            // } else {
                            //     if(shouldSendHeap(blockId)) {
                            //         if (distanceBuffer[queryOffset + curDistancePosition] < simi[0]) {
                            //             //比堆顶
                            //             heap_replace_top<METRIC_L2>(k, simi, idxi, distanceBuffer[queryOffset + curDistancePosition], index->lists[ivfId].get_candidate_id()[v]);
                            //         }
                            //     } 
                            // }
                        }
                    } else {
                        float dis = calculatedEuclideanDistance(query,
                                                                index->lists[ivfId].get_candidate_codes() + v * info.block_dim,
                                                                info.block_dim);
                        // cout << " " << queryOffset + curDistancePosition  << endl;
                        // assert(queryOffset + curDistancePosition < totalQueryCompareSize);
                        // distanceBuffer[queryOffset + curDistancePosition] += dis;
                        distanceBuffer[queryOffset + curDistancePosition] += dis;
                        if(shouldSendHeap(blockId)) {
                            dis = distanceBuffer[queryOffset + curDistancePosition];
                            if (dis < simi[0]) {
                                // cout << format("q {} replace {} {} top {}", q, distanceBuffer[queryOffset + curDistancePosition], index->lists[ivfId].get_candidate_id()[v], simi[0]) << endl;
                                heap_replace_top<METRIC_L2>(k, simi, idxi, dis, index->lists[ivfId].get_candidate_id()[v]);
                            }
                        }
                    }
                    curDistancePosition++;
                    
                }
            }
        }
        // cout << curDistancePosition << " " << queryCompareSize[q] << endl;
        // assert(curDistancePosition == queryCompareSize[q]);
        // sort_result(METRIC_L2, k, simi, idxi);
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
    if(shouldSendHeap(blockId)) {
        MPI_Send(distanceHeap[groupId].get() + blockId * blockSize * k, blockSize * k, MPI_FLOAT  , 0, tag, MPI_COMM_WORLD);
        MPI_Send(idHeap[groupId].get()       + blockId * blockSize * k, blockSize * k, MPI_INT64_T, 0, tag, MPI_COMM_WORLD);
        // init_result(METRIC_L2, blockSize * k, distanceHeap.get(), idHeap.get());
    } else {
        MPI_Isend(distanceBuffer, totalQueryCompareSize, MPI_FLOAT, getReceiver(groupId, blockId), tag, MPI_COMM_WORLD, &sendRequests[blockId]);
    }
    waitTime += watch.watch.elapsedSeconds();
    watch.print(format("worker({}) -> block({}) -> worker({}) transmission time", rank, blockId, getReceiver(groupId, blockId)));

    // Distance buffer should not be released until transmission is complete
    // distanceBufferPool->releaseBuffer(blockId);
}

}  // namespace harmony