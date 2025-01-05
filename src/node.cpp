#include "node.h"
#include "Index.h"
namespace tribase {

// void Worker::addIVFs(vector<std::unique_ptr<float[]>>& listCodesBuffer) {
//     assert(rank != 0 && info.block_dim != 0);

//     listCodes = vector<std::unique_ptr<float[]>>(info.nlist);
//     // #pragma omp parallel for
//     for (size_t i = 0; i < info.nlist; i++) {
//         this->listCodes[i] = std::make_unique<float[]>(listSizes[i] * info.d);
//         copy_n_partial_vector(listCodesBuffer[i].get(), this->listCodes[i].get(), info.d, info.block_dim,
//                               (rank - 1) * info.block_dim, listSizes[i]);
//     }

// }
void Worker::init(int rank) {
    MyStopWatch watch(false);

    this->rank = rank;

    // InitInfo
    MPI_Bcast(&info, sizeof(InitInfo), MPI_BYTE, 0, MPI_COMM_WORLD);
    info.print();

    index = std::make_unique<Index>(info.d, info.nlist, info.nprobe);

    // IVF的大小，IVF的向量表示
    listSizes = std::make_unique<size_t[]>(info.nlist);
    MPI_Bcast(listSizes.get(), info.nlist * sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);

#pragma omp parallel for
    for (size_t i = 0; i < info.nlist; i++) {
        index->lists[i].reset(listSizes[i], info.block_dim, 0);
    }
    watch.print("lists");
        
    for (size_t i = 0; i < info.nlist; i++) {
        MPI_Bcast(index->lists[i].candidate_id.get(), listSizes[i] * sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    auto listCodesBuffer = vector<std::unique_ptr<float[]>>(info.nlist);
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

    // Search顺序
    blockSearchOrder = std::make_unique<idx_t[]>(info.blockCount);   // search block的顺序，第i个元素是第i个要进行search的blockId
    MPI_Recv(blockSearchOrder.get(), info.blockCount, MPI_INT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // cout << "Q" << rank << endl;
    // printVector(blockSearchOrder.get(), info.blockCount, BLUE);

    //提前malloc
    int presumeNq = 10000;
    // std::unique_ptr<float[]> querysBuffer = std::make_unique<float[]>(presumeNq * info.d);
    index->originalQuery = std::make_unique<float[]>(presumeNq * info.d);
    this->querys = std::make_unique<float[]>(presumeNq * info.block_dim);
    listidqueries = std::make_unique<idx_t[]>(presumeNq * info.nprobe);  // 最近的nprobe个聚类中心的id
    index->heapTops = std::make_unique<float[]>(presumeNq);
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
    // MPI_Bcast(querysBuffer.get(), nq * info.d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(index->originalQuery.get(), nq * info.d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    uniWatch.print(format("node {} querys", rank), false);
    addQuerys(index->originalQuery.get(), nq);
    // addQuerys(querysBuffer.get(), nq);
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
    MPI_Bcast(index->heapTops.get(), nq, MPI_FLOAT, 0, MPI_COMM_WORLD);
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

void Worker::searchBlock(size_t blockId, bool cut) {
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

    uniWatch.print(format("node {} index search start", rank), false);

    Index::Param param;
    param.mode = Index::SearchMode::DIVIDE_DIM_WORKER;
    param.block_dim = info.block_dim;
    param.queryCompareSize = queryCompareSize.get();
    param.queryCompareSizePreSum = queryCompareSizePreSum.get();
    param.queryStart = queryStart;
    param.cut = cut;

    size_t skip = 0;
    index->search(blockSize, querys.get() + queryStart * info.block_dim, 0, distanceBuffer, NULL, 1, &param);

//     size_t nt = std::min(static_cast<size_t>(omp_get_max_threads()), blockSize);
// //     // cout << "node " << rank << " nt = " << nt << endl;
// #pragma omp parallel for num_threads(nt) reduction(+:skip)
// // #pragma omp parallel for num_threads(nt)
//     for (size_t q = queryStart; q < queryStart + blockSize; q++) {
//         // cout << "node " << rank << " nt = " << omp_get_num_threads() << endl;
//         size_t queryOffset =
//             queryCompareSizePreSum[q] - queryCompareSizePreSum[queryStart];  // 第q个查询的结果应该存的地址偏移量

//         size_t curDistancePosition = 0;  // 在一个查询向量的结果内
//         for (size_t i = 0; i < info.nprobe; i++) {
//             idx_t ivfId = listidqueries[q * info.nprobe + i];
//             for (size_t v = 0; v < listSizes[ivfId]; v++) {
//                 if(cut) {
//                     // if(distancesForBlocks[blockId][queryOffset + curDistancePosition] == INFINITY) {
//                     if(distanceBuffer[queryOffset + curDistancePosition] == INFINITY) {
//                         skip++;
//                     } else {
//                         float dis = calculatedEuclideanDistance(querys.get() + q * info.block_dim,
//                                                                 listCodes[ivfId].get() + v * info.block_dim,
//                                                                 info.block_dim);
//                         // cout << " " << queryOffset + curDistancePosition  << endl;
                        
//                         // assert(queryOffset + curDistancePosition < totalQueryCompareSize);
//                         distanceBuffer[queryOffset + curDistancePosition] += dis;
//                         if (distanceBuffer[queryOffset + curDistancePosition] > heapTops[q]) {
//                             distanceBuffer[queryOffset + curDistancePosition] = INFINITY;
//                         }
//                     }
//                     curDistancePosition++;
//                 } else {
//                     float dis = calculatedEuclideanDistance(querys.get() + q * info.block_dim,
//                                                             listCodes[ivfId].get() + v * info.block_dim,
//                                                             info.block_dim);
//                     // cout << " " << queryOffset + curDistancePosition  << endl;
//                     assert(queryOffset + curDistancePosition < totalQueryCompareSize);
//                     distanceBuffer[queryOffset + curDistancePosition] += dis;
//                     curDistancePosition++;
//                 }
                
//             }
//         }
//         // cout << curDistancePosition << " " << queryCompareSize[q] << endl;
//         assert(curDistancePosition == queryCompareSize[q]);
//     }

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

void BaseWorker::init(int rank) {
        MyStopWatch watch(true);

        this->rank = rank;
        // this->index = index;

        // 1.InitInfo
        MPI_Recv(&info, sizeof(InitInfo), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        info.print();
        // IVF的ID通过startIVFid给出, 顺序排列

        index = std::make_unique<Index>(info.d, info.nlist, info.nprobe);

        watch.print("index");

        // 2.IVF的大小，IVF的向量表示
        listSizes = std::make_unique<size_t[]>(info.ivfCount);
        MPI_Recv(listSizes.get(), info.ivfCount * sizeof(size_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


#pragma omp parallel for
        for (size_t i = 0; i < info.ivfCount; i++) {
            index->lists[i].reset(listSizes[i], info.d, 0);
        }
        watch.print("lists");

        size_t totalNb = 0;
        for(int i = 0; i < info.ivfCount; i++) {
            totalNb += listSizes[i];
        }

        for (size_t i = 0; i < info.ivfCount; i++) {
            MPI_Recv(index->lists[i].candidate_codes.get(), listSizes[i] * info.d , MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(index->lists[i].candidate_id.get(), listSizes[i] * sizeof(size_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        watch.print("listCodes and listIds");

        MPI_Bcast(index->centroid_codes.get(), info.nlist * info.d, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(index->centroid_ids.get(), info.nlist , MPI_INT64_T, 0, MPI_COMM_WORLD);

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

void BaseWorker::search() {

    uniWatch.print(format("node {} index search start", rank), false);

    Index::Param oriParam;
    oriParam.mode = Index::SearchMode::ORIGINAL;
    oriParam.divideIVFVersionOriginal = true;
    oriParam.startIVFId = info.startIVFId;
    oriParam.ivfCount = info.ivfCount;

    index->search(nq, querys.get(), k, distances.get(), labels.get(), 1, &oriParam);

    uniWatch.print(format("node {} search", rank), false);

    MPI_Send(distances.get(), k * nq, MPI_FLOAT, 0, 0, MPI_COMM_WORLD); 
    MPI_Send(labels.get(), k * nq, MPI_INT64_T, 0, 0, MPI_COMM_WORLD); 

    uniWatch.print(format("node {} send", rank), false);
}
}  // namespace tribase
