#include "node.h"
#include "Index.h"
namespace tribase {

void Worker::addIVFs(vector<std::unique_ptr<float[]>>& listCodesBuffer) {
    assert(rank != 0 && info.block_dim != 0);

    listCodes = vector<std::unique_ptr<float[]>>(info.nlist);
    // #pragma omp parallel for
    for (size_t i = 0; i < info.nlist; i++) {
        this->listCodes[i] = std::make_unique<float[]>(listSizes[i] * info.d);
        copy_n_partial_vector(listCodesBuffer[i].get(), this->listCodes[i].get(), info.d, info.block_dim,
                              (rank - 1) * info.block_dim, listSizes[i]);
        // cout << rank << " " << i <<" ";
        // printVector(listCodes[i].get(), info.block_dim, BLUE);
    }

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

        // listCodes = vector<std::unique_ptr<float[]>>(info.ivfCount); //每个聚类对应的codes
        // listIds = vector<std::unique_ptr<size_t[]>>(info.ivfCount); //每个聚类对应的ids
        
        // auto curCodes = listCodes.get();
        // auto curIds = listIds.get();
        for (size_t i = 0; i < info.ivfCount; i++) {
            MPI_Recv(index->lists[i].candidate_codes.get(), listSizes[i] * info.d , MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(index->lists[i].candidate_id.get(), listSizes[i] * sizeof(size_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // listCodes[i] = std::make_unique<float[]>(listSizes[i] * info.d);
            // listIds[i] = std::make_unique<size_t[]>(listSizes[i]);
            // MPI_Recv(listCodes[i].get(), listSizes[i] * info.d , MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // MPI_Recv(listIds[i].get(), listSizes[i] * sizeof(size_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // MPI_Recv(curCodes, listSizes[i] * info.d , MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // MPI_Recv(curIds, listSizes[i] * sizeof(size_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // curCodes += listSizes[i] * info.d;
            // curIds += listSizes[i];
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

    cout << index->opt_level << " index "<< index->added_opt_level << endl;
    for(int i = 0; i < 3; i++) {
        // printVector(index->lists[i].candidate_id.get(), index->lists[i].get_list_size(), RED);
        // printVector(querys.get(), nq, RED);
    }
    uniWatch.print(format("node {} index search start", rank), false);
    Index::Param oriParam;
    oriParam.mode = Index::SearchMode::ORIGINAL;
    oriParam.divideIVFVersionOriginal = true;
    oriParam.startIVFId = info.startIVFId;
    oriParam.ivfCount = info.ivfCount;

    index->search(nq, querys.get(), k, distances.get(), labels.get(), 1, &oriParam);
// for(int i = 0; i < 3; i++) {
//                             // if(diffVector(labels.get() + i * k, labelsB.get() + i * k, k)) {
//                                 std::cout << "Q" << i << " " << std::endl;
//                                 printVector(distances.get() + i * k, k, BLUE);
//                                 printVector(labels.get() + i * k, k, BLUE);
//                             // }
//                         }
    uniWatch.print(format("node {} search", rank), false);
    MPI_Send(distances.get(), k * nq, MPI_FLOAT, 0, 0, MPI_COMM_WORLD); 
    MPI_Send(labels.get(), k * nq, MPI_INT64_T, 0, 0, MPI_COMM_WORLD); 
    uniWatch.print(format("node {} send", rank), false);
}
}  // namespace tribase
