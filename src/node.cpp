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

void BaseWorker::search() {

    for(int i = 0; i < 3; i++) {
        // printVector(index->lists[i].candidate_id.get(), index->lists[i].get_list_size(), RED);
        // printVector(querys.get(), nq, RED);
    }
    uniWatch.print(format("node {} index search start", rank), false);
    Index::Param oriParam;
    oriParam.mode = Index::SearchMode::ORIGINAL;

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
