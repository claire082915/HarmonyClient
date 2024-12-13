#include "node.h"
namespace tribase {

void Worker::addIVFs(vector<std::unique_ptr<float[]>>& listCodesBuffer) {
    assert(rank != 0 && block_dim != 0);

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
}  // namespace tribase