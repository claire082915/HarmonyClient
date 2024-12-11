#include "node.h"
namespace tribase {

void Node::addIVFs(IVF* ivfs, size_t ivfCnt) {

        assert(id != 0 && block_dim != 0);

        this->ivfs = std::make_unique<IVF[]>(ivfCnt);
// #pragma omp parallel for
        for (size_t i = 0; i < ivfCnt; i++) {
            // cout << ivfs[i].d << endl;
            this->ivfs[i].reset(ivfs[i].get_list_size(), block_dim, 0);
// #pragma omp parallel for
            copy_n_partial_vector(ivfs[i].candidate_codes.get(), this->ivfs[i].candidate_codes.get(), d, block_dim, (id - 1) * block_dim, ivfs[i].get_list_size());
        }

}
}