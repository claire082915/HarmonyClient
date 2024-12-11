#ifndef NODE_H
#define NODE_H
#include <cstdlib>
#include <memory>
#include "IVF.h"
#include "cassert"
#include <algorithm>
#include "utils.h"
#include <vector>

namespace tribase {

using namespace std;
class Node
{
private:
    
public:
    class SearchBlock {
    public:
        size_t blockId; //当前是第几个Block, 第i个block对应第i组查询
    };
    //代表一次搜索请求中需要包含的信息

    size_t block_dim = 0;
    size_t d = 0;
    size_t id = 0;
    size_t blockCount = 0;
    size_t blockSize = 0;
    size_t nprobe = 0;
    std::unique_ptr<IVF[]> ivfs; //nlist个聚类,聚类内部的向量维度是block_dim
    std::unique_ptr<float[]> querys; //nq个查询向量, 维度是block_dim
    std::unique_ptr<SearchBlock[]> blocks; //按照搜索顺序排列，第1个先搜索
    std::unique_ptr<idx_t[]> listidqueries; //nq * nprobe 查询向量相近的聚类id

    void addIVFs(IVF* ivfs, size_t ivfCnt);
    void init(size_t id, size_t d, size_t block_dim, size_t nodeCount, IVF* ivfs, size_t nlist, const float* querys, size_t querySize, size_t blockCount, size_t nprobe) 
    {
        this->id = id;
        this->nprobe = nprobe;
        this->d = d;
        this->block_dim = block_dim;
        this->blockSize = querySize / blockCount;
        this->blockCount = blockCount;
        addIVFs(ivfs, nlist);
        addQuerys(querys, querySize);
        initBlock();
    }
    void initBlock() {
        blocks = std::make_unique<SearchBlock[]>(blockCount);
        for(int i = 0; i < blockCount; i++) {
            blocks[i].blockId = i; //先按顺序排布
        }
    }
    void addQuerys(const float* querys, size_t querySize) {
        this->querys = std::make_unique<float[]>(querySize * block_dim);
        copy_n_partial_vector(querys, this->querys.get(), d, block_dim, block_dim * (id - 1), querySize);
    }
    vector<vector<float>> search(size_t blockId) {
        size_t queryStart = blockId * blockSize;
        // cout << blockId << " " << blockSize << " " << queryStart << nprobe << endl;
        vector<vector<float>> distances = vector<vector<float>>(blockSize, vector<float>());

        for(size_t q = queryStart; q < queryStart + blockSize; q++) {
            // cout << "q" << q << endl;
            for(size_t i = 0; i < nprobe; i++) {
                idx_t ivfId = listidqueries[q * nprobe + i];
                // cout << "ivfId" << ivfId << endl;
                for(size_t v = 0; v < ivfs[ivfId].get_list_size(); v++) {
                    float dis = calculatedEuclideanDistance(querys.get() + q * block_dim, ivfs[ivfId].candidate_codes.get() + v * block_dim, block_dim);
                    // cout << "push " << q - queryStart << dis << endl;
                    distances[q - queryStart].push_back(dis);
                }
            }
        }
        // cout << BLUE; 
        // cout << distances.size() << " " << id << endl;
        // for(auto vec : distances) {
        //     cout << vec.size() << ": ";
        //     for(float f : vec) {
        //         cout << f << " ";
        //     }
        //     cout << endl;
        // }
        // cout << RESET;
        return distances;
    }
    
   
    ~Node() {}
};





}
#endif