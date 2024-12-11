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
    std::unique_ptr<size_t[]> queryCompareSize; //nq * nprobe 查询向量相近的聚类id
    //为了计算第q个向量的distancesForQueryies的偏移量,偏移量是queryCompareSizePreSum[q]
    std::unique_ptr<size_t[]> queryCompareSizePreSum;

    void addIVFs(IVF* ivfs, size_t ivfCnt);
    void init(size_t id, size_t d, size_t block_dim, size_t nodeCount, IVF* ivfs, size_t nlist, const float* querys, size_t querySize, size_t blockCount, size_t nprobe) 
    {
        this->id = id;
        if (nprobe > nlist) {
            nprobe = nlist;
        }
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

    struct SearchResult {
        std::unique_ptr<float[]> distances;
        size_t size;

        // Default constructor
        SearchResult() : distances(nullptr), size(0) {}

        // Parameterized constructor
        SearchResult(std::unique_ptr<float[]> dist, size_t sz)
            : distances(std::move(dist)), size(sz) {}

        // Move constructor
        SearchResult(SearchResult&& other) noexcept
            : distances(std::move(other.distances)), size(other.size) {
            other.size = 0;
        }

        // Move assignment operator
        SearchResult& operator=(SearchResult&& other) noexcept {
            if (this != &other) {
                distances = std::move(other.distances);
                size = other.size;
                other.size = 0;
            }
            return *this;
        }

        // Delete copy constructor and copy assignment operator
        SearchResult(const SearchResult&) = delete;
        SearchResult& operator=(const SearchResult&) = delete;
    };

    SearchResult search(size_t blockId) {
        size_t queryStart = blockId * blockSize;
        // cout << blockId << " " << blockSize << " " << queryStart << nprobe << endl;
        // size_t totalQueryCompareSize = 0;
        size_t totalQueryCompareSize = queryCompareSizePreSum[queryStart + blockSize] - queryCompareSizePreSum[queryStart];
        // cout << queryCompareSizePreSum[queryStart + blockSize] << endl;
        cout << blockId << " " << blockSize << " " << queryStart << " " << totalQueryCompareSize << endl;
        // for(size_t q = queryStart; q < queryStart + blockSize; q++) {
        //     totalQueryCompareSize += queryCompareSize[q];
        // }
        // // cout << "node " << id << "block: " << blockId << " compare size" << totalQueryCompareSize << endl;
        auto clock1 = std::chrono::high_resolution_clock::now();
        std::unique_ptr<float[]> distances = std::make_unique<float[]>(totalQueryCompareSize);
        auto clock2 = std::chrono::high_resolution_clock::now();
        std::cout << "node" << id << '|' << blockId << "| malloc" << std::chrono::duration<double>(clock2 - clock1).count() << "s" << std::endl;
        // // size_t curDistancePosition = 0;
        size_t nt = std::min(static_cast<size_t>(omp_get_max_threads()), blockSize);
#pragma omp parallel for num_threads(nt)
        for(size_t q = queryStart; q < queryStart + blockSize; q++) {
            // cout << "q" << q << endl;
            size_t queryOffset = queryCompareSizePreSum[q] - queryCompareSizePreSum[queryStart]; //第q个查询的结果应该存的地址偏移量
            size_t curDistancePosition = 0; //在一个查询向量的结果内
            for(size_t i = 0; i < nprobe; i++) {
                idx_t ivfId = listidqueries[q * nprobe + i];
                // cout << "ivfId" << ivfId << endl;
                for(size_t v = 0; v < ivfs[ivfId].get_list_size(); v++) {
                    float dis = calculatedEuclideanDistance(querys.get() + q * block_dim, ivfs[ivfId].candidate_codes.get() + v * block_dim, block_dim);
                    // cout << "push " << q - queryStart << dis << endl;
                    // distances[q - queryStart].push_back(dis);
                    assert(queryOffset + curDistancePosition < totalQueryCompareSize);
                    distances[queryOffset + curDistancePosition] = dis;
                    curDistancePosition++;
                }
            }
            // cout << curDistancePosition << " " << queryCompareSize[q] << endl;
            assert(curDistancePosition == queryCompareSize[q]);
        }
        // assert(curDistancePosition == totalQueryCompareSize);
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

        //malloc0.3s, search0.9s
        auto clock3 = std::chrono::high_resolution_clock::now();
        std::cout << "node" << id << '|' << blockId << "| search" << std::chrono::duration<double>(clock3 - clock2).count() << "s" << std::endl;
        return SearchResult(move(distances), totalQueryCompareSize);
    }
    
   
    ~Node() {}
};





}
#endif