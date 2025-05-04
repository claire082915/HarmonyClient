#ifndef INDEX_H
#define INDEX_H

#include <memory>
#include "Clustering.h"
#include "IVF.h"
#include "IVFScan.hpp"
#include "common.h"
#include "node.h"

namespace harmony {

class Index {
   public:
    Index(size_t d = 0,
          size_t nlist = 0,
          size_t nprobe = 0,
          MetricType metric = MetricType::METRIC_L2,
          OptLevel opt_level = OptLevel::OPT_NONE,
          size_t sub_k = 0,
          size_t sub_nlist = 1,
          size_t sub_nprobe = 1,
          bool verbose = false,
          EdgeDevice edge_device_enabled = EdgeDevice::EDGEDEVIVE_DISABLED
          );

    Index& operator=(Index&& other) noexcept;
    
    enum class SearchMode {
        BRUTE_FORCE,
        ORIGINAL,
        DIVIDE_IVF,
        DIVIDE_DIM,
        DIVIDE_GROUP,
    };

    static std::string to_string(SearchMode mode) {
        switch (mode) {
            case SearchMode::BRUTE_FORCE: return "Brute Force";
            case SearchMode::ORIGINAL: return "Original";
            case SearchMode::DIVIDE_IVF: return "Baseline";
            case SearchMode::DIVIDE_DIM: return "Dim     ";
            case SearchMode::DIVIDE_GROUP: return "Group   ";
            default: return "Unknown";
        }
    }

    struct Param {
        bool orderOptimize = true;
        SearchMode mode;
        bool divideIVFVersionOriginal = false;
        size_t startIVFId = 0, ivfCount = 0;
        size_t block_dim;
        idx_t* queryCompareSize;
        idx_t* queryCompareSizePreSum;
        idx_t queryStart;
        idx_t* listidqueries;
        bool pruning = false;
        bool period = false;
        bool fullWarmUp = false;
        float* heapTops = nullptr;

        size_t groupCount, teamCount, teamSize;
        bool hardInBalance = false;
        size_t hardInBalanceTeam;
        float hardInBalanceRatio;
        float hardInBalanceTeamRatio;
    };
    void train(size_t n, const float* codes, bool faiss = false, bool lite = false);

    void single_thread_nearest_cluster_search(size_t n, const float* queries, float* distances, idx_t* labels);
    void single_thread_search(size_t n, const float* queries, size_t k, float* distances, idx_t* labels, float ratio, Stats* stats);
    int single_thread_search(size_t n, const float* queries, size_t k, float* distances, idx_t* labels, float ratio,
                                 Stats* stats, size_t startIVF, size_t ivfCount);
    void single_thread_search_simple(size_t n, const float* queries, size_t k, float* distances, idx_t* labels, float ratio, Stats* stats);
    // void single_thread_search_worker(size_t n, const float* queries, float* distances, float ratio, Stats* stats, Param* param, float* originalQuery, float* heapTop, idx_t* listidqueries);
    void single_thread_search_block(size_t n, const float* queries, size_t k, float* distances, idx_t* label);
    void search_group_master(size_t n, const float* queries, size_t k, float* distances, idx_t* label);
    void search_divide_ivf(size_t n, const float* queries, size_t k, float* distances, idx_t* labels);
    void add(size_t n, const float* codes);
    void add_simple(size_t n, const float* codes);
    Stats search(size_t n, const float* queries, size_t k, float* distances, idx_t* labels, float ratio = 1);
    void save_index(std::string path) const;
    void save_index(std::string path, SearchMode mode) const;
    void load_index(std::string path);
    void load_SPANN(std::string path);
    // void initWorkers(size_t workerCount, float* querys, size_t querySize, size_t blockCount, size_t nb);
    void printIndex();
   
    void preSearch(size_t nb, size_t workerCount, size_t blockCount, size_t warmUpSearchList, size_t warmUpSearchListSize, Param* param, std::string path);
    void postSearch();
    // Declare other search-related methods
    std::unique_ptr<IVFScanBase> get_scanner(MetricType metric, OptLevel opt_level, size_t k, EdgeDevice edge_device_enabled = EdgeDevice::EDGEDEVIVE_DISABLED);

   private:
    void warmUpSearch(size_t n, const float* queries, size_t k, float* distances, idx_t* labels, idx_t* listidqueries);
    void findNearNprobeOfCentroidIds(size_t n, const float* queries);
   public:
    size_t d;
    size_t nlist;
    size_t nprobe;
    size_t workerCount;
    size_t blockCount;
    size_t blockSize, groupSize;
    MetricType metric;
    OptLevel opt_level;
    OptLevel added_opt_level;

    size_t sub_k;
    size_t sub_nlist;
    size_t sub_nprobe;

    bool verbose;
    EdgeDevice edge_device_enabled;
    MyStopWatch uniWatch;

    std::unique_ptr<IVF[]> lists;             // Represents all clusters, identified by index, each IVF represents a cluster
    std::unique_ptr<float[]> centroid_codes;  // Cluster centroids represented as vectors
    std::unique_ptr<idx_t[]> centroid_ids;    // Cluster centroid IDs, usually sequential numbers like 1, 2, 3,...
    // std::unique_ptr<tribase::Node[]> nodes;
    std::unique_ptr<float[]> distancesForNQuerys;
    idx_t presumeTotalQueryCompareSize = 0;
    // std::vector<std::unique_ptr<float[]>> blockDistancesBuffer;
    std::vector<std::vector<idx_t>> workerSearchBlockOrder;  // Each worker has a corresponding order of block IDs for computation
    std::vector<std::vector<idx_t>> blockSearchedOrder;
    size_t warmUpSearchList = 0;      // Number of clusters to search in warm-up phase
    size_t warmUpSearchListSize = 0;  // Number of vectors to take from each cluster during warm-up

    // idx_t presumeNq = 370, 
    idx_t presumeK = 100;
    std::vector<std::unique_ptr<float[]>> distancesHeapBuffer;
    std::vector<std::unique_ptr<idx_t[]>> labelsHeapBuffer;

    // idx_t threashHold = INT_MAX;
    bool blockMalloc = false;  // Whether distanceForNQuerys represents just one block or all blocks

    std::unique_ptr<float[]> originalQuery;
    // std::unique_ptr<float[]> heapTops;

    // std::unique_ptr<float[]> distanceHeapForBlock;
    // std::unique_ptr<idx_t[]> idHeapForBlock;
    Param* param;

    SearchOrder groupSearchOrder, blockSearchOrder;
    vector<size_t> beginIVFs;
    vector<size_t> ivfCounts;
    std::unique_ptr<idx_t[]> listidqueries;
    double trainTime = 0, addTime = 0, preSearchTime = 0;
};

}  // namespace harmony

#endif  // INDEX_H
