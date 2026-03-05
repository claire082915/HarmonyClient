#pragma once

#include <grpcpp/grpcpp.h>
#include <memory>
#include <string>
#include <mutex>
#include <condition_variable>
#include <queue>

// Forward-declare generated stubs (included after proto compilation)
#include "harmony.grpc.pb.h"

namespace harmony {

// ---------------------------------------------------------------------------
// Thread-safe work queue shared between the gRPC service and the MPI master
// ---------------------------------------------------------------------------
struct SearchJob {
    // Input
    std::vector<float> vectors;  // nq * d
    uint32_t nq;
    uint32_t d;
    uint32_t k;

    // Output (filled by MPI master, then signalled)
    std::vector<float>   distances;  // nq * k
    std::vector<int64_t> labels;     // nq * k
    bool done = false;

    // Synchronisation for this individual job
    std::mutex              mtx;
    std::condition_variable cv;
};

// ---------------------------------------------------------------------------
// gRPC service implementation – lives on the gRPC thread pool
// ---------------------------------------------------------------------------
class HarmonyServiceImpl final : public HarmonyService::Service {
public:
    explicit HarmonyServiceImpl(std::shared_ptr<std::queue<std::shared_ptr<SearchJob>>> job_queue,
                                std::shared_ptr<std::mutex>              queue_mutex,
                                std::shared_ptr<std::condition_variable> queue_cv)
        : job_queue_(std::move(job_queue)),
          queue_mutex_(std::move(queue_mutex)),
          queue_cv_(std::move(queue_cv)) {}

    grpc::Status Search(grpc::ServerContext* ctx,
                        const SearchRequest* request,
                        SearchResponse*      response) override;

private:
    std::shared_ptr<std::queue<std::shared_ptr<SearchJob>>> job_queue_;
    std::shared_ptr<std::mutex>              queue_mutex_;
    std::shared_ptr<std::condition_variable> queue_cv_;
};

// ---------------------------------------------------------------------------
// Wrapper that starts/stops the gRPC server in a background thread
// ---------------------------------------------------------------------------
class MasterGrpcServer {
public:
    MasterGrpcServer(const std::string& address,
                     std::shared_ptr<std::queue<std::shared_ptr<SearchJob>>> job_queue,
                     std::shared_ptr<std::mutex>              queue_mutex,
                     std::shared_ptr<std::condition_variable> queue_cv);

    ~MasterGrpcServer();

    void Start();
    void Shutdown();

private:
    std::string address_;
    std::unique_ptr<grpc::Server> server_;
    std::shared_ptr<HarmonyServiceImpl> service_;
    std::shared_ptr<std::queue<std::shared_ptr<SearchJob>>> job_queue_;
    std::shared_ptr<std::mutex>              queue_mutex_;
    std::shared_ptr<std::condition_variable> queue_cv_;
};

} // namespace harmony