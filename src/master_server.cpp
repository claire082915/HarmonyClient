#include "master_server.h"

#include <iostream>
#include <thread>

namespace harmony {

// ---------------------------------------------------------------------------
// HarmonyServiceImpl::Search
//   Called on a gRPC thread for every incoming RPC.
//   Packages the request into a SearchJob, enqueues it for the MPI master
//   loop (running on the main thread), then blocks until the result is ready.
// ---------------------------------------------------------------------------
grpc::Status HarmonyServiceImpl::Search(grpc::ServerContext* /*ctx*/,
                                         const SearchRequest* request,
                                         SearchResponse*      response) {
    // Basic validation
    uint32_t nq = request->nq();
    uint32_t d  = request->d();
    uint32_t k  = request->k();

    if (nq == 0 || d == 0 || k == 0) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                            "nq, d and k must all be > 0");
    }
    if ((uint32_t)request->vectors_size() != nq * d) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                            "vectors size does not match nq * d");
    }

    // Build the job
    auto job = std::make_shared<SearchJob>();
    job->vectors.assign(request->vectors().begin(), request->vectors().end());
    job->nq = nq;
    job->d  = d;
    job->k  = k;
    job->done = false;

    // Enqueue for the MPI master
    {
        std::lock_guard<std::mutex> lk(*queue_mutex_);
        job_queue_->push(job);
    }
    queue_cv_->notify_one();

    // Wait for the MPI master to fill in results
    {
        std::unique_lock<std::mutex> lk(job->mtx);
        job->cv.wait(lk, [&job] { return job->done; });
    }

    // Pack results into the response
    for (uint32_t q = 0; q < nq; ++q) {
        QueryResult* qr = response->add_results();
        for (uint32_t i = 0; i < k; ++i) {
            Neighbour* nb = qr->add_neighbours();
            nb->set_id(job->labels[q * k + i]);
            nb->set_distance(job->distances[q * k + i]);
        }
    }

    return grpc::Status::OK;
}

// ---------------------------------------------------------------------------
// MasterGrpcServer
// ---------------------------------------------------------------------------
MasterGrpcServer::MasterGrpcServer(
    const std::string& address,
    std::shared_ptr<std::queue<std::shared_ptr<SearchJob>>> job_queue,
    std::shared_ptr<std::mutex>              queue_mutex,
    std::shared_ptr<std::condition_variable> queue_cv)
    : address_(address),
      job_queue_(std::move(job_queue)),
      queue_mutex_(std::move(queue_mutex)),
      queue_cv_(std::move(queue_cv)) {}

MasterGrpcServer::~MasterGrpcServer() {
    Shutdown();
}

void MasterGrpcServer::Start() {
    service_ = std::make_shared<HarmonyServiceImpl>(job_queue_, queue_mutex_, queue_cv_);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(address_, grpc::InsecureServerCredentials());
    builder.RegisterService(service_.get());

    server_ = builder.BuildAndStart();
    if (!server_) {
        throw std::runtime_error("Failed to start gRPC server on " + address_);
    }
    std::cout << "[Master] gRPC server listening on " << address_ << std::endl;
}

void MasterGrpcServer::Shutdown() {
    if (server_) {
        server_->Shutdown();
        server_.reset();
    }
}

} // namespace harmony