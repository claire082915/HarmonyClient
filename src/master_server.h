#pragma once

// master_server.h  — TCP server for the query phase
//
// Protocol (little-endian, same as client.cpp):
//   Client -> Master:  [uint64 nq][uint64 k][nq*d floats]
//   Master -> Client:  [nq*k floats distances][nq*k int64 labels]
//   Shutdown:          [uint64 nq=0][uint64 k=0]

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace harmony {

// ---------------------------------------------------------------------------
// SearchJob — shared between the TCP listener thread and the MPI master loop
// ---------------------------------------------------------------------------
struct SearchJob {
    // Input
    std::vector<float> vectors;   // nq * d floats
    uint32_t nq = 0;
    uint32_t d  = 0;
    uint32_t k  = 0;

    // Output (filled by MPI master, then signalled back to listener thread)
    std::vector<float>   distances;   // nq * k
    std::vector<int64_t> labels;      // nq * k
    bool done = false;

    // Per-job synchronisation
    std::mutex              mtx;
    std::condition_variable cv;
};

// ---------------------------------------------------------------------------
// MasterTcpServer
//
// Runs a background thread that:
//   1. Accepts one TCP client connection at a time.
//   2. Reads [nq, k] header + query vectors.
//   3. Enqueues a SearchJob for the MPI master loop (main thread).
//   4. Waits for the job to be marked done.
//   5. Sends distances + labels back to the client.
//   6. Loops until nq==0 (shutdown) or client disconnects.
//
// IsDone() returns true once the background thread has exited its serve loop,
// allowing the MPI master loop to unblock and shut down cleanly.
// ---------------------------------------------------------------------------
class MasterTcpServer {
public:
    // address is ignored (kept for API compatibility); port is used directly.
    MasterTcpServer(const std::string& address,
                    uint16_t port,
                    size_t   vector_dim,
                    std::shared_ptr<std::queue<std::shared_ptr<SearchJob>>> job_queue,
                    std::shared_ptr<std::mutex>              queue_mutex,
                    std::shared_ptr<std::condition_variable> queue_cv);

    ~MasterTcpServer();

    // Start listening in a background thread. Returns immediately.
    void Start();

    // Signal the background thread to stop and join it.
    void Shutdown();

    // Returns true once ServeLoop has finished (client sent nq=0, disconnected,
    // or Shutdown() was called).  The MPI master loop uses this to know when
    // no more jobs will arrive.
    bool IsDone() const { return done_; }

private:
    void ServeLoop();   // runs on background thread

    static void send_all(int fd, const void* buf, size_t n);
    static void recv_all(int fd, void* buf, size_t n);

    std::string address_;
    uint16_t    port_;
    size_t      d_;          // vector dimension

    std::shared_ptr<std::queue<std::shared_ptr<SearchJob>>> job_queue_;
    std::shared_ptr<std::mutex>              queue_mutex_;
    std::shared_ptr<std::condition_variable> queue_cv_;

    int         server_fd_ = -1;
    std::thread thread_;
    std::atomic<bool> stop_{ false };
    std::atomic<bool> done_{ false };  // set to true when ServeLoop exits
};

} // namespace harmony