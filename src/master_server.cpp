// master_server.cpp  — TCP replacement for the gRPC-based server

#include "master_server.h"

#include <iostream>
#include <format>
#include <stdexcept>

namespace harmony {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
void MasterTcpServer::send_all(int fd, const void* buf, size_t n) {
    const char* p = static_cast<const char*>(buf);
    while (n > 0) {
        ssize_t s = ::send(fd, p, n, 0);
        if (s <= 0) throw std::runtime_error("TCP send failed");
        p += s; n -= s;
    }
}

void MasterTcpServer::recv_all(int fd, void* buf, size_t n) {
    char* p = static_cast<char*>(buf);
    while (n > 0) {
        ssize_t s = ::recv(fd, p, n, 0);
        if (s <= 0) throw std::runtime_error("TCP recv failed");
        p += s; n -= s;
    }
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
MasterTcpServer::MasterTcpServer(
    const std::string& address,
    uint16_t port,
    size_t   vector_dim,
    std::shared_ptr<std::queue<std::shared_ptr<SearchJob>>> job_queue,
    std::shared_ptr<std::mutex>              queue_mutex,
    std::shared_ptr<std::condition_variable> queue_cv)
    : address_(address),
      port_(port),
      d_(vector_dim),
      job_queue_(std::move(job_queue)),
      queue_mutex_(std::move(queue_mutex)),
      queue_cv_(std::move(queue_cv)) {}

MasterTcpServer::~MasterTcpServer() {
    Shutdown();
}

// ---------------------------------------------------------------------------
// Start — open the listening socket and launch the background thread
// ---------------------------------------------------------------------------
void MasterTcpServer::Start() {
    server_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0)
        throw std::runtime_error("[MasterTcpServer] socket() failed");

    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port_);

    if (::bind(server_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0)
        throw std::runtime_error(
            std::format("[MasterTcpServer] bind() failed on port {}", port_));

    ::listen(server_fd_, 1);

    std::cout << std::format("[Master] TCP server listening on port {}\n", port_);
    std::cout << "[Master] Waiting for client...\n";

    thread_ = std::thread(&MasterTcpServer::ServeLoop, this);
}

// ---------------------------------------------------------------------------
// Shutdown — close the server socket and join the background thread
// ---------------------------------------------------------------------------
void MasterTcpServer::Shutdown() {
    stop_ = true;
    if (server_fd_ >= 0) {
        ::close(server_fd_);
        server_fd_ = -1;
    }
    if (thread_.joinable())
        thread_.join();
}

// ---------------------------------------------------------------------------
// ServeLoop — background thread body
//
// Accepts one client at a time and processes jobs until:
//   - nq == 0  (graceful shutdown from client), or
//   - TCP error (client disconnected), or
//   - stop_ flag is set.
// ---------------------------------------------------------------------------
void MasterTcpServer::ServeLoop() {
    while (!stop_) {
        sockaddr_in client_addr{};
        socklen_t   client_len = sizeof(client_addr);

        int client_fd = ::accept(server_fd_,
                                 reinterpret_cast<sockaddr*>(&client_addr),
                                 &client_len);
        if (client_fd < 0) {
            if (!stop_)
                std::cerr << "[MasterTcpServer] accept() failed\n";
            break;
        }

        // Disable Nagle for lower latency
        int flag = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip));
        std::cout << std::format("[Master] Client connected from {}\n", client_ip);

        // ---- per-client job loop ----------------------------------------
        while (!stop_) {
            // 1. Read header: [nq (uint64), k (uint64)]
            uint64_t hdr[2] = {};
            try { recv_all(client_fd, hdr, sizeof(hdr)); }
            catch (...) {
                std::cout << "[Master] Client disconnected.\n";
                break;
            }

            size_t nq    = static_cast<size_t>(hdr[0]);
            size_t job_k = static_cast<size_t>(hdr[1]);

            // nq == 0 is the graceful shutdown signal from the client
            if (nq == 0) {
                std::cout << "[Master] Graceful shutdown signal received from client.\n";
                stop_ = true;
                break;
            }

            std::cout << std::format("[Master] Job received: nq={} k={}\n", nq, job_k);

            // 2. Read query vectors
            auto job = std::make_shared<SearchJob>();
            job->nq = static_cast<uint32_t>(nq);
            job->d  = static_cast<uint32_t>(d_);
            job->k  = static_cast<uint32_t>(job_k);
            job->vectors.resize(nq * d_);

            try { recv_all(client_fd, job->vectors.data(), nq * d_ * sizeof(float)); }
            catch (...) {
                std::cout << "[Master] Client disconnected while sending vectors.\n";
                break;
            }

            // 3. Enqueue for MPI master loop
            {
                std::lock_guard<std::mutex> lk(*queue_mutex_);
                job_queue_->push(job);
            }
            queue_cv_->notify_one();

            // 4. Wait for MPI master to complete the search
            {
                std::unique_lock<std::mutex> lk(job->mtx);
                job->cv.wait(lk, [&job] { return job->done; });
            }

            // 5. Send results: distances then labels
            try {
                send_all(client_fd, job->distances.data(),
                         nq * job_k * sizeof(float));
                send_all(client_fd, job->labels.data(),
                         nq * job_k * sizeof(int64_t));
            } catch (...) {
                std::cout << "[Master] Client disconnected while sending results.\n";
                break;
            }

            std::cout << std::format("[Master] Results sent for nq={}\n", nq);
        }
        // ---- end per-client job loop ------------------------------------

        ::close(client_fd);
    }

    std::cout << "[Master] TCP server thread exiting.\n";
}

} // namespace harmony