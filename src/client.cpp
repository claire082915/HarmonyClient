// client.cpp
//
// Sends query vectors to the Harmony master over a plain TCP socket.
// No MPI dynamic process management (MPI_Comm_connect) needed.
//
// Protocol (all little-endian):
//   Client -> Master:  [uint64 nq][uint64 k][nq*d floats]
//   Master -> Client:  [nq*k floats distances][nq*k int64 labels]
//   Shutdown:          [uint64 nq=0][uint64 k=0]
//
// Run:
//   # Terminal 1: start server
//   mpirun -n 5 ./release/bin/query \
//       --benchmarks_path ./benchmarks --dataset sift1m \
//       --cache --group=2 --team=2 --block=4 --nprobe 100 --serve --tcp_port 7777
//
//   # Terminal 2: run client (no mpirun needed)
//   ./release/bin/harmony_client \
//       --query ./benchmarks/sift1m/origin/sift1m_query.fvecs \
//       --host 127.0.0.1 --port 7777 --k 10 --nq 100

#include <argparse/argparse.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void send_all(int fd, const void* buf, size_t n) {
    const char* p = static_cast<const char*>(buf);
    while (n > 0) {
        ssize_t sent = ::send(fd, p, n, 0);
        if (sent <= 0) throw std::runtime_error("send failed");
        p += sent; n -= sent;
    }
}
static void recv_all(int fd, void* buf, size_t n) {
    char* p = static_cast<char*>(buf);
    while (n > 0) {
        ssize_t got = ::recv(fd, p, n, 0);
        if (got <= 0) throw std::runtime_error("recv failed");
        p += got; n -= got;
    }
}

static std::tuple<std::vector<float>, size_t, size_t>
load_fvecs(const std::string& path, size_t max_vecs = 0) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);
    std::vector<float> data;
    size_t nq = 0, d = 0;
    while (f) {
        int32_t dim = 0;
        if (!f.read(reinterpret_cast<char*>(&dim), sizeof(int32_t))) break;
        if (d == 0) d = static_cast<size_t>(dim);
        if (static_cast<size_t>(dim) != d) throw std::runtime_error("Inconsistent dim");
        std::vector<float> vec(d);
        f.read(reinterpret_cast<char*>(vec.data()), d * sizeof(float));
        if (!f) break;
        data.insert(data.end(), vec.begin(), vec.end());
        ++nq;
        if (max_vecs > 0 && nq >= max_vecs) break;
    }
    return {std::move(data), nq, d};
}

int main(int argc, char** argv) {
    argparse::ArgumentParser prog("harmony_client");
    prog.add_argument("--query").help("Path to query .fvecs file").required();
    prog.add_argument("--host").help("Master IP address").default_value(std::string("127.0.0.1"));
    prog.add_argument("--port").help("Master TCP port").default_value(7777ul)
        .action([](const std::string& s) -> size_t { return std::stoul(s); });
    prog.add_argument("--k").help("Number of nearest neighbours").default_value(10ul)
        .action([](const std::string& s) -> size_t { return std::stoul(s); });
    prog.add_argument("--nq").help("Max queries (0=all)").default_value(0ul)
        .action([](const std::string& s) -> size_t { return std::stoul(s); });
    prog.add_argument("--loop").help("Repeat N times").default_value(1ul)
        .action([](const std::string& s) -> size_t { return std::stoul(s); });

    try { prog.parse_args(argc, argv); }
    catch (const std::runtime_error& err) { std::cerr << err.what() << "\n" << prog; return 1; }

    size_t k      = prog.get<size_t>("--k");
    size_t max_nq = prog.get<size_t>("--nq");
    size_t loop   = prog.get<size_t>("--loop");
    std::string host = prog.get<std::string>("--host");
    uint16_t port    = static_cast<uint16_t>(prog.get<size_t>("--port"));

    std::cout << "[Client] Loading queries...\n";
    auto [vectors, nq, d] = load_fvecs(prog.get<std::string>("--query"), max_nq);
    std::cout << std::format("[Client] nq={} d={} k={}\n", nq, d, k);

    int sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) throw std::runtime_error("socket() failed");
    int flag = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0)
        throw std::runtime_error("Invalid host: " + host);

    std::cout << std::format("[Client] Connecting to {}:{}...\n", host, port);
    if (::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0)
        throw std::runtime_error("connect() failed - is the server running with --serve?");
    std::cout << "[Client] Connected.\n";

    for (size_t iter = 0; iter < loop; ++iter) {
        auto t0 = std::chrono::high_resolution_clock::now();

        uint64_t hdr[2] = {nq, k};
        send_all(sock, hdr, sizeof(hdr));
        send_all(sock, vectors.data(), nq * d * sizeof(float));
        std::cout << std::format("[Client] iter={} sent {} queries\n", iter, nq);

        std::vector<float>   distances(nq * k);
        std::vector<int64_t> labels(nq * k);
        recv_all(sock, distances.data(), nq * k * sizeof(float));
        recv_all(sock, labels.data(),    nq * k * sizeof(int64_t));

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        std::cout << std::format("[Client] iter={} done in {:.4f}s\n", iter, elapsed);

        for (size_t q = 0; q < std::min(nq, (size_t)5); ++q) {
            std::cout << std::format("  Query {:3d}: ", q);
            for (size_t i = 0; i < std::min(k, (size_t)5); ++i)
                std::cout << std::format("[id={} d={:.3f}] ", labels[q*k+i], distances[q*k+i]);
            std::cout << "\n";
        }
    }

    uint64_t shutdown[2] = {0, 0};
    send_all(sock, shutdown, sizeof(shutdown));
    ::close(sock);
    std::cout << "[Client] Done.\n";
    return 0;
}