// client.cpp
//
// Harmony TCP client — supports batched INSERT and batched QUERY, with
// optional recall@k computation against a groundtruth file.
//
// Protocol (all little-endian):
//
//   OP_INSERT     (0x01):
//     Client -> Master: [uint8 op=1][uint64 n][n*d floats]
//     Master -> Client: [uint8 status]   (0=OK, 1=error)
//
//   OP_BUILD_DONE (0x02):
//     Client -> Master: [uint8 op=2]
//     Master -> Client: [uint8 status]
//
//   OP_QUERY      (0x03):
//     Client -> Master: [uint8 op=3][uint64 nq][uint64 k][nq*d floats]
//     Master -> Client: [nq*k floats distances][nq*k int64 labels]
//
//   OP_SHUTDOWN   (0x00):
//     Client -> Master: [uint8 op=0]
//
// Recall computation:
//   Provide --groundtruth to a .ivecs or .bin groundtruth file.
//   recall@k is printed per query batch and averaged across all batches.
//   If --groundtruth is not provided the query phase runs without recall.
//
// Usage examples:
//
//   # Insert only
//   ./release/bin/harmony_client \
//       --base  ./benchmarks/sift1m/origin/sift1m_base.fvecs \
//       --host 127.0.0.1 --port 7777 \
//       --insert_batch 10000
//
//   # Insert then query with recall
//   ./release/bin/harmony_client \
//       --base  ./benchmarks/sift1m/origin/sift1m_base.fvecs \
//       --query ./benchmarks/sift1m/origin/sift1m_query.fvecs \
//       --groundtruth ./benchmarks/sift1m/result/groundtruth_10.bin \
//       --host 127.0.0.1 --port 7777 \
//       --insert_batch 10000 --query_batch 2000 --k 10
//
//   # Query only with recall (assumes server already has index built)
//   ./release/bin/harmony_client \
//       --query ./benchmarks/sift1m/origin/sift1m_query.fvecs \
//       --groundtruth ./benchmarks/sift1m/result/groundtruth_10.bin \
//       --host 127.0.0.1 --port 7777 \
//       --k 10 --nq 100 --skip_build

#include <argparse/argparse.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <thread>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include <cerrno>
#include <csignal>

// ---------------------------------------------------------------------------
// Opcodes — must match query.cpp
// ---------------------------------------------------------------------------
static constexpr uint8_t OP_SHUTDOWN   = 0x00;
static constexpr uint8_t OP_INSERT     = 0x01;
static constexpr uint8_t OP_BUILD_DONE = 0x02;
static constexpr uint8_t OP_QUERY      = 0x03;

static constexpr uint8_t STATUS_OK    = 0x00;
static constexpr uint8_t STATUS_ERROR = 0x01;

// ---------------------------------------------------------------------------
// TCP helpers
// ---------------------------------------------------------------------------
static void send_all(int fd, const void* buf, size_t n) {
    const char* p = static_cast<const char*>(buf);
    while (n > 0) {
        ssize_t s = ::send(fd, p, n, MSG_NOSIGNAL);
        if (s < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("send failed: ") + strerror(errno));
        }
        if (s == 0) throw std::runtime_error("send failed: connection closed");
        p += s; n -= s;
    }
}
static void recv_all(int fd, void* buf, size_t n) {
    char* p = static_cast<char*>(buf);
    while (n > 0) {
        ssize_t s = ::recv(fd, p, n, 0);
        if (s < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("recv failed: ") + strerror(errno));
        }
        if (s == 0) throw std::runtime_error("recv failed: server disconnected");
        p += s; n -= s;
    }
}

// ---------------------------------------------------------------------------
// Load fvecs — returns (data, nVecs, dim)
// ---------------------------------------------------------------------------
static std::tuple<std::vector<float>, size_t, size_t>
load_fvecs(const std::string& path, size_t max_vecs = 0) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);

    std::vector<float> data;
    size_t nv = 0, d = 0;
    while (f) {
        int32_t dim = 0;
        if (!f.read(reinterpret_cast<char*>(&dim), sizeof(int32_t))) break;
        if (d == 0) d = static_cast<size_t>(dim);
        if (static_cast<size_t>(dim) != d) throw std::runtime_error("Inconsistent dim in " + path);

        data.resize(data.size() + d);
        f.read(reinterpret_cast<char*>(data.data() + nv * d), d * sizeof(float));
        if (!f) break;
        ++nv;
        if (max_vecs > 0 && nv >= max_vecs) break;
    }
    return {std::move(data), nv, d};
}

// Load bvecs — converts uint8 to float, returns (data, nVecs, dim)
// ---------------------------------------------------------------------------
static std::tuple<std::vector<float>, size_t, size_t>
load_bvecs(const std::string& path, size_t max_vecs = 0) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);
    std::vector<float> data;
    size_t nv = 0, d = 0;
    while (f) {
        int32_t dim = 0;
        if (!f.read(reinterpret_cast<char*>(&dim), sizeof(int32_t))) break;
        if (d == 0) d = static_cast<size_t>(dim);
        if (static_cast<size_t>(dim) != d) throw std::runtime_error("Inconsistent dim in " + path);
        std::vector<uint8_t> buf(d);
        f.read(reinterpret_cast<char*>(buf.data()), d);
        if (!f) break;
        data.resize(data.size() + d);
        for (size_t i = 0; i < d; ++i)
            data[nv * d + i] = static_cast<float>(buf[i]);
        ++nv;
        if (max_vecs > 0 && nv >= max_vecs) break;
    }
    return {std::move(data), nv, d};
}

// ---------------------------------------------------------------------------
// Load ivecs — returns (data, nVecs, dim)
// ---------------------------------------------------------------------------
static std::tuple<std::vector<int32_t>, size_t, size_t>
load_ivecs(const std::string& path, size_t max_vecs = 0) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);

    std::vector<int32_t> data;
    size_t nv = 0, d = 0;
    while (f) {
        int32_t dim = 0;
        if (!f.read(reinterpret_cast<char*>(&dim), sizeof(int32_t))) break;
        if (d == 0) d = static_cast<size_t>(dim);
        if (static_cast<size_t>(dim) != d) throw std::runtime_error("Inconsistent dim in " + path);

        data.resize(data.size() + d);
        f.read(reinterpret_cast<char*>(data.data() + nv * d), d * sizeof(int32_t));
        if (!f) break;
        ++nv;
        if (max_vecs > 0 && nv >= max_vecs) break;
    }
    return {std::move(data), nv, d};
}

// ---------------------------------------------------------------------------
// Load groundtruth from either .ivecs or .bin format.
//
// .ivecs format: standard vecs format with int32 entries.
// .bin format (Harmony internal): sequence of [idx_t (int64)] entries stored
//   as raw binary, preceded by no per-vector header.  The file is assumed to
//   contain nq * k int64 labels followed by nq * k float distances (same
//   layout as writeResultsToFile / loadResults in utils.h).
//
// Returns (labels_int64, nVecs, k) — labels are int64 for direct comparison
// with the server response.
// ---------------------------------------------------------------------------
static std::tuple<std::vector<int64_t>, size_t, size_t>
load_groundtruth(const std::string& path, size_t nq, size_t k) {
    // Determine format from file extension
    auto ext_pos = path.rfind('.');
    std::string ext = (ext_pos != std::string::npos) ? path.substr(ext_pos) : "";

    std::vector<int64_t> labels;

    if (ext == ".ivecs") {
        auto [data, nv, dim] = load_ivecs(path, nq);
        if (dim < k)
            throw std::runtime_error(std::format(
                "Groundtruth dim {} < k {}", dim, k));
        labels.resize(nv * k);
        for (size_t i = 0; i < nv; ++i)
            for (size_t j = 0; j < k; ++j)
                labels[i * k + j] = static_cast<int64_t>(data[i * dim + j]);
        return {std::move(labels), nv, k};
    }

    // .bin format: int64 labels then float distances, nq*k each
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open groundtruth: " + path);

    labels.resize(nq * k);
    f.read(reinterpret_cast<char*>(labels.data()), nq * k * sizeof(int64_t));
    if (!f)
        throw std::runtime_error("Failed to read groundtruth labels from: " + path);

    return {std::move(labels), nq, k};
}

// ---------------------------------------------------------------------------
// Compute recall@k for a single batch.
//
// For each query, checks how many of the retrieved `k` labels appear in the
// ground-truth top-k set.  Returns the mean recall across all queries.
// ---------------------------------------------------------------------------
static float compute_recall(
        const std::vector<int64_t>& gt_labels,   // [batch_nq * gt_k]
        size_t                       gt_k,
        const std::vector<int64_t>& pred_labels,  // [batch_nq * k]
        size_t                       k,
        size_t                       batch_nq,
        size_t                       gt_offset)   // first query index in the global gt
{
    double total = 0.0;
    for (size_t q = 0; q < batch_nq; ++q) {
        // Build ground-truth set for this query
        std::unordered_set<int64_t> gt_set;
        gt_set.reserve(gt_k);
        for (size_t i = 0; i < gt_k; ++i)
            gt_set.insert(gt_labels[(gt_offset + q) * gt_k + i]);

        // Count hits in predictions
        size_t hits = 0;
        for (size_t i = 0; i < k; ++i) {
            if (gt_set.count(pred_labels[q * k + i]))
                ++hits;
        }
        total += static_cast<double>(hits) / static_cast<double>(std::min(k, gt_k));
    }
    return static_cast<float>(total / static_cast<double>(batch_nq));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    signal(SIGPIPE, SIG_IGN);
    argparse::ArgumentParser prog("harmony_client");
    prog.add_argument("--base")
        .help("Path to base .fvecs for INSERT phase")
        .default_value(std::string(""));
    prog.add_argument("--query")
        .help("Path to query .fvecs for QUERY phase")
        .default_value(std::string(""));
    prog.add_argument("--groundtruth")
        .help("Path to groundtruth file (.ivecs or .bin) for recall computation")
        .default_value(std::string(""));
    prog.add_argument("--host")
        .help("Master IP address")
        .default_value(std::string("127.0.0.1"));
    prog.add_argument("--port")
        .help("Master TCP port")
        .default_value(7777ul)
        .action([](const std::string& s) -> size_t { return std::stoul(s); });
    prog.add_argument("--k")
        .help("Number of nearest neighbours for QUERY")
        .default_value(10ul)
        .action([](const std::string& s) -> size_t { return std::stoul(s); });
    prog.add_argument("--nq")
        .help("Max query vectors to use (0 = all)")
        .default_value(0ul)
        .action([](const std::string& s) -> size_t { return std::stoul(s); });
    prog.add_argument("--nb")
        .help("Max base vectors to insert (0 = all)")
        .default_value(0ul)
        .action([](const std::string& s) -> size_t { return std::stoul(s); });
    prog.add_argument("--insert_batch")
        .help("Number of vectors per INSERT batch")
        .default_value(10000ul)
        .action([](const std::string& s) -> size_t { return std::stoul(s); });
    prog.add_argument("--query_batch")
        .help("Number of queries per QUERY batch")
        .default_value(2000ul)
        .action([](const std::string& s) -> size_t { return std::stoul(s); });
    prog.add_argument("--skip_build")
        .help("Skip INSERT + BUILD_DONE phase (server already has index)")
        .default_value(false).implicit_value(true);
    prog.add_argument("--query_loop")
        .help("Repeat the full query phase N times")
        .default_value(1ul)
        .action([](const std::string& s) -> size_t { return std::stoul(s); });
    prog.add_argument("--group")
        .help("groupCount used by server (query_batch must be divisible by group*block)")
        .default_value(1ul)
        .action([](const std::string& s) -> size_t { return std::stoul(s); });
    prog.add_argument("--block")
        .help("blockCount used by server (query_batch must be divisible by group*block)")
        .default_value(1ul)
        .action([](const std::string& s) -> size_t { return std::stoul(s); });

    try { prog.parse_args(argc, argv); }
    catch (const std::runtime_error& err) { std::cerr << err.what() << "\n" << prog; return 1; }

    std::string base_path      = prog.get<std::string>("--base");
    std::string query_path     = prog.get<std::string>("--query");
    std::string gt_path        = prog.get<std::string>("--groundtruth");
    size_t k                   = prog.get<size_t>("--k");
    size_t max_nq              = prog.get<size_t>("--nq");
    size_t max_nb              = prog.get<size_t>("--nb");
    size_t insert_batch        = prog.get<size_t>("--insert_batch");
    size_t query_batch         = prog.get<size_t>("--query_batch");
    size_t query_loop          = prog.get<size_t>("--query_loop");
    size_t group_count         = prog.get<size_t>("--group");
    size_t block_count         = prog.get<size_t>("--block");
    bool   skip_build          = prog.get<bool>("--skip_build");

    // query_batch must be divisible by (group * block) for DIVIDE_GROUP mode.
    // Round down silently so users don't have to compute this themselves.
    size_t batch_align = group_count * block_count;
    if (batch_align > 1 && query_batch % batch_align != 0) {
        size_t aligned = (query_batch / batch_align) * batch_align;
        if (aligned == 0) aligned = batch_align;
        std::cerr << std::format("[Client] WARNING: query_batch {} not divisible by group*block={}. "
                                 "Rounding down to {}.", query_batch, batch_align, aligned);
        query_batch = aligned;
    }
    std::string host           = prog.get<std::string>("--host");
    uint16_t port              = static_cast<uint16_t>(prog.get<size_t>("--port"));

    bool do_insert = !skip_build && !base_path.empty();
    bool do_query  = !query_path.empty();

    if (!do_insert && !do_query) {
        std::cerr << "[Client] Nothing to do: provide --base for insert and/or --query for queries.\n";
        return 1;
    }

    // ------------------------------------------------------------------
    // Connect
    // ------------------------------------------------------------------
    int sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) throw std::runtime_error("socket() failed");
    {
        int flag = 1;
        setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0)
        throw std::runtime_error("Invalid host: " + host);

    std::cout << std::format("[Client] Connecting to {}:{}...\n", host, port);
    int max_retries = 100; // retry for up to 30 minutes
    int retry_delay = 60;   // seconds between retries
    bool connected = false;
    for (int attempt = 0; attempt < max_retries; attempt++) {
        if (::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
            connected = true;
            break;
        }
        if (attempt == 0)
            std::cout << "[Client] Server not ready yet, waiting...\n";
        std::cout << std::format("[Client] Retrying connection in {}s (attempt {}/{})...\n",
                                    retry_delay, attempt + 1, max_retries);
        std::this_thread::sleep_for(std::chrono::seconds(retry_delay));
        ::close(sock);
        sock = ::socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) throw std::runtime_error("socket() failed on retry");
        // Re-set up the address
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port   = htons(static_cast<uint16_t>(port));
        if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0)
            throw std::runtime_error("Invalid host: " + host);
    }
    if (!connected)
        throw std::runtime_error("connect() failed after 30 minutes — is the server running with --serve?");
    std::cout << "[Client] Connected.\n";

    // ------------------------------------------------------------------
    // INSERT phase
    // ------------------------------------------------------------------
    if (do_insert) {
        std::cout << std::format("[Client] Loading base vectors from {}...\n", base_path);
        auto [base_data, total_nb, d] = base_path.ends_with(".bvecs") 
            ? load_bvecs(base_path, max_nb) 
            : load_fvecs(base_path, max_nb);
        std::cout << std::format("[Client] Loaded {} base vectors (d={})\n", total_nb, d);

        auto t_insert_start = std::chrono::high_resolution_clock::now();
        size_t sent = 0;
        size_t batch_idx = 0;
        while (sent < total_nb) {
            size_t this_batch = std::min(insert_batch, total_nb - sent);
            const float* ptr = base_data.data() + sent * d;

            auto t0 = std::chrono::high_resolution_clock::now();

            // Send opcode + batch size
            send_all(sock, &OP_INSERT, 1);
            uint64_t n64 = static_cast<uint64_t>(this_batch);
            send_all(sock, &n64, sizeof(n64));

            auto t_header_sent = std::chrono::high_resolution_clock::now();

            // Send vector data
            send_all(sock, ptr, this_batch * d * sizeof(float));

            auto t_data_sent = std::chrono::high_resolution_clock::now();
            double header_time = std::chrono::duration<double>(t_header_sent - t0).count();
            double data_time   = std::chrono::duration<double>(t_data_sent - t_header_sent).count();
            double send_time   = std::chrono::duration<double>(t_data_sent - t0).count();
            double throughput  = (this_batch * d * sizeof(float)) / (1024.0 * 1024.0 * 1024.0) / send_time;

            std::cout << std::format("[Client] INSERT batch {}: sent {} vectors (offset={}) "
                                    "[send={:.3f}s  data={:.3f}s  {:.2f} GB/s]\n",
                                    batch_idx, this_batch, sent, send_time, data_time, throughput);

            uint8_t status = STATUS_ERROR;
            recv_all(sock, &status, 1);
            if (status != STATUS_OK) {
                std::cerr << std::format("[Client] INSERT batch {} failed (status={})\n",
                                        batch_idx, status);
                ::close(sock);
                return 1;
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed     = std::chrono::duration<double>(t1 - t0).count();
            double server_ack  = std::chrono::duration<double>(t1 - t_data_sent).count();
            auto   t_total     = std::chrono::duration<double>(t1 - t_insert_start).count();

            std::cout << std::format("[Client] INSERT batch {} OK  "
                                    "[round_trip={:.3f}s  server_ack={:.3f}s  cumulative={:.1f}s  "
                                    "vectors_sent={}/{}]\n",
                                    batch_idx, elapsed, server_ack, t_total, sent + this_batch, total_nb);

            sent += this_batch;
            ++batch_idx;
        }

        auto t_insert_end = std::chrono::high_resolution_clock::now();
        double total_insert_time = std::chrono::duration<double>(t_insert_end - t_insert_start).count();
        double total_gb = (total_nb * d * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
        std::cout << std::format("[Client] INSERT complete: {} vectors in {:.1f}s  "
                                "({:.2f} GB  avg {:.2f} GB/s)\n",
                                total_nb, total_insert_time, total_gb, total_gb / total_insert_time);

        std::cout << "[Client] Sending OP_BUILD_DONE...\n";
        send_all(sock, &OP_BUILD_DONE, 1);

        uint8_t build_status = STATUS_ERROR;
        recv_all(sock, &build_status, 1);
        if (build_status != STATUS_OK) {
            std::cerr << "[Client] OP_BUILD_DONE failed.\n";
            ::close(sock);
            return 1;
        }
        std::cout << "[Client] OP_BUILD_DONE acknowledged. Index distributed to workers.\n";

        // The INSERT-phase socket is now dead — MasterTcpServer has opened a
        // new listener on the same port.  Close and reconnect before querying.
        ::close(sock);
        sock = ::socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) throw std::runtime_error("socket() failed on reconnect");
        {
            int flag = 1;
            setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
        }
        // Brief retry loop to handle the race between server close() and bind()
        for (int attempt = 0; attempt < 20; ++attempt) {
            if (::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) break;
            if (attempt == 19) throw std::runtime_error("reconnect() failed after BUILD_DONE");
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        std::cout << "[Client] Reconnected to MasterTcpServer for query phase.\n";
    }

    // ------------------------------------------------------------------
    // QUERY phase
    // ------------------------------------------------------------------
    if (do_query) {
        std::cout << std::format("[Client] Loading query vectors from {}...\n", query_path);
        auto [query_data, total_nq, d] = query_path.ends_with(".bvecs")
            ? load_bvecs(query_path, max_nq)
            : load_fvecs(query_path, max_nq);
        std::cout << std::format("[Client] Loaded {} query vectors (d={})\n", total_nq, d);

        // Load groundtruth if provided
        bool have_gt = !gt_path.empty();
        std::vector<int64_t> gt_labels;
        size_t gt_k = 0;

        if (have_gt) {
            std::cout << std::format("[Client] Loading groundtruth from {}...\n", gt_path);
            try {
                auto [gt, gt_nv, gt_dim] = load_groundtruth(gt_path, total_nq, k);
                if (gt_nv < total_nq) {
                    std::cerr << std::format("[Client] WARNING: groundtruth has {} entries but {} queries "
                                             "requested — capping nq to {}.\n",
                                             gt_nv, total_nq, gt_nv);
                    total_nq = gt_nv;
                }
                gt_labels = std::move(gt);
                gt_k      = gt_dim;
                std::cout << std::format("[Client] Groundtruth loaded: {} vectors, top-{}\n",
                                         gt_nv, gt_k);
            } catch (const std::exception& e) {
                std::cerr << std::format("[Client] WARNING: could not load groundtruth ({}). "
                                         "Proceeding without recall.\n", e.what());
                have_gt = false;
            }
        }

        for (size_t loop_iter = 0; loop_iter < query_loop; ++loop_iter) {
            std::cout << std::format("[Client] --- Query loop {}/{} ---\n", loop_iter + 1, query_loop);

            size_t queried = 0;
            size_t batch_idx = 0;
            double total_query_time = 0.0;

            // Accumulators for overall recall across the full loop iteration
            double recall_sum    = 0.0;
            size_t recall_batches = 0;

            while (queried < total_nq) {
                size_t this_batch = std::min(query_batch, total_nq - queried);
                const float* ptr = query_data.data() + queried * d;

                auto t0 = std::chrono::high_resolution_clock::now();

                // MasterTcpServer always reads [uint64 nq][uint64 k] with no
                // opcode prefix — opcodes are only used in the INSERT phase.
                uint64_t hdr[2] = {static_cast<uint64_t>(this_batch),
                                   static_cast<uint64_t>(k)};
                send_all(sock, hdr, sizeof(hdr));
                send_all(sock, ptr, this_batch * d * sizeof(float));

                auto t_sent = std::chrono::high_resolution_clock::now();
                double send_time = std::chrono::duration<double>(t_sent - t0).count();

                std::cout << std::format("[Client] QUERY batch {}: sent {} queries\n",
                                         batch_idx, this_batch);

                std::vector<float>   distances(this_batch * k);
                std::vector<int64_t> labels(this_batch * k);
                recv_all(sock, distances.data(), this_batch * k * sizeof(float));
                recv_all(sock, labels.data(),    this_batch * k * sizeof(int64_t));

                auto t1 = std::chrono::high_resolution_clock::now();
                double recv_time = std::chrono::duration<double>(t1 - t_sent).count();
                double elapsed   = std::chrono::duration<double>(t1 - t0).count();
                total_query_time += elapsed;

                std::cout << std::format("[Client] QUERY batch {} done in {:.4f}s  "
                         "[send={:.4f}s  server+recv={:.4f}s]\n",
                         batch_idx, elapsed, send_time, recv_time);

                // Print first few results
                for (size_t q = 0; q < std::min(this_batch, (size_t)3); ++q) {
                    std::cout << std::format("  Query {:4d}: ", queried + q);
                    for (size_t i = 0; i < std::min(k, (size_t)5); ++i)
                        std::cout << std::format("[id={} d={:.3f}] ",
                                                 labels[q*k+i], distances[q*k+i]);
                    std::cout << "\n";
                }

                // Recall computation
                if (have_gt) {
                    float batch_recall = compute_recall(
                        gt_labels, gt_k,
                        labels, k,
                        this_batch,
                        queried  // offset into global groundtruth
                    );
                    recall_sum    += batch_recall;
                    recall_batches += 1;

                    std::cout << std::format("[Client] QUERY batch {} recall@{}: {:.4f}\n",
                                             batch_idx, k, batch_recall);
                }

                queried += this_batch;
                ++batch_idx;
            }

            std::cout << std::format("[Client] Loop {}: queried {} vectors in {:.4f}s total\n",
                                     loop_iter + 1, total_nq, total_query_time);

            if (have_gt && recall_batches > 0) {
                float mean_recall = static_cast<float>(recall_sum / recall_batches);
                std::cout << std::format("[Client] Loop {}: mean recall@{} = {:.4f}  "
                                         "({} batches)\n",
                                         loop_iter + 1, k, mean_recall, recall_batches);
            }
        }
    }

    // ------------------------------------------------------------------
    // Shutdown
    // ------------------------------------------------------------------
    std::cout << "[Client] Sending shutdown signal...\n";
    if (do_query) {
        // MasterTcpServer shutdown: send nq=0, k=0
        uint64_t hdr[2] = {0, 0};
        send_all(sock, hdr, sizeof(hdr));
    }
    // If insert-only (do_query=false), just close the socket.
    // Do NOT send OP_SHUTDOWN or nq=0 to MasterTcpServer — that would kill
    // the query loop before a separate query client gets a chance to connect.
    ::close(sock);
    std::cout << "[Client] Done.\n";
    return 0;
}