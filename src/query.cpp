// query.cpp  (serve mode with INSERT + MasterTcpServer query phase)
//
// Rank layout is IDENTICAL to the original:
//   rank 0   = master
//   rank 1..N = workers
//
// Serve mode flow:
//
//   PHASE 1 — INSERT (opcode protocol, same TCP connection):
//     Client sends OP_INSERT batches, then OP_BUILD_DONE.
//     Master accumulates vectors via index.add(), then calls preSearch().
//     If --skip_insert is passed the INSERT phase is skipped entirely and
//     the index is loaded from disk (identical to the old serve mode).
//
//   PHASE 2 — QUERY (MasterTcpServer, same or new TCP connections):
//     After preSearch() the master starts a MasterTcpServer on --tcp_port
//     (insert used --insert_port, or the same port if --skip_insert).
//     The master thread drives the MPI search loop by dequeuing SearchJobs
//     from the shared job queue, signalling workers, calling index.search(),
//     and marking each job done.  MasterTcpServer handles all TCP I/O on its
//     background thread — the master thread never touches sockets during this
//     phase.
//
// TCP opcodes used only during the INSERT phase:
//   OP_INSERT     (0x01)  [uint8][uint64 n][n*d floats]  -> [uint8 status]
//   OP_BUILD_DONE (0x02)  [uint8]                        -> [uint8 status]
//   OP_SHUTDOWN   (0x00)  [uint8]  (aborts before BUILD_DONE)
//
// MasterTcpServer query protocol (see master_server.h):
//   Client -> Master: [uint64 nq][uint64 k][nq*d floats]
//   Master -> Client: [nq*k floats distances][nq*k int64 labels]
//   Shutdown:         [uint64 nq=0][uint64 k=0]

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <mpi.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <argparse/argparse.hpp>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "master_server.h"
#include "tribase.h"
#include "utils.h"

using namespace harmony;
using namespace std;

static constexpr int WORKER_SIGNAL_SEARCH   = 1;
static constexpr int WORKER_SIGNAL_SHUTDOWN = 0;
static constexpr int WORKER_SIGNAL_INIT     = 2;  // signals workers to call init() before preSearch()

// INSERT-phase opcodes (only used before MasterTcpServer takes over)
static constexpr uint8_t OP_SHUTDOWN   = 0x00;
static constexpr uint8_t OP_INSERT     = 0x01;
static constexpr uint8_t OP_BUILD_DONE = 0x02;

static constexpr uint8_t STATUS_OK    = 0x00;
static constexpr uint8_t STATUS_ERROR = 0x01;

// ---------------------------------------------------------------------------
// Reliable TCP helpers (INSERT phase only — MasterTcpServer has its own)
// ---------------------------------------------------------------------------
static void send_all(int fd, const void* buf, size_t n) {
    const char* p = static_cast<const char*>(buf);
    while (n > 0) {
        ssize_t s = ::send(fd, p, n, 0);
        if (s <= 0) throw std::runtime_error("TCP send failed");
        p += s; n -= s;
    }
}
static void recv_all(int fd, void* buf, size_t n) {
    char* p = static_cast<char*>(buf);
    while (n > 0) {
        ssize_t s = ::recv(fd, p, n, 0);
        if (s <= 0) throw std::runtime_error("TCP recv failed / client disconnected");
        p += s; n -= s;
    }
}

static bool str_lower_equal(const std::string& a, const std::string& b) {
    return std::equal(a.begin(), a.end(), b.begin(), b.end(),
                      [](char a, char b) { return std::tolower(a) == std::tolower(b); });
}

static int workerMain(int rank, bool cut, Index::SearchMode mode, bool blockSend, bool minorCut) {
    if (mode == Index::SearchMode::DIVIDE_IVF) {
        BaseWorker worker;
        worker.init(rank);
        worker.search(cut);
    } else if (mode == Index::SearchMode::DIVIDE_DIM) {
        Worker node;
        node.init(rank, blockSend);
        node.search(cut);
        node.postSearch();
    } else {
        GroupWorker worker;
        worker.init(rank, blockSend);
        worker.receiveQuery();
        worker.search(cut, minorCut);
    }
    return 0;
}

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("harmony");
    program.add_argument("--benchmarks_path").default_value(std::string("/home/xuqian/Triangle/benchmarks"));
    program.add_argument("--dataset").default_value(std::string("msong"));
    program.add_argument("--input_format").default_value(std::string("fvecs"));
    program.add_argument("--output_format").default_value(std::string("bin"));
    program.add_argument("--k").default_value(100ul)
        .action([](const std::string& v) -> size_t { return std::stoul(v); });
    program.add_argument("--nprobes").default_value(std::vector<size_t>({0ul})).nargs(0,100).scan<'u',size_t>();
    program.add_argument("--opt_levels").default_value(std::vector<std::string>({"OPT_NONE"})).nargs(0,10);
    program.add_argument("--train_only").default_value(false).implicit_value(true);
    program.add_argument("--cache").default_value(false).implicit_value(true);
    program.add_argument("--sub_nprobe_ratio").default_value(1.0f)
        .action([](const std::string& v) -> float { return std::stof(v); });
    program.add_argument("--metric").default_value("l2");
    program.add_argument("--run_faiss").default_value(false).implicit_value(true);
    program.add_argument("--loop").default_value(1ul)
        .action([](const std::string& v) -> size_t { return std::stoul(v); });
    program.add_argument("--nlist").default_value(0ul)
        .action([](const std::string& v) -> size_t { return std::stoul(v); });
    program.add_argument("--verbose").default_value(false).implicit_value(true);
    program.add_argument("--ratios").default_value(std::vector<float>({1.0f})).nargs(0,100).scan<'f',float>();
    program.add_argument("--csv").default_value(std::string(""));
    program.add_argument("--dataset_info").default_value(false).implicit_value(true);
    program.add_argument("--early_stop").default_value(false).implicit_value(true);
    program.add_argument("--block").default_value(0ul)
        .action([](const std::string& v) -> size_t { return std::stoul(v); });
    program.add_argument("--warmup_list_size").default_value(0ul)
        .action([](const std::string& v) -> size_t { return std::stoul(v); });
    program.add_argument("--warmup_list").default_value(10ul)
        .action([](const std::string& v) -> size_t { return std::stoul(v); });
    program.add_argument("--disablePruning").default_value(false).implicit_value(true);
    program.add_argument("--minorCut").default_value(false).implicit_value(true);
    program.add_argument("--disableOrderOpt").default_value(false).implicit_value(true);
    program.add_argument("--blockSend").default_value(false).implicit_value(true);
    program.add_argument("--fullWarmUp").default_value(false).implicit_value(true);
    program.add_argument("--group").default_value(1ul)
        .action([](const std::string& v) -> size_t { return std::stoul(v); });
    program.add_argument("--team").default_value(1ul)
        .action([](const std::string& v) -> size_t { return std::stoul(v); });
    program.add_argument("--mode").default_value(std::string("group"))
        .choices("base","group","block","original","brute");
    program.add_argument("--HardInBalance").default_value(false).implicit_value(true);
    program.add_argument("--HardInBalanceRatio").default_value(0.0f).scan<'f',float>();
    program.add_argument("--HardInBalanceTeamRatio").default_value(0.0f).scan<'f',float>();
    program.add_argument("--HardInBalanceTeam").default_value(0ul)
        .action([](const std::string& v) -> size_t { return std::stoul(v); });
    program.add_argument("--brute_force_search_ratio").default_value(1.0f).scan<'f',float>();

    // Serve mode
    program.add_argument("--serve")
        .help("Accept TCP client connections and serve queries")
        .default_value(false).implicit_value(true);
    program.add_argument("--nb")
        .help("Number of base vectors (serve mode, skips file read)")
        .default_value(size_t(0))
        .scan<'u', size_t>();
    program.add_argument("--dim")
        .help("Vector dimension (serve mode, skips file read)")
        .default_value(size_t(0))
        .scan<'u', size_t>();
    program.add_argument("--nprobe")
        .help("nprobe for serve mode")
        .default_value(100ul)
        .action([](const std::string& v) -> size_t { return std::stoul(v); });
    program.add_argument("--tcp_port")
        .help("TCP port for MasterTcpServer (query phase)")
        .default_value(7777ul)
        .action([](const std::string& v) -> size_t { return std::stoul(v); });

    // INSERT phase arguments
    program.add_argument("--skip_insert")
        .help("Skip INSERT phase — load index from disk and go straight to query serving "
              "(replicates old serve-mode behaviour)")
        .default_value(false).implicit_value(true);
    program.add_argument("--insert_port")
        .help("TCP port for the INSERT phase (defaults to --tcp_port if not set)")
        .default_value(0ul)
        .action([](const std::string& v) -> size_t { return std::stoul(v); });
    program.add_argument("--train_data")
        .help("Path to seed .fvecs for train() in INSERT mode "
              "(falls back to the dataset base file if not set)")
        .default_value(std::string(""));

    try { program.parse_args(argc, argv); }
    catch (const std::runtime_error& err) { std::cerr << err.what() << "\n" << program; return 1; }

    int pro;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &pro);

    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    int workerCount = worldSize - 1;  // rank 0 is master

    bool serve_mode  = program.get<bool>("serve");
    bool skip_insert = program.get<bool>("skip_insert");

    bool disableOrderOptimize = program.get<bool>("disableOrderOpt");
    bool pruning    = !program.get<bool>("disablePruning");
    bool minorCut   = program.get<bool>("minorCut");
    bool run_faiss  = program.get<bool>("run_faiss");
    bool hardInBalance      = program.get<bool>("HardInBalance");
    size_t hardInBalanceTeam = program.get<size_t>("HardInBalanceTeam");
    float inBalanceRatio     = program.get<float>("HardInBalanceRatio");
    float inBalanceTeamRatio = program.get<float>("HardInBalanceTeamRatio");
    bool blockSend  = program.get<bool>("blockSend");
    bool fullWarmUp = program.get<bool>("fullWarmUp");

    size_t groupCount = program.get<size_t>("group");
    size_t teamCount  = program.get<size_t>("team");
    size_t teamSize   = workerCount / std::max(teamCount, (size_t)1);
    size_t blockCount = program.get<size_t>("block");

    std::string mode_str = program.get<std::string>("--mode");
    Index::SearchMode searchMode = Index::SearchMode::DIVIDE_GROUP;
    if      (mode_str == "original") searchMode = Index::SearchMode::ORIGINAL;
    else if (mode_str == "block")    searchMode = Index::SearchMode::DIVIDE_DIM;
    else if (mode_str == "base")     searchMode = Index::SearchMode::DIVIDE_IVF;
    else if (mode_str == "brute")    searchMode = Index::SearchMode::BRUTE_FORCE;

    // =======================================================================
    // WORKER RANKS (rank != 0) — identical to original, plus serve loop
    // =======================================================================
    if (rank != 0) {
        if (!run_faiss) {
            if (!serve_mode) {
                // Benchmark mode: unchanged — master calls preSearch() which ends
                // with MPI_Barrier; workers hit that barrier here, then search.
                MPI_Barrier(MPI_COMM_WORLD);
                workerMain(rank, pruning, searchMode, blockSend, minorCut);
            } else if (skip_insert) {
                // skip_insert serve mode: master calls preSearch() which does
                // MPI_Send to each worker then ends with MPI_Barrier.
                // Workers must call init() first (issuing matching MPI_Recvs),
                // then hit MPI_Barrier (matching preSearch's final barrier).
                if (searchMode == Index::SearchMode::DIVIDE_GROUP) {
                    GroupWorker worker;
                    worker.init(rank, blockSend);  // MPI_Recvs match preSearch() MPI_Sends
                    MPI_Barrier(MPI_COMM_WORLD);   // matches preSearch() final MPI_Barrier
                    while (true) {
                        int signal = WORKER_SIGNAL_SHUTDOWN;
                        MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
                        if (signal == WORKER_SIGNAL_SHUTDOWN) break;
                        worker.receiveQuery();
                        worker.search(pruning, minorCut);
                    }
                } else if (searchMode == Index::SearchMode::DIVIDE_DIM) {
                    Worker worker;
                    worker.init(rank, blockSend);
                    MPI_Barrier(MPI_COMM_WORLD);
                    while (true) {
                        int signal = WORKER_SIGNAL_SHUTDOWN;
                        MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
                        if (signal == WORKER_SIGNAL_SHUTDOWN) break;
                        worker.search(pruning);
                        worker.postSearch();
                    }
                } else {
                    MPI_Barrier(MPI_COMM_WORLD);
                    while (true) {
                        int signal = WORKER_SIGNAL_SHUTDOWN;
                        MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
                        if (signal == WORKER_SIGNAL_SHUTDOWN) break;
                        workerMain(rank, pruning, searchMode, blockSend, minorCut);
                    }
                }
            } else {
                // INSERT serve mode: master is handling TCP while workers wait.
                // Workers must NOT call init() or hit MPI_Barrier until master
                // broadcasts WORKER_SIGNAL_INIT (just before preSearch()).
                // preSearch() sends data via MPI_Send — init() issues the
                // matching MPI_Recvs.  preSearch() ends with MPI_Barrier —
                // workers hit it after init() returns.
                {
                    int signal = WORKER_SIGNAL_SHUTDOWN;
                    MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    if (signal == WORKER_SIGNAL_SHUTDOWN) {
                        // Master aborted before BUILD_DONE
                        MPI_Finalize();
                        return 0;
                    }
                    // WORKER_SIGNAL_INIT received — fall through to init()
                }

                if (searchMode == Index::SearchMode::DIVIDE_GROUP) {
                    GroupWorker worker;
                    worker.init(rank, blockSend);  // MPI_Recvs match preSearch() MPI_Sends
                    MPI_Barrier(MPI_COMM_WORLD);   // matches preSearch() final MPI_Barrier
                    while (true) {
                        int signal = WORKER_SIGNAL_SHUTDOWN;
                        MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
                        if (signal == WORKER_SIGNAL_SHUTDOWN) break;
                        worker.receiveQuery();
                        worker.search(pruning, minorCut);
                    }
                } else if (searchMode == Index::SearchMode::DIVIDE_DIM) {
                    Worker worker;
                    worker.init(rank, blockSend);
                    while (true) {
                        int signal = WORKER_SIGNAL_SHUTDOWN;
                        MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
                        if (signal == WORKER_SIGNAL_SHUTDOWN) break;
                        worker.search(pruning);
                        worker.postSearch();
                    }
                } else {
                    while (true) {
                        int signal = WORKER_SIGNAL_SHUTDOWN;
                        MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
                        if (signal == WORKER_SIGNAL_SHUTDOWN) break;
                        workerMain(rank, pruning, searchMode, blockSend, minorCut);
                    }
                }
            }
        }
        MPI_Finalize();
        return 0;
    }

    // =======================================================================
    // MASTER (rank == 0)
    // =======================================================================
    cout << YELLOW << std::format("Mode: {}", Index::to_string(searchMode)) << RESET << endl;

    std::vector<size_t> nprobes = program.get<std::vector<size_t>>("nprobes");
    std::vector<std::string> opt_levels_str = program.get<std::vector<std::string>>("opt_levels");
    std::vector<float> ratios = program.get<std::vector<float>>("ratios");
    size_t k        = program.get<size_t>("k");
    size_t loop     = program.get<size_t>("loop");
    size_t nlist    = program.get<size_t>("nlist");
    bool verbose    = program.get<bool>("verbose");
    bool train_only = program.get<bool>("train_only");
    bool cache      = program.get<bool>("cache");
    float sub_nprobe_ratio  = program.get<float>("sub_nprobe_ratio");
    size_t serve_nprobe     = program.get<size_t>("nprobe");
    uint16_t tcp_port       = static_cast<uint16_t>(program.get<size_t>("tcp_port"));
    size_t insert_port_arg  = program.get<size_t>("insert_port");
    uint16_t insert_port    = insert_port_arg
                                  ? static_cast<uint16_t>(insert_port_arg)
                                  : tcp_port;
    std::string train_data_path = program.get<std::string>("train_data");

    std::string benchmarks_path = program.get<std::string>("benchmarks_path");
    std::string dataset         = program.get<std::string>("dataset");
    std::string input_format    = program.get<std::string>("input_format");
    std::string output_format   = program.get<std::string>("output_format");
    std::string metric_str      = program.get<std::string>("metric");

    MetricType metric;
    if      (str_lower_equal(metric_str, "l2")) metric = MetricType::METRIC_L2;
    else if (str_lower_equal(metric_str, "ip")) metric = MetricType::METRIC_IP;
    else throw std::runtime_error("Invalid metric type");

    std::string base_path  = std::format("{}/{}/origin/{}_base.{}",  benchmarks_path, dataset, dataset, input_format);
    std::string query_path = std::format("{}/{}/origin/{}_query.{}", benchmarks_path, dataset, dataset, input_format);

    size_t nb = program.get<size_t>("--nb");
    size_t d  = program.get<size_t>("--dim");
    if (!serve_mode || nb == 0 || d == 0)
        std::tie(nb, d) = loadXvecsInfo(base_path);
    if (nlist == 0) nlist = static_cast<size_t>(std::sqrt(nb));
    size_t sub_nlist  = std::sqrt(nb / nlist);
    size_t sub_nprobe = std::max(static_cast<size_t>(sub_nlist * sub_nprobe_ratio), 1ul);

    OptLevel added_opt_levels = OptLevel::OPT_NONE;
    for (const auto& s : opt_levels_str)
        added_opt_levels = static_cast<OptLevel>(
            static_cast<int>(added_opt_levels) | static_cast<int>(str2OptLevel(s)));

    auto get_index_path = [&]() {
        int target = static_cast<int>(added_opt_levels);
        for (int i = 0; i < 8; i++) {
            if ((target & i) == target) {
                std::string p = std::format("{}/{}/index/index_nlist_{}_opt_{}_subNprobeRatio_{}.index",
                                            benchmarks_path, dataset, nlist, i, sub_nprobe_ratio);
                if (std::filesystem::exists(p)) return p;
            }
        }
        return std::format("{}/{}/index/index_nlist_{}_opt_{}_subNprobeRatio_{}.index",
                           benchmarks_path, dataset, nlist, static_cast<int>(added_opt_levels), sub_nprobe_ratio);
    };

    std::string index_path       = get_index_path();
    std::string faiss_index_path = std::format("{}/{}/index/faiss_index_nlist_{}.index",
                                               benchmarks_path, dataset, nlist);
    prepareDirectory(faiss_index_path);

    std::unique_ptr<float[]> base = nullptr;
    faiss::IndexFlatL2 quantizer(d);
    std::unique_ptr<faiss::IndexIVFFlat> index_faiss =
        std::make_unique<faiss::IndexIVFFlat>(&quantizer, d, nlist);

    auto train_load_faiss = [&]() {
        if (!std::filesystem::exists(faiss_index_path)) {
            if (base == nullptr) std::tie(base, nb, d) = loadXvecs(base_path);
            index_faiss->train(nb, base.get());
            index_faiss->add(nb, base.get());
            faiss::write_index(index_faiss.get(), faiss_index_path.c_str());
        } else {
            index_faiss.reset(dynamic_cast<faiss::IndexIVFFlat*>(
                faiss::read_index(faiss_index_path.c_str())));
        }
    };

    // =======================================================================
    // SERVE MODE
    // =======================================================================
    if (serve_mode) {
        size_t warmUpSearchList     = program.get<size_t>("warmup_list");
        size_t warmUpSearchListSize = program.get<size_t>("warmup_list_size");
        if (warmUpSearchList > 0 && warmUpSearchListSize == 0) warmUpSearchListSize = k;

        Index::Param param;
        param.orderOptimize          = !disableOrderOptimize;
        param.mode                   = searchMode;
        param.pruning                = pruning;
        param.fullWarmUp             = fullWarmUp;
        param.groupCount             = groupCount;
        param.teamCount              = teamCount;
        param.teamSize               = teamSize;
        param.hardInBalance          = hardInBalance;
        param.hardInBalanceTeam      = hardInBalanceTeam;
        param.hardInBalanceRatio     = inBalanceRatio;
        param.hardInBalanceTeamRatio = inBalanceTeamRatio;

        Index index;
        size_t total_inserted = 0;

        // -------------------------------------------------------------------
        // PHASE 1A — skip_insert: load/build index from disk (old behaviour)
        // -------------------------------------------------------------------
        if (skip_insert) {
            if (std::filesystem::exists(index_path) && cache) {
                if (verbose) cout << std::format("Loading index from {}\n", index_path);
                index.load_index(index_path);
                if (verbose) cout << "Index loaded\n";
            } else {
                std::tie(base, nb, d) = loadXvecs(base_path);
                nlist = static_cast<size_t>(std::sqrt(nb));
                index = Index(d, nlist, 0, metric, added_opt_levels, OPT_ALL, sub_nlist, sub_nprobe, verbose);
                index.train(nb, base.get());
                index.add(nb, base.get());
                if (verbose) cout << "Index trained\n";
                index.save_index(index_path);
                if (verbose) cout << std::format("Index saved to {}\n", index_path);
            }
            total_inserted = nb;

            index.nprobe = serve_nprobe;
            if (index.nprobe > nlist) index.nprobe = nlist;

            std::string tmp_path = std::format("{}/{}/index/index_nlist_{}_{}.index",
                                               benchmarks_path, dataset, nlist,
                                               Index::to_string(searchMode));
            index.preSearch(nb, workerCount, blockCount,
                            warmUpSearchList, warmUpSearchListSize, &param, tmp_path);

        // -------------------------------------------------------------------
        // PHASE 1B — INSERT phase via opcode protocol
        //
        //   1. Train on seed data.
        //   2. Open TCP server on --insert_port.
        //   3. Accept one client; receive OP_INSERT batches + OP_BUILD_DONE.
        //   4. Call preSearch() to distribute the index to workers.
        //   5. Close the insert connection.
        // -------------------------------------------------------------------
        } else {
            std::string seed_path = train_data_path.empty() ? base_path : train_data_path;
            if (!std::filesystem::exists(seed_path)) {
                cerr << std::format("[Master] Seed data not found at: {}\n", seed_path);
                MPI_Finalize(); return 1;
            }

            size_t seed_nb, seed_d;
            std::unique_ptr<float[]> seed_data;
            std::tie(seed_data, seed_nb, seed_d) = loadXvecs(seed_path);
            d = seed_d;

            if (nlist == 0) nlist = static_cast<size_t>(std::sqrt(seed_nb));
            sub_nlist  = std::sqrt(seed_nb / nlist);
            sub_nprobe = std::max(static_cast<size_t>(sub_nlist * sub_nprobe_ratio), 1ul);

            cout << YELLOW
                 << std::format("[Master] Training on {} seed vectors (d={}, nlist={})...\n",
                                seed_nb, d, nlist)
                 << RESET;

            index = Index(d, nlist, 0, metric, added_opt_levels, OPT_ALL, sub_nlist, sub_nprobe, verbose);
            index.train(seed_nb, seed_data.get());
            seed_data.reset();

            cout << YELLOW << "[Master] Training complete. Opening INSERT port...\n" << RESET;

            // Open TCP server for the INSERT phase
            int ins_server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
            if (ins_server_fd < 0) { cerr << "socket() failed\n"; MPI_Finalize(); return 1; }
            {
                int opt_val = 1;
                setsockopt(ins_server_fd, SOL_SOCKET, SO_REUSEADDR, &opt_val, sizeof(opt_val));
            }
            sockaddr_in ins_addr{};
            ins_addr.sin_family      = AF_INET;
            ins_addr.sin_addr.s_addr = INADDR_ANY;
            ins_addr.sin_port        = htons(insert_port);
            if (::bind(ins_server_fd, reinterpret_cast<sockaddr*>(&ins_addr), sizeof(ins_addr)) < 0) {
                cerr << std::format("[Master] bind() failed on insert port {}\n", insert_port);
                MPI_Finalize(); return 1;
            }
            ::listen(ins_server_fd, 1);

            cout << YELLOW
                 << std::format("[Master] INSERT server listening on port {}\n", insert_port)
                 << "[Master] Waiting for insert client...\n"
                 << RESET;

            sockaddr_in ins_client_addr{};
            socklen_t   ins_client_len = sizeof(ins_client_addr);
            int ins_client_fd = ::accept(ins_server_fd,
                                         reinterpret_cast<sockaddr*>(&ins_client_addr),
                                         &ins_client_len);
            if (ins_client_fd < 0) {
                cerr << "[Master] INSERT accept() failed\n"; MPI_Finalize(); return 1;
            }
            {
                int flag = 1;
                setsockopt(ins_client_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
            }
            cout << YELLOW << "[Master] Insert client connected.\n" << RESET;

            // Buffer all vectors — Index::add() resets lists on every call,
            // so we must call it once with the full dataset on BUILD_DONE.
            std::vector<float> all_vectors;
            all_vectors.reserve(1000000 * d);

            // INSERT opcode loop
            bool build_done = false;
            while (!build_done) {
                uint8_t op = OP_SHUTDOWN;
                try { recv_all(ins_client_fd, &op, 1); }
                catch (...) {
                    cout << "[Master] Insert client disconnected before BUILD_DONE.\n";
                    break;
                }

                if (op == OP_SHUTDOWN) {
                    cout << "[Master] OP_SHUTDOWN during INSERT phase — aborting.\n";
                    break;
                }

                // OP_INSERT ------------------------------------------------
                if (op == OP_INSERT) {
                    uint64_t n_recv = 0;
                    try { recv_all(ins_client_fd, &n_recv, sizeof(n_recv)); }
                    catch (...) {
                        send_all(ins_client_fd, &STATUS_ERROR, 1);
                        break;
                    }
                    size_t n = static_cast<size_t>(n_recv);
                    size_t old_size = total_inserted * d;
                    all_vectors.resize(old_size + n * d);

                    auto t_recv_start = std::chrono::high_resolution_clock::now();
                    cout << std::format("[Master] OP_INSERT: receiving {} vectors (d={})...\n", n, d);

                    try { recv_all(ins_client_fd, all_vectors.data() + old_size, n * d * sizeof(float)); }
                    catch (...) {
                        send_all(ins_client_fd, &STATUS_ERROR, 1);
                        break;
                    }

                    auto t_recv_end = std::chrono::high_resolution_clock::now();
                    double recv_time  = std::chrono::duration<double>(t_recv_end - t_recv_start).count();
                    double recv_gb    = (n * d * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
                    double throughput = recv_gb / recv_time;

                    total_inserted += n;
                    cout << std::format("[Master] OP_INSERT: buffered {} vectors (total={})  "
                                        "[recv={:.3f}s  {:.2f} GB/s]\n",
                                        n, total_inserted, recv_time, throughput);
                    send_all(ins_client_fd, &STATUS_OK, 1);
                }

                // OP_BUILD_DONE --------------------------------------------
                else if (op == OP_BUILD_DONE) {
                    cout << std::format("[Master] OP_BUILD_DONE: calling index.add() on {} vectors...\n",
                                        total_inserted);
                    {
                        Stopwatch sw;
                        index.add(total_inserted, all_vectors.data());
                        cout << std::format("[Master] index.add() done in {:.3f}s\n", sw.elapsedSeconds());
                        // Save index to disk for future --cache --skip_insert runs
                        cout << std::format("[Master] Saving index to {}...\n", index_path);
                        index.save_index(index_path);
                        cout << std::format("[Master] Index saved.\n");
                    }
                    cout << "[Master] Sample IDs in list 0: ";
                    for (size_t i = 0; i < std::min((size_t)5, index.lists[0].get_list_size()); ++i)
                        cout << index.lists[0].candidate_id[i] << " ";
                    cout << "\n";
                    all_vectors.clear();
                    all_vectors.shrink_to_fit();

                    cout << std::format("[Master] OP_BUILD_DONE: calling preSearch() on {} vectors "
                                        "with {} workers...\n",
                                        total_inserted, workerCount);

                    index.nprobe = serve_nprobe;
                    if (index.nprobe > nlist) index.nprobe = nlist;

                    // Signal workers to unblock and call init(), which issues
                    // the MPI_Recvs that match preSearch()'s MPI_Sends.
                    // This must happen BEFORE preSearch() — not after.
                    {
                        int sig = WORKER_SIGNAL_INIT;
                        MPI_Bcast(&sig, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    }

                    std::string tmp_path = std::format("{}/{}/index/index_nlist_{}_{}.index",
                                                       benchmarks_path, dataset, nlist,
                                                       Index::to_string(searchMode));
                    Stopwatch sw;
                    index.preSearch(total_inserted, workerCount, blockCount,
                                    warmUpSearchList, warmUpSearchListSize, &param, tmp_path);
                    double elapsed = sw.elapsedSeconds();

                    cout << std::format("[Master] preSearch complete in {:.3f}s\n", elapsed);

                    send_all(ins_client_fd, &STATUS_OK, 1);
                    build_done = true;
                }

                else {
                    cerr << std::format("[Master] Unknown INSERT-phase opcode 0x{:02x}\n", op);
                    send_all(ins_client_fd, &STATUS_ERROR, 1);
                }
            }

            ::close(ins_client_fd);
            ::close(ins_server_fd);

            if (!build_done) {
                // Client disconnected or sent SHUTDOWN before BUILD_DONE.
                // Workers are blocked on MPI_Bcast waiting for INIT.
                // Just send SHUTDOWN directly — workers check for it first.
                cerr << "[Master] BUILD_DONE never received — shutting down workers.\n";
                int sig = WORKER_SIGNAL_SHUTDOWN;
                MPI_Bcast(&sig, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Finalize();
                return 1;
            }
        } // end INSERT phase

        // -------------------------------------------------------------------
        // PHASE 2 — QUERY via MasterTcpServer
        //
        // MasterTcpServer accepts connections and posts SearchJobs to the
        // shared queue on its background thread.  The master thread (here)
        // is the sole MPI driver: it dequeues jobs, broadcasts SEARCH to
        // workers, calls index.search(), then marks the job done so
        // MasterTcpServer can send the results back to the TCP client.
        //
        // This cleanly separates TCP I/O (MasterTcpServer background thread)
        // from MPI coordination (master thread here), which is the design
        // intent of MasterTcpServer.
        // -------------------------------------------------------------------
        auto job_queue   = std::make_shared<std::queue<std::shared_ptr<SearchJob>>>();
        auto queue_mutex = std::make_shared<std::mutex>();
        auto queue_cv    = std::make_shared<std::condition_variable>();

        MasterTcpServer tcp_server("0.0.0.0", tcp_port, d,
                                   job_queue, queue_mutex, queue_cv);
        tcp_server.Start();

        cout << YELLOW << "[Master] MasterTcpServer started on port " << tcp_port
             << " — entering MPI query loop.\n" << RESET;

        // Groundtruth for recall/CSV logging (same as original serve mode)
        std::string gt_path = std::format("{}/{}/result/groundtruth_{}.{}",
                                          benchmarks_path, dataset, k, output_format);

        // MPI query loop
        while (true) {
            std::shared_ptr<SearchJob> job;
            {
                std::unique_lock<std::mutex> lk(*queue_mutex);
                queue_cv->wait(lk, [&] {
                    return !job_queue->empty() || tcp_server.IsDone();
                });

                if (job_queue->empty()) break;  // server shut down, no more jobs

                job = job_queue->front();
                job_queue->pop();
            }

            size_t nq    = static_cast<size_t>(job->nq);
            size_t job_k = static_cast<size_t>(job->k);

            cout << std::format("[Master] MPI search: nq={} k={}\n", nq, job_k);

            // Signal workers: SEARCH
            int sig = WORKER_SIGNAL_SEARCH;
            MPI_Bcast(&sig, 1, MPI_INT, 0, MPI_COMM_WORLD);

            // MPI_Barrier matches the one at the top of worker::receiveQuery()
            MPI_Barrier(MPI_COMM_WORLD);

            // Results are written directly into the job; MasterTcpServer reads them
            job->distances.resize(nq * job_k);
            job->labels.resize(nq * job_k);

            Stopwatch search_timer;
            index.search(nq, job->vectors.data(), job_k,
                         job->distances.data(), job->labels.data());
            double search_time = search_timer.elapsedSeconds();

            // Recall + CSV logging (identical to original serve mode)
            if (std::filesystem::exists(gt_path)) {
                auto gt_I = std::make_unique<idx_t[]>(job_k * nq);
                auto gt_D = std::make_unique<float[]>(job_k * nq);
                loadResults(gt_path, gt_I.get(), gt_D.get(), nq, job_k);

                float recall = calculate_recall(job->labels.data(), job->distances.data(),
                                                gt_I.get(), gt_D.get(), nq, job_k, metric);
                float r2     = calculate_r2(job->labels.data(), job->distances.data(),
                                            gt_I.get(), gt_D.get(), nq, job_k, metric);

                Stats stats;
                stats.mode       = Index::to_string(searchMode);
                stats.nb         = total_inserted;
                stats.nq         = nq;
                stats.d          = d;
                stats.k          = job_k;
                stats.nlist      = nlist;
                stats.nprobe     = serve_nprobe;
                stats.worker     = workerCount;
                stats.block      = blockCount;
                stats.group      = groupCount;
                stats.team       = teamCount;
                stats.divideIVF          = false;
                stats.disableOrderOptimize = disableOrderOptimize;
                stats.blockSend          = blockSend;
                stats.pruning            = pruning;
                stats.opt_level          = added_opt_levels;
                stats.query_time         = search_time;
                stats.original_time      = 0;
                stats.faiss_query_time   = 0;
                stats.trainTime          = index.trainTime;
                stats.addTime            = index.addTime;
                stats.preSearchTime      = index.preSearchTime;
                stats.recall             = recall;
                stats.r2                 = r2;
                stats.variance           = 0;
                stats.inBalanceRatio     = 0;
                stats.inBalanceRatioTeam = 0;
                stats.brute_ratio        = 0;
                stats.simi_ratio         = 0;
                stats.print();

                std::string log_path = std::format("{}/{}/result/log_serve.csv",
                                                   benchmarks_path, dataset);
                stats.myToCsv(log_path, true, dataset);
                cout << std::format("[Master] recall={:.4f} r2={:.4f} time={:.4f}s\n",
                                     recall, r2, search_time);
            } else {
                cout << YELLOW
                     << std::format("[Master] No groundtruth at {} — run benchmark mode first\n",
                                    gt_path)
                     << RESET;
                cout << std::format("[Master] search time: {:.4f}s\n", search_time);
            }

            // Mark job done — MasterTcpServer::ServeLoop wakes and sends results
            {
                std::lock_guard<std::mutex> lk(job->mtx);
                job->done = true;
            }
            job->cv.notify_one();
        }

        // Drain any jobs that arrived between the last dequeue and shutdown signal
        {
            std::lock_guard<std::mutex> lk(*queue_mutex);
            while (!job_queue->empty()) {
                auto j = job_queue->front();
                job_queue->pop();
                std::lock_guard<std::mutex> jlk(j->mtx);
                j->done = true;
                j->cv.notify_one();
            }
        }

        // Broadcast shutdown to workers and clean up
        {
            int sig = WORKER_SIGNAL_SHUTDOWN;
            MPI_Bcast(&sig, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
        tcp_server.Shutdown();

        MPI_Finalize();
        return 0;
    }

    // =======================================================================
    // BENCHMARK MODE — original logic unchanged
    // =======================================================================
    Index index;
    if (std::filesystem::exists(index_path) && cache) {
        if (verbose) cout << std::format("Loading index from {}\n", index_path);
        index.load_index(index_path);
        if (verbose) cout << "Index loaded\n";
    } else {
        std::tie(base, nb, d) = loadXvecs(base_path);
        nlist = static_cast<size_t>(std::sqrt(nb));
        index = Index(d, nlist, 0, metric, added_opt_levels, OPT_ALL, sub_nlist, sub_nprobe, verbose);
        index.train(nb, base.get());
        index.add(nb, base.get());
        if (verbose) cout << "Index trained\n";
        index.save_index(index_path);
        if (verbose) cout << std::format("Index saved to {}\n", index_path);
    }

    if (train_only) { MPI_Finalize(); return 0; }

    size_t warmUpSearchList     = program.get<size_t>("warmup_list");
    size_t warmUpSearchListSize = program.get<size_t>("warmup_list_size");
    if (warmUpSearchList > 0 && warmUpSearchListSize == 0) warmUpSearchListSize = k;
    if (nprobes.back() == 0) nprobes.back() = nlist;

    auto [query, nq, _] = loadXvecs(query_path);

    cout << YELLOW
         << std::format("[dim:{}, nb:{}, nq:{}, k:{}, worker:{}]", d, nb, nq, k, workerCount)
         << RESET << endl;

    std::string groundtruth_path = std::format("{}/{}/result/groundtruth_{}.{}",
                                               benchmarks_path, dataset, k, output_format);
    auto ground_truth_I = std::make_unique<idx_t[]>(k * nq);
    auto ground_truth_D = std::make_unique<float[]>(k * nq);

    if (!std::filesystem::exists(groundtruth_path)) {
        train_load_faiss();
        index_faiss->nprobe = nlist;
        index_faiss->search(nq, query.get(), k, ground_truth_D.get(), ground_truth_I.get());
        writeResultsToFile(ground_truth_I.get(), ground_truth_D.get(), nq, k, groundtruth_path);
    } else {
        loadResults(groundtruth_path, ground_truth_I.get(), ground_truth_D.get(), nq, k);
    }

    if (run_faiss) {
        if (!index_faiss->is_trained) train_load_faiss();
        std::string log_path = std::format("{}/{}/result/log_faiss.csv", benchmarks_path, dataset);
        CsvWriter writer(log_path, {"dataset","nlist","nprobe","time","recall","r2"}, true, false);
        auto fd = std::make_unique<float[]>(k * nq);
        auto fi = std::make_unique<idx_t[]>(k * nq);
        for (size_t np : nprobes) {
            index_faiss->nprobe = np;
            Stopwatch sw;
            for (size_t j = 0; j < loop; j++)
                index_faiss->search(nq, query.get(), k, fd.get(), fi.get());
            double t = sw.elapsedSeconds() / loop;
            float recall = calculate_recall(fi.get(), fd.get(),
                                            ground_truth_I.get(), ground_truth_D.get(), nq, k, metric);
            float r2 = calculate_r2(fi.get(), fd.get(),
                                    ground_truth_I.get(), ground_truth_D.get(), nq, k, metric);
            cout << std::format("Faiss nprobe:{} time:{:.4f} recall:{:.4f}", np, t, recall) << endl;
            writer << dataset << nlist << np << t << recall << r2 << endl;
        }
        MPI_Finalize();
        return 0;
    }

    std::string log_path = std::format("{}/{}/result/log.csv", benchmarks_path, dataset);

    for (size_t nprobe : nprobes) {
        index.nprobe = nprobe;
        auto distances = std::make_unique<float[]>(nq * k);
        auto labels    = std::make_unique<idx_t[]>(nq * k);

        MPI_Barrier(MPI_COMM_WORLD);

        Index::Param param;
        param.orderOptimize          = !disableOrderOptimize;
        param.mode                   = searchMode;
        param.pruning                = pruning;
        param.fullWarmUp             = fullWarmUp;
        param.groupCount             = groupCount;
        param.teamCount              = teamCount;
        param.teamSize               = teamSize;
        param.hardInBalance          = hardInBalance;
        param.hardInBalanceTeam      = hardInBalanceTeam;
        param.hardInBalanceRatio     = inBalanceRatio;
        param.hardInBalanceTeamRatio = inBalanceTeamRatio;

        std::string tmp_path = std::format("{}/{}/index/index_nlist_{}_{}.index",
                                           benchmarks_path, dataset, nlist,
                                           Index::to_string(searchMode));
        index.preSearch(nb, workerCount, blockCount,
                        warmUpSearchList, warmUpSearchListSize, &param, tmp_path);

        if (loop > 1)
            index.search(nq, query.get(), k, distances.get(), labels.get());

        Stopwatch sw;
        Stats stats;
        for (size_t j = 0; j < loop; j++)
            stats = index.search(nq, query.get(), k, distances.get(), labels.get());
        double search_time = sw.elapsedSeconds() / loop;
        index.postSearch();

        float recall = calculate_recall(labels.get(), distances.get(),
                                        ground_truth_I.get(), ground_truth_D.get(), nq, k, metric);
        float r2 = calculate_r2(labels.get(), distances.get(),
                                 ground_truth_I.get(), ground_truth_D.get(), nq, k, metric);
        stats.nprobe     = nprobe;
        stats.query_time = search_time;
        stats.recall     = recall;
        stats.r2         = r2;
        stats.worker     = workerCount;
        stats.nb = nb; stats.nq = nq; stats.d = d; stats.k = k;
        stats.mode = Index::to_string(searchMode);
        stats.print();
        stats.myToCsv(log_path, true, dataset);
    }

    MPI_Finalize();
    cout << CRAN << "return worker:" << rank << RESET << endl;
    return 0;
}