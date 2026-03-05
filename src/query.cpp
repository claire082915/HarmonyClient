// query.cpp  (modified for TCP client-server mode)
//
// Rank layout is IDENTICAL to the original:
//   rank 0   = master
//   rank 1..N = workers
//
// Added: --serve flag.  Master opens a TCP socket, accepts one client,
// serves query batches in a loop.  Workers loop on MPI_Bcast signal.
// Nothing in node.cpp / node.h is changed.

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <mpi.h>

// TCP socket headers
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

#include "tribase.h"
#include "utils.h"

using namespace harmony;
using namespace std;

// Signal broadcast master->workers each search iteration
static constexpr int WORKER_SIGNAL_SEARCH   = 1;
static constexpr int WORKER_SIGNAL_SHUTDOWN = 0;

// ---------------------------------------------------------------------------
// Reliable TCP send / recv
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
        if (s <= 0) throw std::runtime_error("TCP recv failed");
        p += s; n -= s;
    }
}

bool str_lower_equal(const std::string& a, const std::string& b) {
    return std::equal(a.begin(), a.end(), b.begin(), b.end(),
                      [](char a, char b) { return std::tolower(a) == std::tolower(b); });
}

int workerMain(int rank, bool cut, Index::SearchMode mode, bool blockSend, bool minorCut) {
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

    // NEW: serve mode arguments
    program.add_argument("--serve")
        .help("Accept TCP client connections and serve queries")
        .default_value(false).implicit_value(true);
    program.add_argument("--nprobe")
        .help("nprobe for serve mode")
        .default_value(100ul)
        .action([](const std::string& v) -> size_t { return std::stoul(v); });
    program.add_argument("--tcp_port")
        .help("TCP port to listen on in serve mode")
        .default_value(7777ul)
        .action([](const std::string& v) -> size_t { return std::stoul(v); });

    try { program.parse_args(argc, argv); }
    catch (const std::runtime_error& err) { std::cerr << err.what() << "\n" << program; return 1; }

    int pro;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &pro);

    int rank, workerCount;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &workerCount);
    workerCount--;  // rank 0 is master — identical to original

    bool serve_mode = program.get<bool>("serve");

    bool disableOrderOptimize = program.get<bool>("disableOrderOpt");
    bool pruning    = !program.get<bool>("disablePruning");
    bool minorCut   = program.get<bool>("minorCut");
    bool run_faiss  = program.get<bool>("run_faiss");
    bool hardInBalance      = program.get<bool>("HardInBalance");
    size_t hardInBalanceTeam = program.get<size_t>("HardInBalanceTeam");
    float inBalanceRatio     = program.get<float>("HardInBalanceRatio");
    float inBalanceTeamRatio = program.get<float>("HardInBalanceTeamRatio");
    float brute_force_search_ratio = program.get<float>("brute_force_search_ratio");
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
            MPI_Barrier(MPI_COMM_WORLD);
            if (!serve_mode) {
                workerMain(rank, pruning, searchMode, blockSend, minorCut);
            } else {
                // In serve mode, init() was already done by preSearch (via MPI_Barrier above).
                // Each iteration: wait for signal, then receiveQuery() + search().
                // receiveQuery() starts with MPI_Barrier — master matches it in the serve loop.
                if (searchMode == Index::SearchMode::DIVIDE_GROUP) {
                    GroupWorker worker;
                    worker.init(rank, blockSend);
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
                    // DIVIDE_IVF: BaseWorker does init+search together, not re-entrant
                    // Fall back to workerMain per iteration
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
    float sub_nprobe_ratio = program.get<float>("sub_nprobe_ratio");
    size_t serve_nprobe    = program.get<size_t>("nprobe");
    uint16_t tcp_port      = static_cast<uint16_t>(program.get<size_t>("tcp_port"));

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

    size_t nb, d;
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

    // Build / load index
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

    // =======================================================================
    // SERVE MODE — TCP socket server
    // =======================================================================
    if (serve_mode) {
        index.nprobe = serve_nprobe;
        if (index.nprobe > nlist) index.nprobe = nlist;

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

        // Open TCP server socket
        int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd < 0) { cerr << "socket() failed\n"; MPI_Finalize(); return 1; }
        int opt = 1;
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port        = htons(tcp_port);
        if (::bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            cerr << "bind() failed\n"; MPI_Finalize(); return 1;
        }
        ::listen(server_fd, 1);

        cout << YELLOW
             << std::format("[Master] TCP server listening on port {}\n", tcp_port)
             << "[Master] Waiting for client...\n"
             << RESET;

        sockaddr_in client_addr{};
        socklen_t   client_len = sizeof(client_addr);
        int client_fd = ::accept(server_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) { cerr << "accept() failed\n"; MPI_Finalize(); return 1; }
        cout << YELLOW << "[Master] Client connected. Entering serve loop.\n" << RESET;

        // Serve loop
        while (true) {
            // 1. Receive header: [nq, k] as two uint64
            uint64_t hdr[2] = {};
            try { recv_all(client_fd, hdr, sizeof(hdr)); }
            catch (...) { cout << "[Master] Client disconnected.\n";
                          int sig = WORKER_SIGNAL_SHUTDOWN;
                          MPI_Bcast(&sig, 1, MPI_INT, 0, MPI_COMM_WORLD);
                          break; }

            size_t nq    = static_cast<size_t>(hdr[0]);
            size_t job_k = static_cast<size_t>(hdr[1]);

            if (nq == 0) {
                cout << "[Master] Shutdown signal received.\n";
                int sig = WORKER_SIGNAL_SHUTDOWN;
                MPI_Bcast(&sig, 1, MPI_INT, 0, MPI_COMM_WORLD);
                break;
            }

            cout << std::format("[Master] Job: nq={} k={}\n", nq, job_k);

            // 2. Receive query vectors from TCP client
            auto query = std::make_unique<float[]>(nq * d);
            recv_all(client_fd, query.get(), nq * d * sizeof(float));

            // 3. Signal workers: SEARCH
            int sig = WORKER_SIGNAL_SEARCH;
            MPI_Bcast(&sig, 1, MPI_INT, 0, MPI_COMM_WORLD);

            // 4. MPI_Barrier — workers call MPI_Barrier at the top of receiveQuery()
            //    master must match it here before index.search() which calls search_group_master()
            MPI_Barrier(MPI_COMM_WORLD);

            // 5. Search (internally does all the MPI Bcasts/Sends/Recvs with workers)
            auto distances = std::make_unique<float[]>(nq * job_k);
            auto labels    = std::make_unique<int64_t[]>(nq * job_k);

            Stopwatch search_timer;
            index.search(nq, query.get(), job_k, distances.get(), labels.get());
            double search_time = search_timer.elapsedSeconds();

            // 6. Recall + CSV logging (same as benchmark mode)
            std::string gt_path = std::format("{}/{}/result/groundtruth_{}.{}",
                                              benchmarks_path, dataset, job_k, output_format);
            if (std::filesystem::exists(gt_path)) {
                auto gt_I = std::make_unique<idx_t[]>(job_k * nq);
                auto gt_D = std::make_unique<float[]>(job_k * nq);
                loadResults(gt_path, gt_I.get(), gt_D.get(), nq, job_k);

                float recall = calculate_recall(labels.get(), distances.get(),
                                                gt_I.get(), gt_D.get(), nq, job_k, metric);
                float r2     = calculate_r2(labels.get(), distances.get(),
                                            gt_I.get(), gt_D.get(), nq, job_k, metric);

                Stats stats;
                stats.nprobe     = serve_nprobe;
                stats.query_time = search_time;
                stats.recall     = recall;
                stats.r2         = r2;
                stats.worker     = workerCount;
                stats.nb = nb; stats.nq = nq; stats.d = d; stats.k = job_k;
                stats.mode = Index::to_string(searchMode);
                stats.print();

                std::string log_path = std::format("{}/{}/result/log_serve.csv",
                                                   benchmarks_path, dataset);
                stats.myToCsv(log_path, true, dataset);
            } else {
                cout << YELLOW
                     << std::format("[Master] No groundtruth at {} — run benchmark mode first\n", gt_path)
                     << RESET;
                cout << std::format("[Master] search time: {:.4f}s\n", search_time);
            }

            // 7. Send results back over TCP
            send_all(client_fd, distances.get(), nq * job_k * sizeof(float));
            send_all(client_fd, labels.get(),    nq * job_k * sizeof(int64_t));

            cout << std::format("[Master] Results sent for nq={}\n", nq);
        }

        ::close(client_fd);
        ::close(server_fd);
        MPI_Finalize();
        return 0;
    }

    // =======================================================================
    // BENCHMARK MODE — original logic unchanged
    // =======================================================================
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
            float recall = calculate_recall(fi.get(), fd.get(), ground_truth_I.get(), ground_truth_D.get(), nq, k, metric);
            float r2 = calculate_r2(fi.get(), fd.get(), ground_truth_I.get(), ground_truth_D.get(), nq, k, metric);
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