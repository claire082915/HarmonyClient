#pragma once

#include <mkl.h>
// #include <mkl_cblas.h>
#include <immintrin.h>  
#include <algorithm>    
#include <cassert>
#include <chrono>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>
// #include "avx512.h"
#include "common.h"
#include "faiss/faiss/utils/distances.h"
#include "platform_macros.h"
#include <random>
#include <cmath>

namespace harmony {
    const std::string BLUE = "\033[1;34m"; // Blue text
    const std::string YELLOW = "\033[1;33m"; // Blue text
    const std::string GREEN = "\033[1;32m"; // Blue text
    const std::string CRAN = "\033[1;36m"; // Blue text
    const std::string MAG = "\033[1;35m"; // Blue text
    const std::string RED = "\033[1;31m"; // Blue text
    const std::string RESET = "\033[0m"; // Reset color


// void copyPartialVector(const float* const src, float* dest, size_t id_start, size_t ) {

// }

class CsvWriter {
private:
    std::ofstream file_;
    std::string filename_;
    std::vector<std::string> headers_;
    bool hasData_ = false;
    bool isAppend_ = false;
    bool echo_ = false;

public:
    CsvWriter() = default;
    CsvWriter(const std::string& filename, bool isAppend = true, bool echo = true)
        : filename_(filename), isAppend_(isAppend), echo_(echo) {
        if (isAppend && std::filesystem::exists(filename)) {
            file_.open(filename, std::ios::app);
        } else {
            file_.open(filename);
        }
        if (!file_.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
    }
    CsvWriter(const std::string& filename, const std::vector<std::string>& headers, bool isAppend = true,
              bool echo = true)
        : filename_(filename), headers_(headers), isAppend_(isAppend), echo_(echo) {
        if (isAppend && std::filesystem::exists(filename)) {
            file_.open(filename, std::ios::app);
            if (!file_.is_open()) {
                throw std::runtime_error("Failed to open file: " + filename);
            }
        } else {
            file_.open(filename);
            if (!file_.is_open()) {
                throw std::runtime_error("Failed to open file: " + filename);
            }
            for (size_t i = 0; i < headers.size(); ++i) {
                file_ << headers[i];
                if (i < headers.size() - 1) {
                    file_ << ",";
                }
            }
            file_ << std::endl;
        }
    }
    CsvWriter(const CsvWriter&) = delete;
    CsvWriter& operator=(const CsvWriter&) = delete;
    ~CsvWriter() { file_.close(); }
    void setHeader(const std::vector<std::string>& headers) {
        headers_ = headers;
        for (size_t i = 0; i < headers.size(); ++i) {
            file_ << headers[i];
            if (i < headers.size() - 1) {
                file_ << ",";
            }
        }
        file_ << std::endl;
    }
    std::ofstream& getFile() { return file_; }
    template <typename T>
    CsvWriter& operator<<(const T& value) {
        if (file_) {
            if (hasData_) {
                file_ << ",";
                if (echo_) {
                    std::cout << ",";
                }
            } else {
                if (echo_) {
                    std::cout << filename_ << ":\t";
                }
            }
            if constexpr (std::is_floating_point<T>::value) {
                file_ << std::fixed << std::setprecision(std::numeric_limits<T>::digits10 + 1) << value;
                if (echo_) {
                    std::cout << std::fixed << std::setprecision(std::numeric_limits<T>::digits10 + 1) << value;
                }
            } else {
                file_ << value;
                if (echo_) {
                    std::cout << value;
                }
            }
            hasData_ = true;  
        }
        return *this;
    }
    CsvWriter& operator<<(std::ostream& (*manip)(std::ostream&)) {
        if (manip == static_cast<std::ostream& (*)(std::ostream&)>(std::endl)) {
            if (file_ && hasData_) {
                file_ << std::endl;
                hasData_ = false;  
                if (echo_) {
                    std::cout << std::endl;
                }
            }
        }
        return *this;
    }
};
template <typename T>
inline void copy_n_partial_vector(const T* from, T* to, size_t dFrom, size_t dTo, size_t offset, size_t n) {
    for (size_t vectorIndex = 0; vectorIndex < n; vectorIndex++) {
        std::copy_n(from + vectorIndex * dFrom + offset, dTo, to + vectorIndex * dTo);
    }
}

template <typename T>
inline void add_n(const T* src1, const T* src2, T* dest, size_t n) {
    size_t nt = omp_get_max_threads();
#pragma omp parallel for num_threads(nt)
    for (size_t i = 0; i < n; i++) {
        dest[i] = src1[i] + src2[i];
    }
}
template <typename T>
inline bool diffVector(const T* const v1, const T* const v2, size_t d) {
    using namespace std;
    for (size_t i = 0; i < d; i++) {
        if (v1[i] != v2[i]) {
            return true;
        }
    }
    return false;
}
template <typename T>
inline void printVector(std::vector<T>& v, std::string color, std::string str = "") {
    using namespace std;
    std::cout << color << str << "[";
    for (size_t i = 0; i < v.size(); i++) {
        cout << v[i] << " ";
    }
    cout << "]";
    cout << endl;
    cout << RESET;
}
template <typename T>
inline void printVector(const T* const from, size_t d, std::string color, std::string str = "") {
    using namespace std;
    std::cout << color << str << "[";
    for (size_t i = 0; i < d; i++) {
        cout << from[i] << " ";
    }
    cout << "]";
    cout << endl;
    cout << RESET;
}
inline std::pair<size_t, int> loadFvecsInfo(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return {};
    }

    int d;
    file.read(reinterpret_cast<char*>(&d), sizeof(int));

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    size_t n = fileSize / (4 + d * 4);  

    return {n, d};
}

inline std::tuple<std::unique_ptr<float[]>, size_t, int> loadFvecs(const std::string& filePath,
                                                                   std::pair<int, int> bounds = {1, 0}) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return {};
    }

    int d;
    file.read(reinterpret_cast<char*>(&d), sizeof(int));

    int vecSizeof = 4 + d * 4;  // int + d * float

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    size_t bmax = fileSize / vecSizeof;

    size_t a = bounds.first;
    size_t b = (bounds.second == 0) ? bmax : bounds.second;

    assert(a >= 1 && b <= bmax && b >= a);

    size_t n = b - a + 1;
    std::unique_ptr<float[]> vectors = std::make_unique<float[]>(n * d);

    file.seekg((a - 1) * vecSizeof, std::ios::beg);

    for (size_t i = 0; i < n; ++i) {
        file.seekg(4, std::ios::cur);
        file.read(reinterpret_cast<char*>(vectors.get() + i * d), d * sizeof(float));
    }

    return std::make_tuple(std::move(vectors), n, d);
}
inline std::pair<size_t, int> loadTvecsInfo(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return {};
    }

    size_t m = 0;
    int d = 0;

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        std::getline(ss, value, ',');

        int current_d = 0;
        while (std::getline(ss, value, ',')) {
            current_d++;
        }

        if (m == 0) {
            d = current_d;
        }

        m++;
    }
    return {m, d};
}

inline std::pair<size_t, int> loadBvecsInfo(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return {};
    }

    int d;
    file.read(reinterpret_cast<char*>(&d), sizeof(int));

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    size_t n = fileSize / (4 + d);  

    return {n, d};
}

inline std::tuple<std::unique_ptr<uint8_t[]>, size_t, int> loadBvecs(const std::string& filePath,
                                                                     std::pair<int, int> bounds = {1, 0}) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return {};
    }

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    int d;
    file.read(reinterpret_cast<char*>(&d), sizeof(int));
    int vecSizeof = 4 + d;  // int + d * uint8_t

    size_t bmax = (fileSize - 4) / vecSizeof;

    size_t a = bounds.first;
    size_t b = (bounds.second == 0) ? bmax : bounds.second;

    assert(a >= 1 && b <= bmax && b >= a);

    size_t n = b - a + 1;
    std::unique_ptr<uint8_t[]> vectors = std::make_unique<uint8_t[]>(n * d);

    file.seekg(4 + (a - 1) * vecSizeof, std::ios::beg);

    for (size_t i = 0; i < n; ++i) {
        file.seekg(4, std::ios::cur);
        file.read(reinterpret_cast<char*>(vectors.get() + i * d), d * sizeof(uint8_t));
    }

    return std::make_tuple(std::move(vectors), n, d);
}

inline std::tuple<std::unique_ptr<float[]>, size_t, int> loadBvecs2Fvecs(const std::string& filePath,
                                                                         std::pair<int, int> bounds = {1, 0}) {
    auto [vectors, n, d] = loadBvecs(filePath, bounds);
    std::unique_ptr<float[]> vectors2 = std::make_unique<float[]>(n * d);
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d; j++) {
            vectors2[i * d + j] = static_cast<float>(vectors[i * d + j]);
        }
    }
    return std::make_tuple(std::move(vectors2), n, d);
}

inline std::pair<size_t, int> loadXvecsInfo(const std::string& filePath) {
    if (filePath.ends_with(".fvecs")) {
        return loadFvecsInfo(filePath);
    } else if (filePath.ends_with(".bvecs")) {
        return loadBvecsInfo(filePath);
    } else if (filePath.ends_with(".txt")) {
        return loadTvecsInfo(filePath);
    }

    throw std::runtime_error("no support file");
}
inline std::tuple<std::unique_ptr<float[]>, size_t, int> loadTvecs(const std::string& filePath,
                                                                   std::pair<int, int> bounds = {1, 0}) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return {};
    }

    size_t m = 0;
    int d = 0;
    std::vector<std::vector<float>> data;

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<float> row;

        std::getline(ss, value, ',');

        while (std::getline(ss, value, ',')) {
            row.push_back(std::stof(value));
        }

        if (m == 0) {
            d = row.size();
        }

        data.push_back(row);
        m++;
    }

    size_t a = bounds.first;
    size_t b = (bounds.second == 0) ? m : bounds.second;

    assert(a >= 1 && b <= m && b >= a);

    size_t n = b - a + 1;

    std::unique_ptr<float[]> float_vectors = std::make_unique<float[]>(n * d);

    for (size_t i = a - 1; i < b; ++i) {
        for (int j = 0; j < d; ++j) {
            float_vectors[(i - (a - 1)) * d + j] = data[i][j];
        }
    }

    return std::make_tuple(std::move(float_vectors), n, d);
}

inline std::tuple<std::unique_ptr<float[]>, size_t, int> loadXvecs(const std::string& filePath,
                                                                   std::pair<int, int> bounds = {1, 0}) {
    if (filePath.ends_with(".fvecs")) {
        return loadFvecs(filePath, bounds);
    } else if (filePath.ends_with(".bvecs")) {
        return loadBvecs2Fvecs(filePath, bounds);
    } else if (filePath.ends_with(".txt")) {
        return loadTvecs(filePath);
    }

    throw std::runtime_error("no support file");
}

// A class for measuring execution time
class Stopwatch {
public:
    // Constructor initializes the start time
    Stopwatch() : start(std::chrono::high_resolution_clock::now()) {}

    // Resets the start time to the current time
    inline void reset() { start = std::chrono::high_resolution_clock::now(); }

    // Returns the elapsed time in milliseconds since the stopwatch was started or last reset
    inline double elapsedMilliseconds(bool isReset = false) {
        auto end = std::chrono::high_resolution_clock::now();
        auto ret = std::chrono::duration<double, std::milli>(end - start).count();
        if (isReset) {
            reset();
        }
        return ret;
    }

    inline double elapsedSeconds(bool isReset = false) {
        auto end = std::chrono::high_resolution_clock::now();
        auto ret = std::chrono::duration<double>(end - start).count();
        if (isReset) {
            reset();
        }
        return ret;
    }

private:
    // The start time
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};
class MyStopWatch {
public:
    Stopwatch watch;
    bool shouldPrint;
    std::string name;
    std::string color;
    MyStopWatch(bool shouldPrint = true, std::string name = "MyStopWatch", std::string color = GREEN)
        : shouldPrint(shouldPrint), name(name), color(color) {
        watch.reset();
    }

    void print(std::string s, bool reset = true) {
        static bool printSwitch = false;
        if (!shouldPrint || !printSwitch) {
            return;
        }
        double time = watch.elapsedSeconds(reset);
        std::string star = "";
        if (time > 1) {
            star = "***";
        }
        std::cout << color << format("[{}:{:>30}]:{:.4f}s  {}", name, s, time, star) << RESET << std::endl;
    }
    void reset() { watch.reset(); }
};



#ifdef DEBUG
__attribute__((optimize("O0"))) inline float calculatedEuclideanDistance(const float* vec1, const float* vec2,
                                                                         size_t size) {
    float distance = 0.0;
    for (size_t i = 0; i < size; ++i) {
        float diff = vec1[i] - vec2[i];
        distance += diff * diff;
    }
    return distance;
}
#else
inline float calculatedEuclideanDistance(const float* vec1, const float* vec2, size_t size) {
    float distance = 0.0;
    for (size_t i = 0; i < size; ++i) {
        float diff = vec1[i] - vec2[i];
        distance += diff * diff;
    }
    return distance;
}
#endif

__attribute__((optimize("O0"))) inline float calculatedEuclideanDistance0(const float* vec1, const float* vec2,
                                                                          size_t size) {
    float distance = 0.0;
    for (size_t i = 0; i < size; ++i) {
        float diff = vec1[i] - vec2[i];
        distance += diff * diff;
    }
    return distance;
}



#ifdef DEBUG
__attribute__((optimize("O0"))) inline float calculatedInnerProduct(const float* vec1, const float* vec2, size_t size) {
    // return cblas_sdot(size, vec1, 1, vec2, 1);
    float sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum += vec1[i] * vec2[i];
    }
    return sum;
}
#else
inline float calculatedInnerProduct(const float* vec1, const float* vec2, size_t size) {
    // return cblas_sdot(size, vec1, 1, vec2, 1);
    float sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum += vec1[i] * vec2[i];
    }
    return sum;
}
#endif

__attribute__((optimize("O0"))) inline float calculatedInnerProduct0(const float* vec1, const float* vec2,
                                                                     size_t size) {
    // return cblas_sdot(size, vec1, 1, vec2, 1);
    float sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum += vec1[i] * vec2[i];
    }
    return sum;
}

// Calculates the magnitude (length) of a vector
inline float vectorMagnitude(const float* vec, size_t size) {
    float sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum += vec[i] * vec[i];
    }
    return sqrt(sum);
}

// Calculates the cosine similarity between two vectors
inline float calculateCosineSimilarity(const float* vec1, const float* vec2, size_t size) {
    float dotProduct = 0.0;
    for (size_t i = 0; i < size; ++i) {
        dotProduct += vec1[i] * vec2[i];
    }

    float magnitude1 = vectorMagnitude(vec1, size);
    float magnitude2 = vectorMagnitude(vec2, size);

    if (magnitude1 == 0 || magnitude2 == 0) {
        throw std::invalid_argument("One or both vectors are zero vectors.");
    }

    return dotProduct / (magnitude1 * magnitude2);
}

inline void prepareDirectory(const std::string& filePath) {
    std::filesystem::path path(filePath);
    if (!std::filesystem::exists(path.parent_path())) {
        std::filesystem::create_directories(path.parent_path());
    }
}

inline void writeResultsToFile(const idx_t* labels, const float* distances, size_t nq, size_t k, std::string filePath) {
    prepareDirectory(filePath);

    if (filePath.ends_with("txt")) {
        std::ofstream outFile(filePath);
        if (!outFile.is_open()) {
            std::cerr << std::format("Failed to open file: {}", filePath) << std::endl;
            return;
        }
        for (size_t i = 0; i < nq; ++i) {
            for (size_t j = 0; j < k; ++j) {
                outFile << labels[i * k + j] << " " << std::fixed << std::setprecision(6) << distances[i * k + j];
                if (j < k - 1) {
                    outFile << " ";
                }
            }
            outFile << "\n";
        }
        outFile.close();
    } else {
        std::ofstream outFile(filePath, std::ios::binary);
        if (!outFile.is_open()) {
            std::cerr << std::format("Failed to open file: {}", filePath) << std::endl;
            return;
        }
        outFile.write(reinterpret_cast<const char*>(labels), nq * k * sizeof(idx_t));
        outFile.write(reinterpret_cast<const char*>(distances), nq * k * sizeof(float));
        outFile.close();
    }
}

inline void loadResults(const std::string& filePath, idx_t* labels, float* distances, size_t nq, size_t k) {
    if (filePath.ends_with("txt")) {
        std::ifstream inFile(filePath);
        if (!inFile.is_open()) {
            std::cerr << std::format("Failed to open file: {}", filePath) << std::endl;
            return;
        }
        for (size_t i = 0; i < nq; ++i) {
            for (size_t j = 0; j < k; ++j) {
                inFile >> labels[i * k + j] >> distances[i * k + j];
            }
        }
        inFile.close();
        // throw std::invalid_argument("Reading txt files is not supported.");
    } else {
        std::ifstream inFile(filePath, std::ios::binary);
        if (!inFile.is_open()) {
            std::cerr << std::format("Failed to open file: {}", filePath) << std::endl;
            return;
        }
        inFile.read(reinterpret_cast<char*>(labels), nq * k * sizeof(idx_t));
        inFile.read(reinterpret_cast<char*>(distances), nq * k * sizeof(float));
        inFile.close();
    }
}

inline float relative_error(float x, float y) {
    if (x == 0) {
        return std::abs(y);
    } else {
        return std::abs((x - y) / x);
    }
}

#define FEPS 1e-4
inline float calculate_recall_loose(const idx_t* I, const float* D, const idx_t* GT, const float* GD, size_t nq, size_t k, MetricType metric, size_t gt_k = 0) {
    if(D[0] < 0) {
        std::cout << RED << "negative Distance " << RESET << std::endl;
        return 0;
    }
    if (gt_k == 0) {
        gt_k = k;
    }
    size_t true_correct = 0;
    size_t correct = 0;
    if (k > gt_k) {
        throw std::invalid_argument("k should be less than or equal to gt_k.");
    }
#pragma omp parallel for reduction(+ : true_correct, correct)
    for (size_t i = 0; i < nq; ++i) {
        float maxDis = GD[i * k + k - 1];
        std::unordered_set<idx_t> groundtruth(GT + i * gt_k, GT + i * gt_k + k);
        for (size_t j = 0; j < k; ++j) {
            if (I[i * k + j] == -1) {
                break;
            }
            if (groundtruth.find(I[i * k + j]) != groundtruth.end() || D[i * k + j] < maxDis || abs(D[i * k + j] - maxDis) <= FEPS) {
                true_correct++;
            }
        }
    }
    return static_cast<float>(true_correct) / (nq * k);
}

#define FEPS 1e-4
inline float calculate_recall(const idx_t* I, const float* D, const idx_t* GT, const float* GD, size_t nq, size_t k, MetricType metric, size_t gt_k = 0) {
    if(D[0] < 0) {
        std::cout << RED << "negative Distance " << RESET << std::endl;
        return 0;
    }
    if (gt_k == 0) {
        gt_k = k;
    }
    size_t true_correct = 0;
    size_t correct = 0;
    if (k > gt_k) {
        throw std::invalid_argument("k should be less than or equal to gt_k.");
    }
#pragma omp parallel for reduction(+ : true_correct, correct)
    for (size_t i = 0; i < nq; ++i) {
        std::unordered_set<idx_t> groundtruth(GT + i * gt_k, GT + i * gt_k + k);
        for (size_t j = 0; j < k; ++j) {
            if (I[i * k + j] == -1) {
                break;
            }
            if (groundtruth.find(I[i * k + j]) != groundtruth.end()) {
                true_correct++;
            }
        }
    }
    // std::cout << correct << " "<< true_correct <<std::endl;
    if (metric == MetricType::METRIC_L2) {
        // #pragma omp parallel for reduction(+ : correct)
        for (size_t i = 0; i < nq; ++i) {
            float topK = std::numeric_limits<float>::max();
            size_t ii = k - 1;
            while (GT[i * gt_k + ii] == -1) {
                ii--;
            }
            topK = GD[i * gt_k + ii];
            for (size_t j = 0; j < k; ++j) {
                if (I[i * k + j] == -1) {
                    break;
                }
                if (D[i * k + j] <= topK || relative_error(D[i * k + j], topK) < FEPS) {
                    correct++;
                } else {
                    // std::cerr << std::format("D[{}, {}]= {} > topK= {}", i, j, D[i * k + j], topK) << std::endl;
                    // assert(false);
                }
            }
        }
    } else {
#pragma omp parallel for reduction(+ : correct)
        for (size_t i = 0; i < nq; ++i) {
            float topK = std::numeric_limits<float>::lowest();
            size_t ii = k - 1;
            while (GT[i * gt_k + ii] == -1) {
                ii--;
            }
            topK = GD[i * gt_k + ii];
            for (size_t j = 0; j < k; ++j) {
                if (I[i * k + j] == -1) {
                    break;
                }
                if (D[i * k + j] >= topK || relative_error(D[i * k + j], topK) < FEPS) {
                    correct++;
                } else {
                    // std::cerr << std::format("D[{}, {}]= {} < topK= {}", i, j, D[i * k + j], topK) << std::endl;
                    // assert(false);
                }
            }
        }
    }
    assert(1.0 * true_correct / correct > 0.99);
    return static_cast<float>(true_correct) / (nq * k);
}

inline float calculate_r2(const idx_t* I, const float* D, const idx_t* GT, const float* GD, size_t nq, size_t k,
                          MetricType metric, size_t gt_k = 0) {
    if (gt_k == 0) {
        gt_k = k;
    }
    size_t true_correct = 0;
    size_t correct = 0;
    if (k > gt_k) {
        throw std::invalid_argument("k should be less than or equal to gt_k.");
    }
    float g_sum = 0;
    float sum = 0;
    for (size_t i = 0; i < nq; i++) {
        for (size_t j = 0; j < k; j++) {
            if (I[i * k + j] == -1) {
                break;
            }
            g_sum += GD[i * gt_k + j];
            sum += D[i * k + j];
        }
    }
    return sum / g_sum - 1;
}

inline void output_codes(const float* code, size_t d) {
    if (code) {
        for (size_t i = 0; i < d; ++i) {
            std::cerr << code[i] << ",";
        }
    }
    std::cerr << std::endl;
}

inline std::vector<int> distribute_jobs(int total_jobs, int num_workers, float uneven_factor) {
    std::vector<int> jobs(num_workers, 0);

    if (uneven_factor == 0.0f) {
        // Perfectly even distribution
        int even_jobs = total_jobs / num_workers;
        for (int i = 0; i < num_workers; ++i) {
            jobs[i] = even_jobs;
        }
        jobs[0] += total_jobs % num_workers;
    } else if (uneven_factor == 1.0f) {
        // Completely uneven distribution (one worker gets all jobs)
        jobs[0] = total_jobs;
    } else {
        // Intermediate uneven distribution
        // Use control factor to skew the distribution
        int remaining_jobs = total_jobs;

        // Generate a random number distribution where jobs are skewed based on control_factor
        // std::random_device rd;
        // std::mt19937 gen(rd());
        // std::uniform_real_distribution<> dis(0.0, 1.0);

        // First pass: randomly assign some jobs based on the control factor
        // for (int i = 0; i < num_workers; ++i) {
        //     // Determine the skew factor based on control_factor
        //     float skew = dis(gen) * control_factor;
        //     int job_share = static_cast<int>(total_jobs * skew);
        //     jobs[i] = job_share;
        //     remaining_jobs -= job_share;
        // }
        jobs[0] = uneven_factor * total_jobs;
        remaining_jobs -= jobs[0];

        // Second pass: distribute remaining jobs evenly
        for (int i = 0; i < num_workers; ++i) {
            jobs[i] += remaining_jobs / num_workers;
        }
        jobs[0] += remaining_jobs % num_workers;
    }

    return jobs;
}

}  // namespace harmony