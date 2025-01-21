#include "Clustering.h"

#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace harmony {

Clustering::Clustering(size_t d, size_t nlist, bool verbose, const ClusteringParameters& cp)
    : d(d), nlist(nlist), verbose(verbose), cp(cp), centroids(nlist * d, 0.0f) {}

void Clustering::train(size_t n, const float* candidate_codes) {
    float* sampling_codes = nullptr;
    const float* sampled_codes = candidate_codes;
    subsample_training_set(n, candidate_codes, sampling_codes);

    // If subsampling actually allocated memory, update sampled_codes to point to the new sampled data
    if (sampling_codes != nullptr) {
        sampled_codes = sampling_codes;
    }

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    initialize_centroids(n, sampled_codes);

    // End timing
    auto end = std::chrono::high_resolution_clock::now();

    // Compute and output execution time
    std::chrono::duration<double, std::milli> elapsed = end - start;

    for (int iter = 0; iter < cp.niter; ++iter) {
        if (verbose) {
            std::cout << "Iteration " << iter + 1 << " of " << cp.niter << std::endl;
        }
        update_centroids(n, sampled_codes);
    }

    // Note: If sampled_codes was newly allocated, it needs to be freed here
    if (sampling_codes) {
        delete[] sampling_codes;
    }
}

void Clustering::subsample_training_set(size_t& n, const float* candidate_codes, float*& sampling_codes) {
    size_t max_samples = nlist * cp.max_points_per_centroid;
    if (n <= max_samples) {
        // If the number of data points is less than or equal to the maximum sample size, no subsampling is needed
        return;
    }

    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);  // Fill indices

    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(cp.seed));  // Shuffle indices randomly

    sampling_codes = new float[max_samples * d];

#pragma omp parallel for
    for (size_t i = 0; i < max_samples; ++i) {
        std::copy_n(candidate_codes + indices[i] * d, d, sampling_codes + i * d);
    }

    n = max_samples;  // Update the number of data points
}


// Pre
void Clustering::initialize_centroids(size_t n, const float* sampled_codes) {
    if (!sampled_codes) {
        std::cerr << "Error: sampled_codes is a null pointer." << std::endl;
        return;
    }

    // Generate random indices
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);                                       // Fill indices starting from 0
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(cp.seed));  // Shuffle indices using random engine

    // Select initial centroids based on shuffled indices
    centroids.resize(nlist * d);  // Ensure centroids has enough space to store all centers
    for (size_t i = 0; i < nlist; ++i) {
        size_t index = indices[i];                                            // Get the random index
        std::copy_n(sampled_codes + index * d, d, centroids.data() + i * d);  // Copy the selected point as a centroid
    }
}



void Clustering::update_centroids(size_t n, const float* sampled_codes) {
    Eigen::Map<const Eigen::MatrixXf> codes(sampled_codes, d, n);
    Eigen::Map<Eigen::MatrixXf> centers(centroids.data(), d, nlist);

    std::vector<size_t> counts(nlist, 0);
    Eigen::MatrixXf new_centroids = Eigen::MatrixXf::Zero(d, nlist);

    if (cp.metric == MetricType::METRIC_L2) {
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            size_t closest_centroid = 0;
            float min_dist = std::numeric_limits<float>::max();
            for (size_t j = 0; j < nlist; ++j) {
                float dist = (centers.col(j) - codes.col(i)).squaredNorm();
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_centroid = j;
                }
            }

            #pragma omp atomic
            counts[closest_centroid]++;

            #pragma omp critical
            {
                new_centroids.col(closest_centroid) += codes.col(i);
            }
        }
    } else if (cp.metric == MetricType::METRIC_IP) {
        Eigen::MatrixXf normalized_codes = codes.colwise().normalized();

        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            size_t closest_centroid = 0;
            float max_ip = std::numeric_limits<float>::lowest();
            for (size_t j = 0; j < nlist; ++j) {
                float ip = normalized_codes.col(i).dot(centers.col(j));
                if (ip > max_ip) {
                    max_ip = ip;
                    closest_centroid = j;
                }
            }

            #pragma omp atomic
            counts[closest_centroid]++;

            #pragma omp critical
            {
                new_centroids.col(closest_centroid) += codes.col(i); 
            }
        }
    }

    for (size_t i = 0; i < nlist; ++i) {
        if (counts[i] > 0) {
            centers.col(i) = new_centroids.col(i) / counts[i];
        }
    }

}


void Clustering::apply_centroid_perturbations() {
    for (size_t i = 0; i < nlist; ++i) {
        for (size_t j = 0; j < d; ++j) {
            // For centroids with odd indices (starting from 0, so this checks for even indices)
            if (i % 2 == 0) {
                centroids[i * d + j] -= 1e-6;
            } else {  // For centroids with even indices
                centroids[i * d + j] += 1e-6;
            }
        }
    }
}


float* Clustering::get_centroids() const {
    float* centroid_codes = new float[nlist * d];
    std::memcpy(centroid_codes, centroids.data(), sizeof(float) * nlist * d);
    return centroid_codes;
}

void Clustering::get_centroids(float* centroid_codes) const {
    std::memcpy(centroid_codes, centroids.data(), sizeof(float) * nlist * d);
}

}  // namespace tribase