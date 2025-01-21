#pragma once

#include <immintrin.h>
#include <cmath>
#include <cstddef>

namespace harmony {
// Simplified 512-bit vector class, only handles floating-point numbers
class simd512f {
public:
    __m512 data;

    simd512f() : data(_mm512_setzero_ps()) {}

    explicit simd512f(float val) : data(_mm512_set1_ps(val)) {}

    explicit simd512f(const float* ptr) : data(_mm512_loadu_ps(ptr)) {}

    // Load unaligned data
    void loadu(const float* ptr) {
        data = _mm512_loadu_ps(ptr);
    }

    // Store unaligned data
    void storeu(float* ptr) const {
        _mm512_storeu_ps(ptr, data);
    }

    // Calculate the squared difference between two vectors and accumulate
    simd512f& accumulate_square_diff(const simd512f& other) {
        data = _mm512_fmadd_ps(_mm512_sub_ps(data, other.data), _mm512_sub_ps(data, other.data), data);
        return *this;
    }

    // Horizontal addition, sums all elements in the vector
    float horizontal_sum() const {
        __m256 low = _mm512_castps512_ps256(data);
        __m256 high = _mm512_extractf32x8_ps(data, 1);
        __m256 sum256 = _mm256_add_ps(low, high);
        __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256), _mm256_extractf128_ps(sum256, 1));
        sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 0x55));
        return _mm_cvtss_f32(sum128);
    }
};




} // namespace simple