#pragma once

#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>

class SIFTChunkReaderFloat {
public:
    SIFTChunkReaderFloat(const std::string& path,
                         size_t batch_size,
                         size_t max_nb = 0)
        : batch_size_(batch_size),
          max_nb_(max_nb),
          path_(path),
          dim_(0),
          eof_(false),
          read_count_(0)
    {
        file_.open(path_, std::ios::binary);
        if (!file_.is_open())
            throw std::runtime_error("Cannot open file: " + path_);
    }

    bool has_next() const {
        return !eof_;
    }

    size_t get_dim() const {
        return dim_;
    }

    std::vector<float> next() {
        if (eof_) return {};

        std::vector<float> out;
        out.reserve(batch_size_ * dim_);

        size_t count = 0;

        while (count < batch_size_ && !eof_) {

            // 🚨 HARD STOP based on config limit
            if (max_nb_ > 0 && read_count_ >= max_nb_) {
                eof_ = true;
                break;
            }

            int32_t d = 0;

            if (!file_.read(reinterpret_cast<char*>(&d), sizeof(int32_t))) {
                eof_ = true;
                break;
            }

            if (dim_ == 0) {
                dim_ = static_cast<size_t>(d);
            } else if (static_cast<size_t>(d) != dim_) {
                throw std::runtime_error("Inconsistent dimension in bvecs file");
            }

            std::vector<uint8_t> buf(dim_);

            if (!file_.read(reinterpret_cast<char*>(buf.data()), dim_)) {
                eof_ = true;
                break;
            }

            for (size_t i = 0; i < dim_; ++i) {
                out.push_back(static_cast<float>(buf[i]));
            }

            count++;
            read_count_++;
        }

        return out;
    }

private:
    std::ifstream file_;
    size_t batch_size_;
    size_t max_nb_;
    std::string path_;
    size_t dim_;
    bool eof_;
    size_t read_count_;
};
