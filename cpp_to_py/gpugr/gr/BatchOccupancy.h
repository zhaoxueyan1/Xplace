#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace gr {

class BatchOccupancy {
public:
    void reset(int x_size, int y_size) {
        xSize = std::max(0, x_size);
        ySize = std::max(0, y_size);
        wordsPerRow = (ySize + WORD_BITS - 1) / WORD_BITS;
        const std::size_t wordCount = static_cast<std::size_t>(xSize) * wordsPerRow;
        if (bits.size() != wordCount) {
            bits.assign(wordCount, 0);
        } else {
            std::fill(bits.begin(), bits.end(), 0);
        }
    }

    bool noConflict(int xl, int xr, int yl, int yr) const {
        if (!normalizeBox(xl, xr, yl, yr)) {
            return true;
        }
        const int firstWord = yl / WORD_BITS;
        const int lastWord = yr / WORD_BITS;
        for (int x = xl; x <= xr; ++x) {
            const std::size_t rowOffset = static_cast<std::size_t>(x) * wordsPerRow;
            for (int word = firstWord; word <= lastWord; ++word) {
                const uint64_t mask = makeWordMask(word, firstWord, lastWord, yl, yr);
                if ((bits[rowOffset + word] & mask) != 0) {
                    return false;
                }
            }
        }
        return true;
    }

    void insert(int xl, int xr, int yl, int yr) {
        update(xl, xr, yl, yr, true);
    }

    void remove(int xl, int xr, int yl, int yr) {
        update(xl, xr, yl, yr, false);
    }

private:
    static constexpr int WORD_BITS = 64;
    int xSize = 0;
    int ySize = 0;
    int wordsPerRow = 0;
    std::vector<uint64_t> bits;

    bool normalizeBox(int& xl, int& xr, int& yl, int& yr) const {
        if (xSize <= 0 || ySize <= 0) {
            return false;
        }
        xl = std::max(0, xl);
        yl = std::max(0, yl);
        xr = std::min(xr, xSize - 1);
        yr = std::min(yr, ySize - 1);
        return xl <= xr && yl <= yr;
    }

    static uint64_t lowMask(int bit) {
        if (bit <= 0) {
            return ~uint64_t{0};
        }
        if (bit >= WORD_BITS) {
            return uint64_t{0};
        }
        return ~uint64_t{0} << bit;
    }

    static uint64_t highMask(int bit) {
        if (bit >= WORD_BITS - 1) {
            return ~uint64_t{0};
        }
        if (bit < 0) {
            return uint64_t{0};
        }
        return (uint64_t{1} << (bit + 1)) - 1;
    }

    static uint64_t makeWordMask(int word, int firstWord, int lastWord, int yl, int yr) {
        uint64_t mask = ~uint64_t{0};
        if (word == firstWord) {
            mask &= lowMask(yl % WORD_BITS);
        }
        if (word == lastWord) {
            mask &= highMask(yr % WORD_BITS);
        }
        return mask;
    }

    void update(int xl, int xr, int yl, int yr, bool occupied) {
        if (!normalizeBox(xl, xr, yl, yr)) {
            return;
        }
        const int firstWord = yl / WORD_BITS;
        const int lastWord = yr / WORD_BITS;
        for (int x = xl; x <= xr; ++x) {
            const std::size_t rowOffset = static_cast<std::size_t>(x) * wordsPerRow;
            for (int word = firstWord; word <= lastWord; ++word) {
                const uint64_t mask = makeWordMask(word, firstWord, lastWord, yl, yr);
                if (occupied) {
                    bits[rowOffset + word] |= mask;
                } else {
                    bits[rowOffset + word] &= ~mask;
                }
            }
        }
    }
};

}  // namespace gr
