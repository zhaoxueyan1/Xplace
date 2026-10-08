#include "CpuRouteTypes.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace gr {

namespace {

std::size_t checkedMultiply(std::size_t lhs, std::size_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::overflow_error("CPU pattern-routing scratch size overflow");
    }
    return lhs * rhs;
}

}  // namespace

void CpuRouteScratch::reset(std::size_t stride, int64_t mapValue) {
    map.assign(checkedMultiply(6, stride), mapValue);
    prev.assign(checkedMultiply(14, stride), 0);
    mid.assign(stride, 0);
}

void CpuRouteScratch::resetRegion(int layers,
                                  int gridSize,
                                  int direction,
                                  int xl,
                                  int xh,
                                  int yl,
                                  int yh,
                                  int64_t mapValue) {
    if (layers <= 0 || gridSize <= 0 || xl < 0 || yl < 0 || xh < xl || yh < yl ||
        xh >= gridSize || yh >= gridSize) {
        throw std::invalid_argument("invalid CPU pattern-routing scratch region");
    }
    const std::size_t stride = checkedMultiply(
        static_cast<std::size_t>(layers),
        checkedMultiply(static_cast<std::size_t>(gridSize), static_cast<std::size_t>(gridSize)));
    const std::size_t mapSize = checkedMultiply(6, stride);
    const std::size_t prevSize = checkedMultiply(14, stride);
    if (map.size() != mapSize) {
        map.resize(mapSize);
    }
    if (prev.size() != prevSize) {
        prev.resize(prevSize);
    }
    if (mid.size() != stride) {
        mid.resize(stride);
    }

    for (int layer = 0; layer < layers; ++layer) {
        for (int x = xl; x <= xh; ++x) {
            for (int y = yl; y <= yh; ++y) {
                const int mappedX = ((layer & 1) ^ direction) ? x : y;
                const int mappedY = ((layer & 1) ^ direction) ? y : x;
                const std::size_t encoded = static_cast<std::size_t>(layer) * gridSize * gridSize +
                                            static_cast<std::size_t>(mappedX) * gridSize + mappedY;
                for (std::size_t plane = 0; plane < 6; ++plane) {
                    map[plane * stride + encoded] = mapValue;
                }
                for (std::size_t plane = 0; plane < 14; ++plane) {
                    prev[plane * stride + encoded] = 0;
                }
                mid[encoded] = 0;
            }
        }
    }

    for (int x = xl; x <= xh; ++x) {
        for (int y = yl; y <= yh; ++y) {
            const std::size_t location = static_cast<std::size_t>(x) * gridSize + y;
            map[5 * stride + location] = mapValue;
        }
    }
}

std::size_t CpuRouteScratch::bytes() const {
    return checkedMultiply(map.capacity(), sizeof(map.front())) +
           checkedMultiply(prev.capacity(), sizeof(prev.front())) +
           checkedMultiply(mid.capacity(), sizeof(mid.front()));
}

static void hashWord(uint64_t value, uint64_t& hash) {
    constexpr uint64_t FNV_PRIME = 1099511628211ULL;
    for (int byte = 0; byte < 8; ++byte) {
        hash ^= (value >> (byte * 8)) & 0xffULL;
        hash *= FNV_PRIME;
    }
}

uint64_t hashIntVector(const std::vector<int>& values) {
    uint64_t hash = 1469598103934665603ULL;
    hashWord(values.size(), hash);
    for (int value : values) {
        hashWord(static_cast<uint64_t>(static_cast<int64_t>(value)), hash);
    }
    return hash;
}

uint64_t hashRawRoutes(const std::vector<std::vector<int>>& routes) {
    uint64_t hash = 1469598103934665603ULL;
    hashWord(routes.size(), hash);
    for (const auto& route : routes) {
        hashWord(route.size(), hash);
        for (int value : route) {
            hashWord(static_cast<uint64_t>(static_cast<int64_t>(value)), hash);
        }
    }
    return hash;
}


}  // namespace gr
