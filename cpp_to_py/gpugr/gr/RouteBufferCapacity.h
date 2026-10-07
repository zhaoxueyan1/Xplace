#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gr {

// A failed maze traceback is discarded and the complete PR/RRR sequence is
// restarted.  Growing by only 2x made large designs repeatedly discover a
// different short net whose initial 128-record-per-pin reserve was still too
// small, turning one recoverable overflow into many full-chip reruns.
constexpr int kRouteBufferGrowthFactor = 4;

class RouteCapacityExceeded : public std::runtime_error {
public:
    explicit RouteCapacityExceeded(std::vector<int> nets)
        : std::runtime_error("GPU route record capacity exceeded"), netIds(std::move(nets)) {}

    const std::vector<int> netIds;
};

// Keep other nets' reserves unchanged. Maze kernels still use int offsets.
inline std::vector<int64_t> grownRouteOffsets(const int64_t* offsets, int netCount,
                                             const std::vector<int>& netIds) {
    std::vector<bool> grow(netCount, false);
    for (int net : netIds) grow.at(net) = true;
    std::vector<int64_t> result(netCount + 1, 0);
    for (int net = 0; net < netCount; ++net) {
        const int64_t capacity = offsets[net + 1] - offsets[net];
        const int64_t limit = std::numeric_limits<int>::max() - result[net];
        const int multiplier = grow[net] ? kRouteBufferGrowthFactor : 1;
        if (capacity <= 0 || capacity > limit / multiplier) {
            throw std::runtime_error("GPU route buffer exceeds int-indexable capacity");
        }
        result[net + 1] = result[net] + capacity * multiplier;
    }
    return result;
}

}  // namespace gr
