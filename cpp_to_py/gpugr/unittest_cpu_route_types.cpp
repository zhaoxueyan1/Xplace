#include "gpugr/gr/CpuRouteTypes.h"

#include <cassert>
#include <limits>
#include <stdexcept>

namespace {

template <typename Exception, typename Fn>
void assertThrows(Fn&& fn) {
    bool thrown = false;
    try {
        fn();
    } catch (const Exception&) {
        thrown = true;
    }
    assert(thrown);
}

}  // namespace

int main() {
    gr::CpuRouteScratch scratch;
    scratch.reset(100, -7);
    assert(scratch.map.size() == 600);
    assert(scratch.prev.size() == 1400);
    assert(scratch.mid.size() == 100);
    assert(scratch.map.front() == -7 && scratch.map.back() == -7);
    assert(scratch.bytes() >=
           600 * sizeof(int64_t) + 1400 * sizeof(int) + 100 * sizeof(int));

    // resetRegion reuses capacity when the shape is unchanged.
    scratch.resetRegion(2, 10, 0, 0, 9, 0, 9, -3);
    const std::size_t stride = 2 * 10 * 10;
    assert(scratch.map.size() == 6 * stride);
    assert(scratch.prev.size() == 14 * stride);
    assert(scratch.mid.size() == stride);
    const auto* mapData = scratch.map.data();
    scratch.resetRegion(2, 10, 1, 2, 3, 4, 5, 0);
    assert(scratch.map.data() == mapData);

    // Invalid regions are rejected before any resize.
    assertThrows<std::invalid_argument>(
        [&] { scratch.resetRegion(0, 10, 0, 0, 1, 0, 1, 0); });
    assertThrows<std::invalid_argument>(
        [&] { scratch.resetRegion(2, 10, 0, 0, 10, 0, 1, 0); });
    assertThrows<std::invalid_argument>(
        [&] { scratch.resetRegion(2, 10, 0, 5, 4, 0, 1, 0); });

    // Scratch sizing must fail on overflow instead of wrapping.
    const std::size_t hugeStride = std::numeric_limits<std::size_t>::max() / 2;
    assertThrows<std::overflow_error>([&] { scratch.reset(hugeStride, 0); });
    const int hugeGrid = std::numeric_limits<int>::max();
    assertThrows<std::overflow_error>(
        [&] { scratch.resetRegion(4, hugeGrid, 0, 0, hugeGrid - 1, 0, hugeGrid - 1, 0); });

    // Candidates carry an explicit identity and default to NoRoute.
    gr::CpuRouteCandidate candidate;
    assert(candidate.batchOrdinal == -1);
    assert(candidate.netId == -1);
    assert(candidate.status == gr::CpuRouteStatus::NoRoute);
    assert(candidate.rawRoute.empty() && candidate.wireDeltas.empty() && candidate.viaDeltas.empty());

    // Run stats expose the deterministic defaults used by the serial backend.
    gr::CpuRouteRunStats stats;
    assert(stats.schemaVersion == 2);
    assert(stats.requestedWorkers == 1 && stats.effectiveWorkers == 1);
    assert(stats.terminalStatus == "not_started");
    assert(!stats.fluteParallel);
    return 0;
}
