#include "gpugr/gr/CpuRouteBatchScheduler.h"

#include <algorithm>
#include <cassert>
#include <utility>
#include <vector>

namespace {

std::vector<gr::CpuRouteBox> overlappingBoxes(int count) {
    std::vector<gr::CpuRouteBox> boxes;
    for (int netId = 0; netId < count; ++netId) {
        boxes.push_back({netId, 0, 7, 0, 7});
    }
    return boxes;
}

void assertEveryNetScheduled(const gr::CpuRouteBatchSchedule& schedule, int count) {
    std::vector<int> ids = schedule.netOrder;
    std::sort(ids.begin(), ids.end());
    assert(ids.size() == static_cast<std::size_t>(count));
    for (int netId = 0; netId < count; ++netId) {
        assert(ids[static_cast<std::size_t>(netId)] == netId);
    }
}

}  // namespace

int main() {
    const auto serial =
        gr::buildCpuRouteBatchSchedule(overlappingBoxes(6), 8, 8, 0, 4, false);
    assert(serial.batchSizes == std::vector<int>({1, 1, 1, 1, 1, 1}));
    assert(serial.conflictFreeNetCounts == std::vector<int>({1, 1, 1, 1, 1, 1}));
    assert(std::none_of(serial.packedConflictBatches.begin(),
                        serial.packedConflictBatches.end(),
                        [](uint8_t value) { return value != 0; }));
    assertEveryNetScheduled(serial, 6);

    const auto packed =
        gr::buildCpuRouteBatchSchedule(overlappingBoxes(6), 8, 8, 1, 4, true);
    assert(packed.batchSizes == std::vector<int>({2, 4}));
    assert(packed.conflictFreeNetCounts == std::vector<int>({1, 1}));
    assert(packed.packedConflictBatches == std::vector<uint8_t>({1, 1}));
    assertEveryNetScheduled(packed, 6);

    std::vector<gr::CpuRouteBox> disjoint = {
        {0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1},
        {2, 2, 2, 2, 2},
    };
    const auto parallel =
        gr::buildCpuRouteBatchSchedule(std::move(disjoint), 8, 8, 1, 4, true);
    assert(parallel.batchSizes == std::vector<int>({3}));
    assert(parallel.conflictFreeNetCounts == std::vector<int>({3}));
    assert(parallel.packedConflictBatches == std::vector<uint8_t>({0}));
    assertEveryNetScheduled(parallel, 3);
    return 0;
}
