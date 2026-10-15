#pragma once

#include <cstdint>
#include <vector>

namespace gr {

struct CpuRouteBox {
    int netId = -1;
    int xl = 0;
    int xh = 0;
    int yl = 0;
    int yh = 0;
};

struct CpuRouteBatchSchedule {
    std::vector<int> netOrder;
    std::vector<int> batchSizes;
    std::vector<int> conflictFreeNetCounts;
    std::vector<uint8_t> packedConflictBatches;
};

CpuRouteBatchSchedule buildCpuRouteBatchSchedule(std::vector<CpuRouteBox> boxes,
                                                 int gridX,
                                                 int gridY,
                                                 int conflictScanLimit,
                                                 int batchSizeLimit,
                                                 bool packConflictTail);

}  // namespace gr
