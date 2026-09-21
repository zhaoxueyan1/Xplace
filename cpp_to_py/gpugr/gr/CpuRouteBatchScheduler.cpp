#include "CpuRouteBatchScheduler.h"

#include "gpugr/gr/BatchOccupancy.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace gr {

CpuRouteBatchSchedule buildCpuRouteBatchSchedule(std::vector<CpuRouteBox> boxes,
                                                 int gridX,
                                                 int gridY,
                                                 int conflictScanLimit,
                                                 int batchSizeLimit,
                                                 bool packConflictTail) {
    if (gridX <= 0 || gridY <= 0) {
        throw std::invalid_argument("CPU pattern-routing grid dimensions must be positive");
    }
    if (conflictScanLimit < 0) {
        throw std::invalid_argument("CPU pattern-routing conflict scan limit cannot be negative");
    }
    if (batchSizeLimit <= 0) {
        throw std::invalid_argument("CPU pattern-routing batch size limit must be positive");
    }
    for (const auto& box : boxes) {
        if (box.netId < 0 || box.xl < 0 || box.yl < 0 || box.xh < box.xl || box.yh < box.yl ||
            box.xh >= gridX || box.yh >= gridY) {
            throw std::invalid_argument("CPU pattern-routing net bbox is outside the route grid");
        }
    }

    std::sort(boxes.begin(), boxes.end(), [](const CpuRouteBox& lhs, const CpuRouteBox& rhs) {
        const int64_t lhsArea = static_cast<int64_t>(lhs.xh - lhs.xl + 1) * (lhs.yh - lhs.yl + 1);
        const int64_t rhsArea = static_cast<int64_t>(rhs.xh - rhs.xl + 1) * (rhs.yh - rhs.yl + 1);
        return lhsArea == rhsArea ? lhs.netId > rhs.netId : lhsArea > rhsArea;
    });

    CpuRouteBatchSchedule schedule;
    schedule.netOrder.reserve(boxes.size());
    BatchOccupancy occupancy;
    occupancy.reset(gridX, gridY);
    std::size_t firstRemaining = 0;
    while (schedule.netOrder.size() < boxes.size()) {
        const std::size_t batchStart = schedule.netOrder.size();
        int consecutiveConflicts = 0;
        bool hitConflictLimit = false;
        for (std::size_t i = firstRemaining; i < boxes.size(); ++i) {
            auto& box = boxes[i];
            if (box.netId < 0) {
                continue;
            }
            if (occupancy.noConflict(box.xl, box.xh, box.yl, box.yh)) {
                schedule.netOrder.push_back(box.netId);
                occupancy.insert(box.xl, box.xh, box.yl, box.yh);
                box.netId = -1;
                consecutiveConflicts = 0;
            } else {
                ++consecutiveConflicts;
            }
            if (schedule.netOrder.size() - batchStart == static_cast<std::size_t>(batchSizeLimit)) {
                break;
            }
            if (conflictScanLimit > 0 && consecutiveConflicts >= conflictScanLimit) {
                hitConflictLimit = true;
                break;
            }
        }

        const int conflictFreeNetCount =
            static_cast<int>(schedule.netOrder.size() - batchStart);
        bool packedConflictTail = false;
        if (packConflictTail && hitConflictLimit) {
            for (std::size_t i = firstRemaining;
                 i < boxes.size() &&
                 schedule.netOrder.size() - batchStart < static_cast<std::size_t>(batchSizeLimit);
                 ++i) {
                auto& box = boxes[i];
                if (box.netId < 0) {
                    continue;
                }
                schedule.netOrder.push_back(box.netId);
                box.netId = -1;
                packedConflictTail = true;
            }
        }

        while (firstRemaining < boxes.size() && boxes[firstRemaining].netId < 0) {
            ++firstRemaining;
        }
        const int batchSize = static_cast<int>(schedule.netOrder.size() - batchStart);
        if (batchSize <= 0) {
            throw std::runtime_error("CPU pattern-routing scheduler failed to make progress");
        }
        schedule.batchSizes.push_back(batchSize);
        schedule.conflictFreeNetCounts.push_back(conflictFreeNetCount);
        schedule.packedConflictBatches.push_back(packedConflictTail ? uint8_t{1} : uint8_t{0});
        occupancy.reset(gridX, gridY);
    }

    std::reverse(schedule.batchSizes.begin(), schedule.batchSizes.end());
    std::reverse(schedule.conflictFreeNetCounts.begin(), schedule.conflictFreeNetCounts.end());
    std::reverse(schedule.packedConflictBatches.begin(), schedule.packedConflictBatches.end());
    std::reverse(schedule.netOrder.begin(), schedule.netOrder.end());
    return schedule;
}

}  // namespace gr
