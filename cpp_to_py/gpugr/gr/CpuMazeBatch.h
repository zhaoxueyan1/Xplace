#pragma once

#include "CpuMazeRouter.h"
#include "CpuRouteWorkerPool.h"

#include <memory>

namespace gr {

// Logical batches do not depend on worker count or memory-based worker clamps.
constexpr int CPU_MAZE_BATCH_SIZE = 32;

class CpuMazeBatchRouter {
public:
    CpuMazeBatchRouter(const CpuMazeSearchView& view, int requestedWorkers,
                       int maxBatchSize);
    CpuMazeBatchResult search(const std::vector<CpuMazeNet>& nets, const CpuMazeSearchView& view,
                              std::size_t sharedBytes = 0);
    std::size_t reservedBytes() const { return reservedBytes_; }

private:
    int requestedWorkers_;
    int maxBatchSize_;
    std::size_t nodeCount_;
    std::size_t scratchBytes_;
    std::size_t reservedBytes_;
    CpuWorkerLimit limit_;
    std::unique_ptr<CpuRouteWorkerPool> pool_;
    std::vector<CpuMazeSearchScratch> scratches_;
};

}  // namespace gr
