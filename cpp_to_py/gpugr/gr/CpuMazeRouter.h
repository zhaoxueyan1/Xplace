#pragma once

#include "gpugr/gr/CpuRouteTypes.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gr {

struct CpuMazePoint {
    int layer = -1;
    int x = -1;
    int y = -1;
};

struct CpuMazeNet {
    int netId = -1;
    std::vector<std::vector<CpuMazePoint>> pinGroups;
};

enum class CpuMazeFailureReason {
    None,
    InvalidInput,
    Unreachable,
    RouteBuffer,
};

const char* cpuMazeFailureReasonName(CpuMazeFailureReason reason);

// The view borrows all map storage.  A caller must keep the pointed-to vectors
// alive and immutable for the duration of one candidate search batch.
struct CpuMazeSearchView {
    int layerCount = 0;
    int xSize = 0;
    int ySize = 0;
    int direction = 0;
    int routingLayerBegin = 0;
    int routingLayerEnd = -1;
    const std::vector<int>* wireUsage = nullptr;
    const std::vector<int>* viaUsage = nullptr;
    const std::vector<int>* wireCost = nullptr;
    const std::vector<int>* viaCost = nullptr;
    const std::vector<float>* capacity = nullptr;
    const std::vector<float>* fixedUsage = nullptr;
    const std::vector<uint8_t>* blockedWire = nullptr;
    const std::vector<uint8_t>* blockedVia = nullptr;
    int overflowPenalty = 1000000;
    int impassableCost = 10000000;
};

struct CpuMazeSearchResult {
    CpuRouteCandidate candidate;
    CpuMazeFailureReason failureReason = CpuMazeFailureReason::None;
    std::size_t expandedStates = 0;

    bool succeeded() const { return failureReason == CpuMazeFailureReason::None; }
};

struct CpuMazeBatchResult {
    std::vector<CpuMazeSearchResult> results;
    int requestedWorkers = 1;
    int effectiveWorkers = 1;
    std::size_t scratchBytesPerWorker = 0;
    std::size_t memoryBudgetBytes = 0;
    std::string workerClampReason = "serial_backend";
};

class CpuMazeRouter {
public:
    CpuMazeSearchResult route(const CpuMazeNet& net, const CpuMazeSearchView& view) const;
    CpuMazeBatchResult routeBatch(const std::vector<CpuMazeNet>& nets,
                                  const CpuMazeSearchView& view,
                                  int requestedWorkers) const;
};

}  // namespace gr
