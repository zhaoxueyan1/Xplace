#pragma once

#include "gpugr/gr/CpuRouteTypes.h"
#include "gpugr/gr/CpuMazeFrontier.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gr {

class CpuMazeCoarseGrid;

struct CpuMazePoint {
    int layer = -1;
    int x = -1;
    int y = -1;
};

struct CpuMazeBranch {
    std::vector<CpuMazePoint> targets;
    bool pinAccess = false;
};

struct CpuMazeNet {
    int netId = -1;
    std::vector<std::vector<CpuMazePoint>> pinGroups;
    std::vector<CpuMazeBranch> branches{};
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
    // Lower bounds remain valid as commit only increases congestion costs.
    int horizontalWireLowerBound = 1;
    int verticalWireLowerBound = 1;
    int viaLowerBound = 1;
    // Optional, immutable batch snapshot used only to guide fine-grid search.
    const CpuMazeCoarseGrid* coarseGrid = nullptr;
};

struct CpuMazeSearchStats {
    std::size_t coarseExpandedStates = 0;
    std::size_t fineExpandedStates = 0;
    int corridorRetries = 0;
    int fullSearchFallbacks = 0;
};

struct CpuMazeSearchResult {
    CpuRouteCandidate candidate;
    CpuMazeFailureReason failureReason = CpuMazeFailureReason::None;
    std::size_t expandedStates = 0;
    CpuMazeSearchStats searchStats;

    bool succeeded() const { return failureReason == CpuMazeFailureReason::None; }
};

// One instance per worker, also reused by deterministic serial re-search.
// Reset only visited nodes between branches rather than clearing the chip.
struct CpuMazePathScratch {
    std::vector<int64_t> distances;
    std::vector<int> previous;
    std::vector<int> touched;
    CpuMazeFrontier frontier;
    std::size_t bytes() const;

    void reset(std::size_t nodeCount);
};

struct CpuMazeSearchScratch {
    CpuMazePathScratch fine;
    CpuMazePathScratch coarse;
    std::size_t bytes() const;
};

struct CpuMazeBatchResult {
    std::vector<CpuMazeSearchResult> results;
    int requestedWorkers = 1;
    int effectiveWorkers = 1;
    std::size_t scratchBytesPerWorker = 0;
    std::size_t memoryBudgetBytes = 0;
    std::string workerClampReason = "serial_backend";
    int maxActiveSearches = 0;
    double workerSearchSeconds = 0.0;
};

std::size_t cpuMazeGridSize(const CpuMazeSearchView& view);

class CpuMazeRouter {
public:
    CpuMazeSearchResult route(const CpuMazeNet& net, const CpuMazeSearchView& view) const;
    CpuMazeSearchResult route(const CpuMazeNet& net,
                             const CpuMazeSearchView& view,
                             CpuMazeSearchScratch& scratch) const;
    CpuMazeBatchResult routeBatch(const std::vector<CpuMazeNet>& nets,
                                  const CpuMazeSearchView& view,
                                  int requestedWorkers) const;
};

}  // namespace gr
