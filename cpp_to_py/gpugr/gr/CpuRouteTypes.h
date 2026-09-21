#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace gr {

struct CpuRouteScratch {
    std::vector<int64_t> map;
    std::vector<int> prev;
    std::vector<int> mid;

    void reset(std::size_t stride, int64_t mapValue);
    void resetRegion(int layers,
                     int gridSize,
                     int direction,
                     int xl,
                     int xh,
                     int yl,
                     int yh,
                     int64_t mapValue);
    std::size_t bytes() const;
};

enum class CpuRouteStatus {
    Routed,
    NoRoute,
};

struct CpuRouteCandidate {
    int batchOrdinal = -1;
    int netId = -1;
    CpuRouteStatus status = CpuRouteStatus::NoRoute;
    std::vector<int> rawRoute;
    std::vector<std::pair<int, int>> wireDeltas;
    std::vector<int> viaDeltas;
};

struct CpuRouteRunStats {
    int schemaVersion = 2;
    int requestedWorkers = 1;
    int effectiveWorkers = 1;
    int batchCount = 0;
    int parallelBatchCount = 0;
    int conflictBatchCount = 0;
    int serialFallbackBatchCount = 0;
    int maxBatchSize = 0;
    int batchSizeLimit = 250;
    int batchConflictScanLimit = 0;
    int candidateBatchCount = 0;
    int candidateBatchSizeLimit = 0;
    int packedConflictBatchCount = 0;
    std::size_t scratchBytesPerWorker = 0;
    std::size_t memoryBudgetBytes = 0;
    uint64_t routeHash64 = 0;
    uint64_t wireMapHash64 = 0;
    uint64_t viaMapHash64 = 0;
    std::string workerClampReason = "serial_backend";
    std::string terminalStatus = "not_started";
    bool fluteParallel = false;
    double costMapSeconds = 0.0;
    double batchScheduleSeconds = 0.0;
    double prepareSeconds = 0.0;
    double fluteWaitSeconds = 0.0;
    double fluteWallSeconds = 0.0;
    double candidateSeconds = 0.0;
    double commitSeconds = 0.0;
    double serialFallbackSeconds = 0.0;
    double overflowMarkSeconds = 0.0;
    double totalSeconds = 0.0;
};

}  // namespace gr
