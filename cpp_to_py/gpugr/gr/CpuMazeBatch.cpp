#include "CpuMazeBatch.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace gr {
namespace {

std::size_t multiply(std::size_t count, std::size_t bytes) {
    if (bytes == 0) return 0;
    if (count > std::numeric_limits<std::size_t>::max() / bytes)
        throw std::overflow_error("CPU maze memory reservation overflow");
    return count * bytes;
}

std::size_t add(std::size_t lhs, std::size_t rhs) {
    if (lhs > std::numeric_limits<std::size_t>::max() - rhs)
        throw std::overflow_error("CPU maze memory reservation overflow");
    return lhs + rhs;
}

}  // namespace

CpuMazeBatchRouter::CpuMazeBatchRouter(const CpuMazeSearchView& view, int requestedWorkers,
                                     int maxBatchSize)
    : requestedWorkers_(requestedWorkers), maxBatchSize_(maxBatchSize),
      nodeCount_(cpuMazeGridSize(view)),
      // 44 bytes/node for fixed scratch and a bounded heap. Reserve additional
      // space for path/tree/target containers and one in-flight candidate.
      scratchBytes_(multiply(nodeCount_, 192)), reservedBytes_(0) {
    if (requestedWorkers <= 0 || maxBatchSize < 0)
        throw std::invalid_argument("invalid CPU maze batch workers or size");
}

CpuMazeBatchResult CpuMazeBatchRouter::search(const std::vector<CpuMazeNet>& nets,
                                             const CpuMazeSearchView& view,
                                             std::size_t sharedBytes) {
    if (nets.size() > static_cast<std::size_t>(maxBatchSize_) || cpuMazeGridSize(view) != nodeCount_)
        throw std::invalid_argument("CPU maze batch exceeds reserved geometry or net count");
    std::size_t netBytes = nets.capacity() * sizeof(CpuMazeNet), maxNetBytes = 0;
    for (const auto& net : nets) {
        std::size_t bytes = net.pinGroups.capacity() * sizeof(std::vector<CpuMazePoint>) +
                            net.branches.capacity() * sizeof(CpuMazeBranch);
        for (const auto& group : net.pinGroups) bytes += group.capacity() * sizeof(CpuMazePoint);
        for (const auto& branch : net.branches) bytes += branch.targets.capacity() * sizeof(CpuMazePoint);
        netBytes = add(netBytes, bytes);
        maxNetBytes = std::max(maxNetBytes, bytes);
    }
    // Bound retained raw route/delta vectors including their growth capacity.
    const auto pendingBytes = multiply(nodeCount_, multiply(nets.size(), 64));
    reservedBytes_ = add(add(sharedBytes, netBytes), pendingBytes);
    const auto workerBytes = add(scratchBytes_, multiply(maxNetBytes, 2));
    const auto nextLimit = chooseCpuRouteWorkers(requestedWorkers_, maxBatchSize_, workerBytes, reservedBytes_);
    if (!pool_ || nextLimit.effectiveWorkers != limit_.effectiveWorkers) {
        pool_.reset();
        scratches_.resize(nextLimit.effectiveWorkers);
        pool_ = std::make_unique<CpuRouteWorkerPool>(nextLimit.effectiveWorkers);
    }
    limit_ = nextLimit;
    CpuMazeBatchResult result;
    result.requestedWorkers = requestedWorkers_;
    result.effectiveWorkers = limit_.effectiveWorkers;
    result.scratchBytesPerWorker = workerBytes;
    result.memoryBudgetBytes = limit_.memoryBudgetBytes;
    result.workerClampReason = limit_.clampReason;
    result.results.resize(nets.size());
    std::atomic<int> active{0}, maxActive{0};
    std::vector<double> searchSeconds(nets.size(), 0.0);
    pool_->run(nets.size(), [&](int worker, std::size_t index) {
        const int count = active.fetch_add(1) + 1;
        int maximum = maxActive.load();
        while (count > maximum && !maxActive.compare_exchange_weak(maximum, count)) {}
        const auto started = std::chrono::steady_clock::now();
        result.results[index] = CpuMazeRouter{}.route(nets[index], view, scratches_[worker]);
        searchSeconds[index] = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        active.fetch_sub(1);
    });
    result.maxActiveSearches = maxActive.load();
    for (double seconds : searchSeconds) result.workerSearchSeconds += seconds;
    return result;
}

CpuMazeBatchResult CpuMazeRouter::routeBatch(const std::vector<CpuMazeNet>& nets,
                                           const CpuMazeSearchView& view,
                                           int requestedWorkers) const {
    if (nets.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("CPU maze batch net count exceeds INT_MAX");
    return CpuMazeBatchRouter(view, requestedWorkers, static_cast<int>(nets.size())).search(nets, view);
}

}  // namespace gr
