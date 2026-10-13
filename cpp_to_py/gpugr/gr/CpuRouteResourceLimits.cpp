#include "CpuRouteWorkerPool.h"

#include "common/utils/log.h"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sched.h>
#include <stdexcept>

namespace gr {
namespace {

std::size_t routeMemoryBudget(std::size_t availableBytes) {
    // Keep the 80% policy exact without converting large byte counts through
    // floating point or overflowing a multiplication by four.
    return availableBytes / 5 * 4 + (availableBytes % 5) * 4 / 5;
}

std::size_t availableMemoryBytes() {
    std::ifstream meminfo("/proc/meminfo");
    std::string key, unit;
    std::size_t kib = 0, available = 0;
    while (meminfo >> key >> kib >> unit) {
        if (key == "MemAvailable:") {
            if (kib > std::numeric_limits<std::size_t>::max() / 1024)
                throw std::overflow_error("MemAvailable exceeds size_t");
            available = kib * 1024;
            break;
        }
    }
    // A container's cgroup limit can be much lower than host MemAvailable.
    // Inspect the process cgroup and its ancestors, including the mount root.
    const std::filesystem::path root("/sys/fs/cgroup");
    auto path = root;
    std::ifstream groups("/proc/self/cgroup");
    std::string line;
    while (std::getline(groups, line)) {
        if (line.rfind("0::/", 0) != 0) continue;
        const auto relative = std::filesystem::path(line.substr(4)).lexically_normal();
        if (!relative.empty() && *relative.begin() != ".." &&
            std::filesystem::exists(root / relative)) path = root / relative;
    }
    while (true) {
        std::size_t maximum = 0, current = 0;
        std::ifstream maxFile(path / "memory.max"), currentFile(path / "memory.current");
        if (maxFile >> maximum && currentFile >> current) {
            const std::size_t remaining = maximum > current ? maximum - current : 0;
            available = std::min(available, remaining);
        }
        if (path == root) break;
        path = path.parent_path();
    }
    return available;
}

int availableCpus(int requestedWorkers) {
    cpu_set_t affinity;
    if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0)
        return std::max(1, CPU_COUNT(&affinity));
    const auto hardware = std::thread::hardware_concurrency();
    return hardware == 0 ? requestedWorkers : static_cast<int>(hardware);
}

}  // namespace

CpuWorkerLimit limitCpuRouteWorkers(int requestedWorkers, int maxBatchSize,
                                    std::size_t scratchBytesPerWorker,
                                    std::size_t reservedBytes, std::size_t budget,
                                    int hardwareWorkers) {
    if (requestedWorkers <= 0 || scratchBytesPerWorker == 0 || hardwareWorkers <= 0)
        throw std::invalid_argument("CPU routing workers and scratch must be positive");
    if (maxBatchSize <= 0) return {1, budget, "empty_route_set"};
    if (reservedBytes > budget || scratchBytesPerWorker > budget - reservedBytes)
        throw std::runtime_error("insufficient CPU routing memory budget for shared state and one worker");
    const int memoryWorkers = static_cast<int>(std::min<std::size_t>(
        (budget - reservedBytes) / scratchBytesPerWorker,
        static_cast<std::size_t>(std::numeric_limits<int>::max())));
    const int effective = std::min({requestedWorkers, maxBatchSize, hardwareWorkers, memoryWorkers});
    std::string reason = "none";
    if (effective < requestedWorkers) {
        if (effective == memoryWorkers) reason = "memory_budget";
        else if (effective == maxBatchSize) reason = "max_batch_size";
        else reason = "cpu_affinity";
    }
    return {effective, budget, reason};
}

CpuWorkerLimit chooseCpuRouteWorkers(int requestedWorkers, int maxBatchSize,
                                     std::size_t scratchBytesPerWorker,
                                     std::size_t reservedBytes) {
    const auto available = availableMemoryBytes();
    const auto budget = routeMemoryBudget(available);
    const auto hardware = availableCpus(requestedWorkers);
    const bool guard = maxBatchSize > 0 &&
                       (reservedBytes > budget || scratchBytesPerWorker > budget - reservedBytes);
    static std::atomic<bool> budgetLogged{false};
    if (guard) {
        utils::logger.warning(
            "CPU routing memory guard: available=%zu budget=%zu reserved=%zu worker=%zu "
            "requested_workers=%d max_batch=%d hardware_workers=%d policy=0.80",
            available,
            budget,
            reservedBytes,
            scratchBytesPerWorker,
            requestedWorkers,
            maxBatchSize,
            hardware);
    }
    auto result = limitCpuRouteWorkers(requestedWorkers, maxBatchSize, scratchBytesPerWorker,
                                       reservedBytes, budget, hardware);
    if (reservedBytes > 0 && !budgetLogged.exchange(true)) {
        utils::logger.info(
            "CPU routing memory budget: available=%zu budget=%zu reserved=%zu worker=%zu "
            "requested_workers=%d effective_workers=%d max_batch=%d hardware_workers=%d "
            "clamp=%s policy=0.80",
            available,
            budget,
            reservedBytes,
            scratchBytesPerWorker,
            requestedWorkers,
            result.effectiveWorkers,
            maxBatchSize,
            hardware,
            result.clampReason.c_str());
    }
    return result;
}

}  // namespace gr
