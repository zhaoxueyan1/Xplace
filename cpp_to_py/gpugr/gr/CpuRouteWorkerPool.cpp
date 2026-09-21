#include "CpuRouteWorkerPool.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace gr {

namespace {

std::size_t availableMemoryBytes() {
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    std::size_t kib = 0;
    std::string unit;
    while (meminfo >> key >> kib >> unit) {
        if (key == "MemAvailable:") {
            if (kib > std::numeric_limits<std::size_t>::max() / 1024) {
                throw std::overflow_error("MemAvailable exceeds size_t");
            }
            return kib * 1024;
        }
    }
    return 0;
}

}  // namespace

CpuWorkerLimit chooseCpuRouteWorkers(int requestedWorkers,
                                     int maxBatchSize,
                                     std::size_t scratchBytesPerWorker) {
    if (requestedWorkers <= 0) {
        throw std::invalid_argument("CPU pattern-routing workers must be positive");
    }
    if (maxBatchSize <= 0) {
        return {1, 0, "empty_route_set"};
    }
    if (scratchBytesPerWorker == 0) {
        throw std::invalid_argument("CPU pattern-routing scratch size must be positive");
    }

    const unsigned hardware = std::thread::hardware_concurrency();
    const int hardwareWorkers = hardware == 0 ? requestedWorkers : static_cast<int>(hardware);
    const std::size_t available = availableMemoryBytes();
    if (available == 0 && requestedWorkers > 1) {
        throw std::runtime_error("cannot read /proc/meminfo MemAvailable for CPU route worker budget");
    }
    const std::size_t budget = available / 2;
    const int memoryWorkers = available == 0
                                  ? 1
                                  : static_cast<int>(std::min<std::size_t>(
                                        budget / scratchBytesPerWorker,
                                        static_cast<std::size_t>(std::numeric_limits<int>::max())));
    if (memoryWorkers < 1) {
        throw std::runtime_error("insufficient MemAvailable for one CPU pattern-routing worker");
    }

    const int effective = std::max(
        1, std::min({requestedWorkers, maxBatchSize, hardwareWorkers, memoryWorkers}));
    std::string reason = "none";
    if (effective < requestedWorkers) {
        if (effective == memoryWorkers) {
            reason = "memory_budget";
        } else if (effective == maxBatchSize) {
            reason = "max_batch_size";
        } else {
            reason = "hardware_concurrency";
        }
    }
    return {effective, budget, reason};
}

CpuRouteWorkerPool::CpuRouteWorkerPool(int workerCount) {
    if (workerCount <= 0) {
        throw std::invalid_argument("CPU route worker pool size must be positive");
    }
    workers.reserve(static_cast<std::size_t>(workerCount));
    for (int workerId = 0; workerId < workerCount; ++workerId) {
        workers.emplace_back(&CpuRouteWorkerPool::workerLoop, this, workerId);
    }
}

CpuRouteWorkerPool::~CpuRouteWorkerPool() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stopping = true;
        ++generation;
    }
    workReady.notify_all();
    for (auto& worker : workers) {
        worker.join();
    }
}

void CpuRouteWorkerPool::run(std::size_t count,
                             const std::function<void(int, std::size_t)>& job) {
    if (count == 0) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        currentJob = job;
        jobCount = count;
        nextJob.store(0, std::memory_order_relaxed);
        workerError = nullptr;
        workersRemaining = static_cast<int>(workers.size());
        ++generation;
    }
    workReady.notify_all();

    std::unique_lock<std::mutex> lock(mutex);
    workDone.wait(lock, [&] { return workersRemaining == 0; });
    currentJob = {};
    if (workerError) {
        std::rethrow_exception(workerError);
    }
}

void CpuRouteWorkerPool::workerLoop(int workerId) {
    std::size_t observedGeneration = 0;
    while (true) {
        std::function<void(int, std::size_t)> job;
        std::size_t count = 0;
        {
            std::unique_lock<std::mutex> lock(mutex);
            workReady.wait(lock, [&] { return stopping || generation != observedGeneration; });
            if (stopping) {
                return;
            }
            observedGeneration = generation;
            job = currentJob;
            count = jobCount;
        }

        try {
            while (true) {
                const std::size_t index = nextJob.fetch_add(1, std::memory_order_relaxed);
                if (index >= count) {
                    break;
                }
                job(workerId, index);
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            if (!workerError) {
                workerError = std::current_exception();
            }
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            --workersRemaining;
            if (workersRemaining == 0) {
                workDone.notify_one();
            }
        }
    }
}

}  // namespace gr
