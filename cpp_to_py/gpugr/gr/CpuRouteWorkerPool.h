#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace gr {

struct CpuWorkerLimit {
    int effectiveWorkers = 1;
    std::size_t memoryBudgetBytes = 0;
    std::string clampReason = "none";
};

CpuWorkerLimit chooseCpuRouteWorkers(int requestedWorkers,
                                     int maxBatchSize,
                                     std::size_t scratchBytesPerWorker,
                                     std::size_t reservedBytes = 0);
CpuWorkerLimit limitCpuRouteWorkers(int requestedWorkers, int maxBatchSize,
                                    std::size_t scratchBytesPerWorker,
                                    std::size_t reservedBytes, std::size_t memoryBudgetBytes,
                                    int hardwareWorkers);

class CpuRouteWorkerPool {
public:
    explicit CpuRouteWorkerPool(int workerCount);
    ~CpuRouteWorkerPool();

    CpuRouteWorkerPool(const CpuRouteWorkerPool&) = delete;
    CpuRouteWorkerPool& operator=(const CpuRouteWorkerPool&) = delete;

    void run(std::size_t jobCount, const std::function<void(int, std::size_t)>& job);

private:
    void workerLoop(int workerId);

    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable workReady;
    std::condition_variable workDone;
    std::function<void(int, std::size_t)> currentJob;
    std::atomic<std::size_t> nextJob{0};
    std::size_t jobCount = 0;
    std::size_t generation = 0;
    int workersRemaining = 0;
    bool stopping = false;
    std::exception_ptr workerError;
};

}  // namespace gr
