#include "CpuRouteWorkerPool.h"

#include <stdexcept>

namespace gr {

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
