#include "gpugr/gr/CpuRouteWorkerPool.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <stdexcept>
#include <thread>
#include <vector>

int main() {
    gr::CpuRouteWorkerPool pool(4);
    std::vector<int> results(1000, 0);
    std::atomic<int> active{0};
    std::atomic<int> maxActive{0};

    pool.run(results.size(), [&](int workerId, std::size_t index) {
        const int nowActive = active.fetch_add(1) + 1;
        int observed = maxActive.load();
        while (nowActive > observed && !maxActive.compare_exchange_weak(observed, nowActive)) {
        }
        if (index < 4) {
            while (active.load() < 4) {
                std::this_thread::yield();
            }
        }
        results[index] = workerId + 1;
        active.fetch_sub(1);
    });
    assert(std::all_of(results.begin(), results.end(), [](int value) { return value > 0; }));
    assert(maxActive.load() > 1);

    bool sawException = false;
    try {
        pool.run(32, [](int, std::size_t index) {
            if (index == 7) {
                throw std::runtime_error("injected worker failure");
            }
        });
    } catch (const std::runtime_error&) {
        sawException = true;
    }
    assert(sawException);

    std::atomic<int> completed{0};
    pool.run(64, [&](int, std::size_t) { completed.fetch_add(1); });
    assert(completed.load() == 64);
    return 0;
}
