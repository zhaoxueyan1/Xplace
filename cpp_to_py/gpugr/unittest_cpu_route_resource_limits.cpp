#include "gpugr/gr/CpuRouteWorkerPool.h"

#include <cassert>
#include <stdexcept>

int main() {
    // Shared snapshots/candidates consume budget before workers are selected.
    auto limit = gr::limitCpuRouteWorkers(8, 32, 100, 700, 1000, 16);
    assert(limit.effectiveWorkers == 3 && limit.clampReason == "memory_budget");
    limit = gr::limitCpuRouteWorkers(8, 32, 100, 0, 1000, 2);
    assert(limit.effectiveWorkers == 2 && limit.clampReason == "cpu_affinity");
    bool rejected = false;
    try { gr::limitCpuRouteWorkers(8, 32, 100, 901, 1000, 16); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    return 0;
}
