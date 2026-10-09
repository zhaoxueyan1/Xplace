#include "CPURouter.h"

#include <algorithm>

namespace gr {

void CPURouter::markOverflowNets(const std::vector<GrNet>& nets) {
    overflowReport = OverflowReport{};
    auto& qor = overflowReport.qor;
    std::fill(overflowWire.begin(), overflowWire.end(), 0);
    std::fill(overflowVia.begin(), overflowVia.end(), 0);
    for (int idx = 0; idx < LAYER * N * N; ++idx) {
        if (idx % N + 1 >= N) continue;
        const float excess = static_cast<float>(routeState.wires()[idx]) + fixed[idx] +
                             twoCellsViaUsage(idx) - capacity[idx];
        if (excess > 0) {
            overflowWire[idx] = 1;
            ++qor.overflowWireResourceCount;
            qor.overflowWireExcess += excess;
        }
    }
    for (int idx = 0; idx < (LAYER - 1) * N * N; ++idx) {
        const int layer = idx / N / N + 1;
        const int y = idx % (N * N) / N, x = idx % N;
        const int upper = layer * N * N + x * N + y;
        const float excess = std::max(inCellUsedArea(idx) - capacity[idx],
                                      inCellUsedArea(upper) - capacity[upper]);
        if (excess > 0) {
            overflowVia[idx] = 1;
            ++qor.overflowViaResourceCount;
            qor.overflowViaExcess += excess;
        }
    }
    for (int netId = 0; netId < static_cast<int>(nets.size()); ++netId) {
        const auto& route = routeState.routes()[netId];
        const auto& raw = route.rawRoute;
        // Preserve the PR API: missing/failed routes count as overflow.
        if (raw.empty() || raw[0] < 0) ++overflowReport.legacyOverflowNetCount;
        if (nets[netId].noroute || nets[netId].exclusion != NetExclusion::None) continue;
        if (!route.active) {
            if (raw.empty() || raw[0] < 0) ++qor.failedRouteCount;
            else ++qor.unroutedNetCount;
            continue;
        }
        ++qor.routedNetCount;
        bool overflow = false;
        for (const auto& [start, span] : route.wireDeltas) {
            qor.wireLength += span;
            for (int resource = start; resource < start + span; ++resource) {
                overflow = overflow || overflowWire[resource] != 0;
            }
        }
        qor.viaCount += static_cast<int>(route.viaDeltas.size());
        for (int resource : route.viaDeltas) overflow = overflow || overflowVia[resource] != 0;
        if (overflow) {
            ++overflowReport.legacyOverflowNetCount;
            overflowReport.netIds.push_back(netId);
        }
    }
    qor.overflowNetCount = static_cast<int>(overflowReport.netIds.size());
    qor.routeHash64 = hashRawRoutes(routeState.routes());
    qor.wireMapHash64 = hashIntVector(routeState.wires());
    qor.viaMapHash64 = hashIntVector(routeState.vias());
    numOvflNets = overflowReport.legacyOverflowNetCount;
}

}  // namespace gr
