#pragma once

#include "gpugr/gr/CpuRouteTypes.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace gr {

struct CpuRouteSnapshot {
    bool active = false;
    std::vector<int> rawRoute;
    std::vector<std::pair<int, int>> wireDeltas;
    std::vector<int> viaDeltas;
};

class CpuRouteState {
public:
    void resetResources(std::size_t resourceCount);
    void resetRoutes(std::size_t netCount);

    std::vector<int>& wires() { return wires_; }
    const std::vector<int>& wires() const { return wires_; }
    std::vector<int>& vias() { return vias_; }
    const std::vector<int>& vias() const { return vias_; }

    std::vector<std::vector<int>>& rawRoutes() { return rawRoutes_; }
    const std::vector<std::vector<int>>& rawRoutes() const { return rawRoutes_; }

    void setInactiveRoute(int netId, std::vector<int> rawRoute = {0});
    void commit(const CpuRouteCandidate& candidate);
    CpuRouteSnapshot remove(int netId);
    void restore(int netId, const CpuRouteSnapshot& snapshot);
    CpuRouteSnapshot snapshot(int netId) const;

private:
    void validateNetId(int netId) const;
    void validateSnapshot(const CpuRouteSnapshot& snapshot) const;
    void applyDeltas(const CpuRouteSnapshot& snapshot, int direction);

    std::vector<int> wires_;
    std::vector<int> vias_;
    std::vector<std::vector<int>> rawRoutes_;
    std::vector<CpuRouteSnapshot> routeSnapshots_;
};

}  // namespace gr
