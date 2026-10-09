#include "CpuRouteState.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace gr {

namespace {

void validateDeltaRange(int start, int span, std::size_t size, const char* kind) {
    if (start < 0 || span <= 0 || static_cast<std::size_t>(start) >= size ||
        static_cast<std::size_t>(span) > size - static_cast<std::size_t>(start)) {
        throw std::runtime_error(std::string("CPU route state ") + kind +
                                 " delta outside resource range");
    }
}

}  // namespace

void CpuRouteState::resetResources(std::size_t resourceCount) {
    wires_.assign(resourceCount, 0);
    vias_.assign(resourceCount, 0);
}

void CpuRouteState::resetRoutes(std::size_t netCount) {
    routeSnapshots_.assign(netCount, CpuRouteSnapshot{});
}

void CpuRouteState::validateNetId(int netId) const {
    if (netId < 0 || static_cast<std::size_t>(netId) >= routeSnapshots_.size()) {
        throw std::runtime_error("CPU route state net id outside route table");
    }
}

void CpuRouteState::validateSnapshot(const CpuRouteSnapshot& snapshot) const {
    if (!snapshot.active) {
        if (!snapshot.wireDeltas.empty() || !snapshot.viaDeltas.empty() ||
            snapshot.rawRoute.size() > 1 ||
            (!snapshot.rawRoute.empty() && snapshot.rawRoute[0] > 1)) {
            throw std::runtime_error("CPU route state inactive route has resource deltas");
        }
    } else {
        const auto& raw = snapshot.rawRoute;
        if (raw.empty() || raw.size() % 2 != 1 || raw[0] != static_cast<int>(raw.size())) {
            throw std::runtime_error("CPU route state malformed raw route");
        }
        std::size_t wire = 0, via = 0;
        for (std::size_t index = 1; index + 1 < raw.size(); index += 2) {
            if (raw[index + 1] == -1) {
                if (via >= snapshot.viaDeltas.size() || snapshot.viaDeltas[via++] != raw[index]) {
                    throw std::runtime_error("CPU route state raw via and delta mismatch");
                }
            } else if (raw[index + 1] > 0) {
                if (wire >= snapshot.wireDeltas.size() ||
                    snapshot.wireDeltas[wire++] != std::make_pair(raw[index], raw[index + 1])) {
                    throw std::runtime_error("CPU route state raw wire and delta mismatch");
                }
            } else {
                throw std::runtime_error("CPU route state invalid raw span");
            }
        }
        if (wire != snapshot.wireDeltas.size() || via != snapshot.viaDeltas.size()) {
            throw std::runtime_error("CPU route state extra resource deltas");
        }
    }
    for (const auto& [start, span] : snapshot.wireDeltas) {
        validateDeltaRange(start, span, wires_.size(), "wire");
    }
    for (int encoded : snapshot.viaDeltas) {
        if (encoded < 0 || static_cast<std::size_t>(encoded) >= vias_.size()) {
            throw std::runtime_error("CPU route state via delta outside resource range");
        }
    }
}

void CpuRouteState::applyDeltas(const CpuRouteSnapshot& snapshot, int direction) {
    if (direction != 1 && direction != -1) {
        throw std::invalid_argument("CPU route state delta direction must be one or minus one");
    }
    validateSnapshot(snapshot);
    if (direction < 0) {
        std::unordered_map<int, int> removalCounts;
        for (const auto& [start, span] : snapshot.wireDeltas) {
            for (int offset = 0; offset < span; ++offset) {
                int& usage = wires_[static_cast<std::size_t>(start + offset)];
                if (usage < ++removalCounts[start + offset]) {
                    throw std::runtime_error("CPU route state wire usage underflow");
                }
            }
        }
        removalCounts.clear();
        for (int encoded : snapshot.viaDeltas) {
            if (vias_[static_cast<std::size_t>(encoded)] < ++removalCounts[encoded]) {
                throw std::runtime_error("CPU route state via usage underflow");
            }
        }
    }
    for (const auto& [start, span] : snapshot.wireDeltas) {
        for (int offset = 0; offset < span; ++offset) {
            wires_[static_cast<std::size_t>(start + offset)] += direction;
        }
    }
    for (int encoded : snapshot.viaDeltas) {
        vias_[static_cast<std::size_t>(encoded)] += direction;
    }
}

void CpuRouteState::setInactiveRoute(int netId, std::vector<int> rawRoute) {
    validateNetId(netId);
    if (routeSnapshots_[static_cast<std::size_t>(netId)].active) {
        throw std::runtime_error("CPU route state cannot replace an active route");
    }
    CpuRouteSnapshot snapshot;
    snapshot.rawRoute = std::move(rawRoute);
    validateSnapshot(snapshot);
    routeSnapshots_[static_cast<std::size_t>(netId)] = std::move(snapshot);
}

void CpuRouteState::commit(const CpuRouteCandidate& candidate) {
    validateNetId(candidate.netId);
    const std::size_t netId = static_cast<std::size_t>(candidate.netId);
    if (routeSnapshots_[netId].active) {
        throw std::runtime_error("CPU route state cannot commit over an active route");
    }
    CpuRouteSnapshot snapshot;
    snapshot.active = candidate.status == CpuRouteStatus::Routed;
    snapshot.rawRoute = candidate.rawRoute;
    snapshot.wireDeltas = candidate.wireDeltas;
    snapshot.viaDeltas = candidate.viaDeltas;
    validateSnapshot(snapshot);
    applyDeltas(snapshot, 1);
    routeSnapshots_[netId] = std::move(snapshot);
}

CpuRouteSnapshot CpuRouteState::remove(int netId) {
    validateNetId(netId);
    const std::size_t index = static_cast<std::size_t>(netId);
    // Validate and subtract before moving the canonical route out of the table.
    const auto& current = routeSnapshots_[index];
    if (!current.active) {
        throw std::runtime_error("CPU route state cannot remove an inactive route");
    }
    CpuRouteSnapshot inactive;
    inactive.rawRoute = {1};
    applyDeltas(current, -1);
    CpuRouteSnapshot old = std::move(routeSnapshots_[index]);
    routeSnapshots_[index] = std::move(inactive);
    return old;
}

void CpuRouteState::restore(int netId, const CpuRouteSnapshot& snapshot) {
    validateNetId(netId);
    const std::size_t index = static_cast<std::size_t>(netId);
    if (routeSnapshots_[index].active) {
        throw std::runtime_error("CPU route state cannot restore over an active route");
    }
    auto replacement = snapshot;
    validateSnapshot(replacement);
    applyDeltas(replacement, 1);
    routeSnapshots_[index] = std::move(replacement);
}

CpuRouteSnapshot CpuRouteState::snapshot(int netId) const {
    validateNetId(netId);
    return routeSnapshots_[static_cast<std::size_t>(netId)];
}

CpuRouteCheckpoint CpuRouteState::checkpoint() const {
    return {routeSnapshots_};
}

void CpuRouteState::restoreCheckpoint(const CpuRouteCheckpoint& checkpoint) {
    if (checkpoint.routes.size() != routeSnapshots_.size()) {
        throw std::runtime_error("CPU route state checkpoint net count mismatch");
    }
    std::vector<int> rebuiltWires(wires_.size(), 0);
    std::vector<int> rebuiltVias(vias_.size(), 0);
    for (const CpuRouteSnapshot& snapshot : checkpoint.routes) {
        validateSnapshot(snapshot);
        if (!snapshot.active) {
            continue;
        }
        for (const auto& [start, span] : snapshot.wireDeltas) {
            for (int offset = 0; offset < span; ++offset) {
                ++rebuiltWires[static_cast<std::size_t>(start + offset)];
            }
        }
        for (int encoded : snapshot.viaDeltas) {
            ++rebuiltVias[static_cast<std::size_t>(encoded)];
        }
    }
    // Allocate the replacement route table before publishing any rebuilt map.
    auto rebuiltRoutes = checkpoint.routes;
    wires_.swap(rebuiltWires);
    vias_.swap(rebuiltVias);
    routeSnapshots_.swap(rebuiltRoutes);
}

const std::vector<int>& CpuRouteState::rawRoute(int netId) const {
    validateNetId(netId);
    return routeSnapshots_[static_cast<std::size_t>(netId)].rawRoute;
}

std::vector<std::vector<int>> CpuRouteState::rawRoutes() const {
    std::vector<std::vector<int>> result;
    result.reserve(routeSnapshots_.size());
    for (const auto& route : routeSnapshots_) result.push_back(route.rawRoute);
    return result;
}

std::size_t cpuRouteSnapshotsBytes(const std::vector<CpuRouteSnapshot>& routes) {
    std::size_t result = routes.capacity() * sizeof(CpuRouteSnapshot);
    for (const auto& route : routes) {
        result += route.rawRoute.capacity() * sizeof(int) +
                  route.wireDeltas.capacity() * sizeof(std::pair<int, int>) +
                  route.viaDeltas.capacity() * sizeof(int);
    }
    return result;
}

std::size_t CpuRouteState::bytes() const {
    return (wires_.capacity() + vias_.capacity()) * sizeof(int) +
           cpuRouteSnapshotsBytes(routeSnapshots_);
}

}  // namespace gr
