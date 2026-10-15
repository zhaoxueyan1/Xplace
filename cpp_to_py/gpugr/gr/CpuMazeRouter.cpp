#include "CpuMazeRouter.h"
#include "CpuMazeCoarseGrid.h"
#include "CpuMazeSearch.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

namespace gr {

namespace {

using namespace cpu_maze;

void appendWireDelta(CpuRouteCandidate& candidate, int start, int span, int layer, std::size_t stride) {
    if (span <= 0) {
        return;
    }
    const bool previousRouteEntryIsWire = candidate.rawRoute.size() >= 3 && candidate.rawRoute.back() > 0;
    if (previousRouteEntryIsWire && !candidate.wireDeltas.empty()) {
        auto& previous = candidate.wireDeltas.back();
        const int previousLayer = previous.first / static_cast<int>(stride);
        if (previousLayer == layer &&
            (previous.first + previous.second == start || start + span == previous.first)) {
            previous.first = std::min(previous.first, start);
            previous.second += span;
            candidate.rawRoute[candidate.rawRoute.size() - 2] = previous.first;
            candidate.rawRoute.back() = previous.second;
            return;
        }
    }
    candidate.wireDeltas.emplace_back(start, span);
    candidate.rawRoute.push_back(start);
    candidate.rawRoute.push_back(span);
}

void appendPath(CpuRouteCandidate& candidate,
                const std::vector<int>& path,
                const GridShape& shape) {
    for (std::size_t index = 1; index < path.size(); ++index) {
        const int from = path[index - 1];
        const int to = path[index];
        const CpuMazePoint fromPoint = decode(shape, from);
        const CpuMazePoint toPoint = decode(shape, to);
        if (fromPoint.layer != toPoint.layer) {
            const int lowerLayer = std::min(fromPoint.layer, toPoint.layer);
            const int via = lowerLayer * static_cast<int>(shape.stride) +
                            coordinateOffset(shape, lowerLayer, fromPoint.x, fromPoint.y);
            if (std::find(candidate.viaDeltas.begin(), candidate.viaDeltas.end(), via) ==
                candidate.viaDeltas.end()) {
                candidate.rawRoute.push_back(via);
                candidate.rawRoute.push_back(-1);
                candidate.viaDeltas.push_back(via);
            }
            continue;
        }
        const int start = std::min(from, to);
        const bool alreadyPresent = std::any_of(
            candidate.wireDeltas.begin(),
            candidate.wireDeltas.end(),
            [start](const std::pair<int, int>& delta) {
                return start >= delta.first && start < delta.first + delta.second;
            });
        if (!alreadyPresent) {
            appendWireDelta(candidate, start, std::abs(to - from), fromPoint.layer, shape.stride);
        }
    }
}

int viaIndex(const GridShape& shape, int lowerLayer, int x, int y) {
    return lowerLayer * static_cast<int>(shape.stride) + coordinateOffset(shape, lowerLayer, x, y);
}

bool appendViaIfMissing(CpuRouteCandidate& candidate, int via) {
    if (std::find(candidate.viaDeltas.begin(), candidate.viaDeltas.end(), via) !=
        candidate.viaDeltas.end()) {
        return true;
    }
    candidate.rawRoute.push_back(via);
    candidate.rawRoute.push_back(-1);
    candidate.viaDeltas.push_back(via);
    return true;
}

CpuMazeFailureReason appendPinGroupVias(CpuRouteCandidate& candidate,
                        const CpuMazeSearchView& view,
                        const GridShape& shape,
                        const std::vector<CpuMazePoint>& group) {
    if (group.empty()) {
        return CpuMazeFailureReason::InvalidInput;
    }
    const int x = group.front().x;
    const int y = group.front().y;
    int minimumLayer = group.front().layer;
    int maximumLayer = group.front().layer;
    for (const CpuMazePoint& point : group) {
        if (!validPoint(view, shape, point) || point.x != x || point.y != y) {
            return CpuMazeFailureReason::InvalidInput;
        }
        minimumLayer = std::min(minimumLayer, point.layer);
        maximumLayer = std::max(maximumLayer, point.layer);
    }
    for (int lowerLayer = minimumLayer; lowerLayer < maximumLayer; ++lowerLayer) {
        const int via = viaIndex(shape, lowerLayer, x, y);
        if (viaStepCost(view, via) >= INF_DISTANCE) {
            return CpuMazeFailureReason::Unreachable;
        }
        appendViaIfMissing(candidate, via);
    }
    return CpuMazeFailureReason::None;
}

CpuMazeSearchResult failure(CpuMazeFailureReason reason, int netId, std::size_t expandedStates = 0,
                            const CpuMazeSearchStats& stats = {}) {
    CpuMazeSearchResult result;
    result.candidate.netId = netId;
    result.candidate.status = CpuRouteStatus::NoRoute;
    result.candidate.rawRoute = {-1};
    result.failureReason = reason;
    result.expandedStates = expandedStates;
    result.searchStats = stats;
    return result;
}

}  // namespace

std::size_t cpuMazeGridSize(const CpuMazeSearchView& view) {
    cpu_maze::validateView(view);
    return cpu_maze::checkedGridSize(view);
}

std::size_t CpuMazeSearchScratch::bytes() const {
    return fine.bytes() + coarse.bytes();
}

const char* cpuMazeFailureReasonName(CpuMazeFailureReason reason) {
    switch (reason) {
        case CpuMazeFailureReason::None:
            return "none";
        case CpuMazeFailureReason::InvalidInput:
            return "invalid_input";
        case CpuMazeFailureReason::Unreachable:
            return "unreachable";
        case CpuMazeFailureReason::RouteBuffer:
            return "route_buffer";
    }
    return "unknown";
}

CpuMazeSearchResult CpuMazeRouter::route(const CpuMazeNet& net, const CpuMazeSearchView& view) const {
    CpuMazeSearchScratch scratch;
    return route(net, view, scratch);
}

CpuMazeSearchResult CpuMazeRouter::route(const CpuMazeNet& net,
                                        const CpuMazeSearchView& view,
                                        CpuMazeSearchScratch& scratch) const {
    try {
        validateView(view);
        const GridShape shape = shapeOf(view);
        if (net.netId < 0 || net.pinGroups.empty()) {
            return failure(CpuMazeFailureReason::InvalidInput, net.netId);
        }

        CpuMazeSearchResult result;
        result.candidate.netId = net.netId;
        result.candidate.status = CpuRouteStatus::Routed;
        result.candidate.rawRoute = {1};
        const auto access = appendPinGroupVias(result.candidate, view, shape, net.pinGroups.front());
        if (access != CpuMazeFailureReason::None) return failure(access, net.netId);
        std::vector<int> connected;
        connected.reserve(net.pinGroups.front().size());
        for (const CpuMazePoint& point : net.pinGroups.front()) {
            const int encoded = encode(shape, point);
            if (!validPoint(view, shape, point)) {
                return failure(CpuMazeFailureReason::InvalidInput, net.netId);
            }
            connected.push_back(encoded);
        }
        if (connected.empty()) {
            return failure(CpuMazeFailureReason::InvalidInput, net.netId);
        }

        std::vector<CpuMazeBranch> branches = net.branches;
        if (branches.empty()) {
            for (std::size_t index = 1; index < net.pinGroups.size(); ++index) {
                branches.push_back({net.pinGroups[index], true});
            }
        }
        for (const auto& branch : branches) {
            std::unordered_set<int> targets;
            for (const CpuMazePoint& point : branch.targets) {
                if (!validPoint(view, shape, point)) {
                    return failure(CpuMazeFailureReason::InvalidInput, net.netId, result.expandedStates, result.searchStats);
                }
                targets.insert(encode(shape, point));
            }
            if (targets.empty()) {
                return failure(CpuMazeFailureReason::InvalidInput, net.netId, result.expandedStates, result.searchStats);
            }
            const SearchPath path = guidedShortestPath(view, shape, connected, targets, scratch, result.searchStats);
            result.expandedStates += path.expandedStates;
            if (path.nodes.empty()) {
                return failure(CpuMazeFailureReason::Unreachable, net.netId, result.expandedStates, result.searchStats);
            }
            appendPath(result.candidate, path.nodes, shape);
            if (branch.pinAccess) {
                const auto access = appendPinGroupVias(result.candidate, view, shape, branch.targets);
                if (access != CpuMazeFailureReason::None)
                    return failure(access, net.netId, result.expandedStates, result.searchStats);
            }
            connected.insert(connected.end(), path.nodes.begin(), path.nodes.end());
            if (branch.pinAccess) {
                for (const CpuMazePoint& point : branch.targets) {
                    connected.push_back(encode(shape, point));
                }
            }
            std::sort(connected.begin(), connected.end());
            connected.erase(std::unique(connected.begin(), connected.end()), connected.end());
        }
        std::sort(connected.begin(), connected.end());
        for (const auto& group : net.pinGroups) {
            for (const auto& point : group) {
                if (!std::binary_search(connected.begin(), connected.end(), encode(shape, point))) {
                    return failure(CpuMazeFailureReason::Unreachable, net.netId, result.expandedStates, result.searchStats);
                }
            }
        }
        if (result.candidate.rawRoute.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            return failure(CpuMazeFailureReason::RouteBuffer, net.netId, result.expandedStates, result.searchStats);
        }
        result.candidate.rawRoute[0] = static_cast<int>(result.candidate.rawRoute.size());
        result.failureReason = CpuMazeFailureReason::None;
        return result;
    } catch (const std::invalid_argument&) {
        return failure(CpuMazeFailureReason::InvalidInput, net.netId);
    } catch (const std::overflow_error&) {
        return failure(CpuMazeFailureReason::RouteBuffer, net.netId);
    } catch (const std::bad_alloc&) {
        return failure(CpuMazeFailureReason::RouteBuffer, net.netId);
    }
}

}  // namespace gr
