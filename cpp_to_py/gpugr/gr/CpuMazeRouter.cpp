#include "CpuMazeRouter.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

namespace gr {

namespace {

constexpr int64_t INF_DISTANCE = std::numeric_limits<int64_t>::max() / 4;

struct GridShape {
    int layerCount;
    int xSize;
    int ySize;
    int direction;
    std::size_t stride;
};

struct SearchPath {
    std::vector<int> nodes;
    std::size_t expandedStates = 0;
};

std::size_t checkedGridSize(const CpuMazeSearchView& view) {
    if (view.layerCount <= 0 || view.xSize <= 0 || view.ySize <= 0 ||
        view.direction < 0 || view.direction > 1 || view.routingLayerBegin < 0 ||
        view.routingLayerEnd >= view.layerCount || view.routingLayerBegin > view.routingLayerEnd) {
        throw std::invalid_argument("invalid CPU maze search geometry");
    }
    const std::size_t layerCount = static_cast<std::size_t>(view.layerCount);
    const std::size_t xSize = static_cast<std::size_t>(view.xSize);
    const std::size_t ySize = static_cast<std::size_t>(view.ySize);
    if (layerCount > std::numeric_limits<std::size_t>::max() / xSize ||
        layerCount * xSize > std::numeric_limits<std::size_t>::max() / ySize) {
        throw std::overflow_error("CPU maze grid size overflow");
    }
    const std::size_t nodeCount = layerCount * xSize * ySize;
    if (nodeCount > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("CPU maze encoded grid index exceeds INT_MAX");
    }
    return nodeCount;
}

GridShape shapeOf(const CpuMazeSearchView& view) {
    checkedGridSize(view);
    return {view.layerCount, view.xSize, view.ySize, view.direction,
            static_cast<std::size_t>(view.xSize) * static_cast<std::size_t>(view.ySize)};
}

void validateMap(const std::vector<int>* map, std::size_t expected, const char* name) {
    if (map != nullptr && map->size() < expected) {
        throw std::invalid_argument(std::string("CPU maze ") + name + " map is too small");
    }
}

void validateFloatMap(const std::vector<float>* map, std::size_t expected, const char* name) {
    if (map != nullptr && map->size() < expected) {
        throw std::invalid_argument(std::string("CPU maze ") + name + " map is too small");
    }
}

void validateByteMap(const std::vector<uint8_t>* map, std::size_t expected, const char* name) {
    if (map != nullptr && map->size() < expected) {
        throw std::invalid_argument(std::string("CPU maze ") + name + " map is too small");
    }
}

void validateView(const CpuMazeSearchView& view) {
    const std::size_t expected = checkedGridSize(view);
    validateMap(view.wireUsage, expected, "wire-usage");
    validateMap(view.viaUsage, expected, "via-usage");
    validateMap(view.wireCost, expected, "wire-cost");
    validateMap(view.viaCost, expected, "via-cost");
    validateFloatMap(view.capacity, expected, "capacity");
    validateFloatMap(view.fixedUsage, expected, "fixed-usage");
    validateByteMap(view.blockedWire, expected, "blocked-wire");
    validateByteMap(view.blockedVia, expected, "blocked-via");
    if (view.overflowPenalty < 0) {
        throw std::invalid_argument("CPU maze overflow penalty must be non-negative");
    }
    if (view.horizontalWireLowerBound < 1 || view.verticalWireLowerBound < 1 || view.viaLowerBound < 1) {
        throw std::invalid_argument("CPU maze heuristic cost bounds must be positive");
    }
}

int coordinateOffset(const GridShape& shape, int layer, int x, int y) {
    const bool contiguousY = ((layer & 1) ^ shape.direction) != 0;
    return contiguousY ? x * shape.ySize + y : y * shape.xSize + x;
}

int encode(const GridShape& shape, const CpuMazePoint& point) {
    if (point.layer < 0 || point.layer >= shape.layerCount || point.x < 0 || point.x >= shape.xSize ||
        point.y < 0 || point.y >= shape.ySize) {
        return -1;
    }
    return point.layer * static_cast<int>(shape.stride) +
           coordinateOffset(shape, point.layer, point.x, point.y);
}

CpuMazePoint decode(const GridShape& shape, int encoded) {
    if (encoded < 0 || static_cast<std::size_t>(encoded) >= shape.stride * shape.layerCount) {
        return {};
    }
    const int layer = encoded / static_cast<int>(shape.stride);
    const bool contiguousY = ((layer & 1) ^ shape.direction) != 0;
    const int axisSize = contiguousY ? shape.ySize : shape.xSize;
    const int offset = encoded % static_cast<int>(shape.stride);
    const int mappedX = offset / axisSize;
    const int mappedY = offset % axisSize;
    if (contiguousY) {
        return {layer, mappedX, mappedY};
    }
    return {layer, mappedY, mappedX};
}

bool layerEnabled(const CpuMazeSearchView& view, int layer) {
    return layer >= view.routingLayerBegin && layer <= view.routingLayerEnd;
}

int64_t mapCost(const std::vector<int>* map, int encoded, int fallback) {
    if (map == nullptr) {
        return fallback;
    }
    const int value = (*map)[static_cast<std::size_t>(encoded)];
    if (value < 0) {
        return INF_DISTANCE;
    }
    return std::max(1, value);
}

bool isBlocked(const std::vector<uint8_t>* map, int encoded) {
    return map != nullptr && (*map)[static_cast<std::size_t>(encoded)] != 0;
}

int64_t wireStepCost(const CpuMazeSearchView& view, int encoded) {
    if (isBlocked(view.blockedWire, encoded) ||
        (view.capacity != nullptr && (*view.capacity)[encoded] < 0.01f)) {
        return INF_DISTANCE;
    }
    int64_t cost = mapCost(view.wireCost, encoded, 1);
    if (view.capacity != nullptr) {
        const float fixed = view.fixedUsage == nullptr
                                ? 0.0f
                                : (*view.fixedUsage)[static_cast<std::size_t>(encoded)];
        const float usage = view.wireUsage == nullptr
                                ? 0.0f
                                : static_cast<float>((*view.wireUsage)[static_cast<std::size_t>(encoded)]);
        if (fixed + usage >= (*view.capacity)[static_cast<std::size_t>(encoded)]) {
            cost += view.overflowPenalty;
        }
    }
    return cost;
}

int64_t viaStepCost(const CpuMazeSearchView& view, int encoded) {
    if (isBlocked(view.blockedVia, encoded)) {
        return INF_DISTANCE;
    }
    int64_t cost = mapCost(view.viaCost, encoded, 1);
    if (view.capacity != nullptr) {
        const float fixed = view.fixedUsage == nullptr
                                ? 0.0f
                                : (*view.fixedUsage)[static_cast<std::size_t>(encoded)];
        const float usage = view.viaUsage == nullptr
                                ? 0.0f
                                : static_cast<float>((*view.viaUsage)[static_cast<std::size_t>(encoded)]);
        if (fixed + usage >= (*view.capacity)[static_cast<std::size_t>(encoded)]) {
            cost += view.overflowPenalty;
        }
    }
    return cost;
}

template <typename Visitor>
void forEachNeighbour(const CpuMazeSearchView& view,
                      const GridShape& shape,
                      int encoded,
                      Visitor&& visitor) {
    const CpuMazePoint point = decode(shape, encoded);
    if (!layerEnabled(view, point.layer)) {
        return;
    }

    // The first two neighbours follow the preferred wire direction.  The
    // fixed order is part of the deterministic candidate contract.
    const bool contiguousY = ((point.layer & 1) ^ view.direction) != 0;
    const int primaryDelta[2] = {-1, 1};
    for (int delta : primaryDelta) {
        CpuMazePoint next = point;
        if (contiguousY) {
            next.y += delta;
        } else {
            next.x += delta;
        }
        const int nextEncoded = encode(shape, next);
        if (nextEncoded < 0) {
            continue;
        }
        const int wire = std::min(encoded, nextEncoded);
        const int64_t cost = wireStepCost(view, wire);
        if (cost < INF_DISTANCE) {
            visitor(nextEncoded, cost, wire, false);
        }
    }

    const int viaLayers[2] = {point.layer - 1, point.layer + 1};
    for (int nextLayer : viaLayers) {
        if (!layerEnabled(view, nextLayer) || std::abs(nextLayer - point.layer) != 1) {
            continue;
        }
        const int lowerLayer = std::min(point.layer, nextLayer);
        const int via = lowerLayer * static_cast<int>(shape.stride) +
                        coordinateOffset(shape, lowerLayer, point.x, point.y);
        const int64_t cost = viaStepCost(view, via);
        if (cost < INF_DISTANCE) {
            visitor(encode(shape, {nextLayer, point.x, point.y}), cost, via, true);
        }
    }
}

bool validPoint(const CpuMazeSearchView& view, const GridShape& shape, const CpuMazePoint& point) {
    return layerEnabled(view, point.layer) && encode(shape, point) >= 0;
}

SearchPath shortestPath(const CpuMazeSearchView& view,
                        const GridShape& shape,
                        const std::vector<int>& sources,
                        const std::unordered_set<int>& targets,
                        CpuMazeSearchScratch& scratch) {
    const std::size_t nodeCount = shape.stride * static_cast<std::size_t>(shape.layerCount);
    scratch.reset(nodeCount);
    auto& distances = scratch.distances;
    auto& previous = scratch.previous;
    // The minimum legal step costs give an admissible, consistent Manhattan
    // heuristic. A* searches the same full grid; no detour bound is imposed.
    auto remainingCost = [&](int node) {
        const auto point = decode(shape, node);
        int64_t minimum = INF_DISTANCE;
        for (int target : targets) {
            const auto goal = decode(shape, target);
            minimum = std::min(minimum,
                static_cast<int64_t>(std::abs(point.x - goal.x)) * view.horizontalWireLowerBound +
                static_cast<int64_t>(std::abs(point.y - goal.y)) * view.verticalWireLowerBound +
                static_cast<int64_t>(std::abs(point.layer - goal.layer)) * view.viaLowerBound);
        }
        return minimum;
    };
    auto& frontier = scratch.frontier;
    std::vector<int> orderedSources = sources;
    std::sort(orderedSources.begin(), orderedSources.end());
    orderedSources.erase(std::unique(orderedSources.begin(), orderedSources.end()), orderedSources.end());
    for (int source : orderedSources) {
        scratch.touched.push_back(source);
        distances[static_cast<std::size_t>(source)] = 0;
        previous[static_cast<std::size_t>(source)] = source;
        frontier.pushOrDecrease(source, remainingCost(source), 0);
    }

    SearchPath result;
    int reached = -1;
    while (!frontier.empty()) {
        const int current = frontier.pop();
        const int64_t distance = distances[static_cast<std::size_t>(current)];
        ++result.expandedStates;
        if (targets.count(current) != 0) {
            reached = current;
            break;
        }
        forEachNeighbour(view, shape, current, [&](int next, int64_t stepCost, int resource, bool via) {
            (void)resource;
            (void)via;
            const int64_t nextDistance = distance + stepCost;
            if (nextDistance < distances[static_cast<std::size_t>(next)]) {
                if (distances[static_cast<std::size_t>(next)] == INF_DISTANCE) {
                    scratch.touched.push_back(next);
                }
                distances[static_cast<std::size_t>(next)] = nextDistance;
                previous[static_cast<std::size_t>(next)] = current;
                frontier.pushOrDecrease(next, nextDistance + remainingCost(next), nextDistance);
            }
        });
    }
    if (reached < 0) {
        return result;
    }
    for (int cursor = reached;; cursor = previous[static_cast<std::size_t>(cursor)]) {
        result.nodes.push_back(cursor);
        if (previous[static_cast<std::size_t>(cursor)] == cursor) {
            break;
        }
    }
    std::reverse(result.nodes.begin(), result.nodes.end());
    return result;
}

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

bool appendPinGroupVias(CpuRouteCandidate& candidate,
                        const CpuMazeSearchView& view,
                        const GridShape& shape,
                        const std::vector<CpuMazePoint>& group) {
    if (group.empty()) {
        return false;
    }
    const int x = group.front().x;
    const int y = group.front().y;
    int minimumLayer = group.front().layer;
    int maximumLayer = group.front().layer;
    for (const CpuMazePoint& point : group) {
        if (!validPoint(view, shape, point) || point.x != x || point.y != y) {
            return false;
        }
        minimumLayer = std::min(minimumLayer, point.layer);
        maximumLayer = std::max(maximumLayer, point.layer);
    }
    for (int lowerLayer = minimumLayer; lowerLayer < maximumLayer; ++lowerLayer) {
        if (!layerEnabled(view, lowerLayer) || !layerEnabled(view, lowerLayer + 1)) {
            return false;
        }
        const int via = viaIndex(shape, lowerLayer, x, y);
        if (viaStepCost(view, via) >= INF_DISTANCE) {
            return false;
        }
        appendViaIfMissing(candidate, via);
    }
    return true;
}

CpuMazeSearchResult failure(CpuMazeFailureReason reason, int netId, std::size_t expandedStates = 0) {
    CpuMazeSearchResult result;
    result.candidate.netId = netId;
    result.candidate.status = CpuRouteStatus::NoRoute;
    result.candidate.rawRoute = {-1};
    result.failureReason = reason;
    result.expandedStates = expandedStates;
    return result;
}

}  // namespace

std::size_t cpuMazeGridSize(const CpuMazeSearchView& view) {
    validateView(view);
    return checkedGridSize(view);
}

void CpuMazeSearchScratch::reset(std::size_t nodeCount) {
    if (distances.size() != nodeCount || previous.size() != nodeCount) {
        distances.assign(nodeCount, INF_DISTANCE);
        previous.assign(nodeCount, -1);
    } else {
        for (int node : touched) {
            distances[static_cast<std::size_t>(node)] = INF_DISTANCE;
            previous[static_cast<std::size_t>(node)] = -1;
        }
    }
    touched.clear();
    touched.reserve(nodeCount);
    frontier.reset(nodeCount);
}

std::size_t CpuMazeSearchScratch::bytes() const {
    return distances.capacity() * sizeof(int64_t) +
           (previous.capacity() + touched.capacity()) * sizeof(int) + frontier.bytes();
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
        if (!appendPinGroupVias(result.candidate, view, shape, net.pinGroups.front())) {
            return failure(CpuMazeFailureReason::InvalidInput, net.netId);
        }
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
                    return failure(CpuMazeFailureReason::InvalidInput, net.netId, result.expandedStates);
                }
                targets.insert(encode(shape, point));
            }
            if (targets.empty()) {
                return failure(CpuMazeFailureReason::InvalidInput, net.netId, result.expandedStates);
            }
            const SearchPath path = shortestPath(view, shape, connected, targets, scratch);
            result.expandedStates += path.expandedStates;
            if (path.nodes.empty()) {
                return failure(CpuMazeFailureReason::Unreachable, net.netId, result.expandedStates);
            }
            appendPath(result.candidate, path.nodes, shape);
            if (branch.pinAccess && !appendPinGroupVias(result.candidate, view, shape, branch.targets)) {
                return failure(CpuMazeFailureReason::InvalidInput, net.netId, result.expandedStates);
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
                    return failure(CpuMazeFailureReason::Unreachable, net.netId, result.expandedStates);
                }
            }
        }
        if (result.candidate.rawRoute.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            return failure(CpuMazeFailureReason::RouteBuffer, net.netId, result.expandedStates);
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
