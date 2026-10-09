#include "CpuMazeSearch.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace gr {
namespace cpu_maze {

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
                        CpuMazePathScratch& scratch,
                        const CpuMazeCorridor* corridor) {
    const std::size_t nodeCount = shape.stride * static_cast<std::size_t>(shape.layerCount);
    scratch.reset(nodeCount);
    auto& distances = scratch.distances;
    auto& previous = scratch.previous;
    // The minimum legal step costs give an admissible, consistent Manhattan
    // heuristic, shared by full-grid search and corridor refinement.
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
        if (corridor != nullptr && !corridor->contains(decode(shape, source))) continue;
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
            if (corridor != nullptr && !corridor->contains(decode(shape, next))) return;
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

bool CpuMazeCorridor::contains(const CpuMazePoint& point) const {
    const int x = point.x / scale;
    const int y = point.y / scale;
    return point.x >= 0 && point.y >= 0 && x < width &&
           static_cast<std::size_t>(y * width + x) < cells.size() && cells[y * width + x] != 0;
}

}  // namespace cpu_maze

void CpuMazePathScratch::reset(std::size_t nodeCount) {
    if (distances.size() != nodeCount || previous.size() != nodeCount) {
        distances.assign(nodeCount, cpu_maze::INF_DISTANCE);
        previous.assign(nodeCount, -1);
    } else {
        for (int node : touched) {
            distances[static_cast<std::size_t>(node)] = cpu_maze::INF_DISTANCE;
            previous[static_cast<std::size_t>(node)] = -1;
        }
    }
    touched.clear();
    touched.reserve(nodeCount);
    frontier.reset(nodeCount);
}

std::size_t CpuMazePathScratch::bytes() const {
    return distances.capacity() * sizeof(int64_t) +
           (previous.capacity() + touched.capacity()) * sizeof(int) + frontier.bytes();
}

}  // namespace gr
