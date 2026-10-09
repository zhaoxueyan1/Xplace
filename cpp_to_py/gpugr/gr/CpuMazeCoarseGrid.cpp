#include "CpuMazeCoarseGrid.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace gr {
namespace {
using namespace cpu_maze;
int meanCost(int64_t sum, int count, int scale) {
    if (count == 0) return -1;
    return static_cast<int>(std::min<int64_t>(std::numeric_limits<int>::max(),
                                            std::max<int64_t>(1, sum / count) * scale));
}
}  // namespace

CpuMazeCoarseGrid::CpuMazeCoarseGrid(const CpuMazeSearchView& fine, int scale, int width, int height)
    : scale_(scale), width_(width), height_(height) {
    cpuMazeGridSize(fine);
    if (scale < 1 || width < 1 || height < 1 || width > fine.xSize || height > fine.ySize)
        throw std::invalid_argument("invalid CPU maze coarsening geometry");
    geometry_.layerCount = fine.layerCount;
    geometry_.direction = fine.direction;
    geometry_.routingLayerBegin = fine.routingLayerBegin;
    geometry_.routingLayerEnd = fine.routingLayerEnd;
    geometry_.xSize = 1 + (width - 1) / scale;
    geometry_.ySize = 1 + (height - 1) / scale;
    geometry_.overflowPenalty = 0;
    fineShape_ = shapeOf(fine);
}

std::size_t CpuMazeCoarseGrid::nodeCount() const {
    return scale_ == 1 ? 0 : cpu_maze::checkedGridSize(geometry_);
}

void CpuMazeCoarseGrid::rebuild(const CpuMazeSearchView& fine) {
    dirty_.assign(nodeCount(), 1);
    refresh(fine);
}

void CpuMazeCoarseGrid::invalidate(int fineResource) {
    if (scale_ == 1 || dirty_.empty()) return;
    const auto point = decode(fineShape_, fineResource);
    if (point.x < 0 || point.x >= width_ || point.y < 0 || point.y >= height_) return;
    const int node = encode(shapeOf(geometry_), {point.layer, point.x / scale_, point.y / scale_});
    dirty_[node] = 1;
}

void CpuMazeCoarseGrid::refresh(const CpuMazeSearchView& fine) {
    if (scale_ == 1) return;
    validateView(fine);
    const auto fineShape = shapeOf(fine);
    const auto coarseShape = shapeOf(geometry_);
    if (wireCost_.empty()) {
        wireCost_.assign(nodeCount(), -1);
        viaCost_.assign(nodeCount(), -1);
        dirty_.assign(nodeCount(), 1);
    }
    for (int layer = geometry_.routingLayerBegin; layer <= geometry_.routingLayerEnd; ++layer) {
        const bool alongY = ((layer & 1) ^ fine.direction) != 0;
        for (int y = 0; y < geometry_.ySize; ++y) {
            for (int x = 0; x < geometry_.xSize; ++x) {
                const int coarse = encode(coarseShape, {layer, x, y});
                if (!dirty_[coarse]) continue;
                int64_t wireSum = 0, viaSum = 0;
                int wireCount = 0, viaCount = 0;
                const int xEnd = std::min(width_, (x + 1) * scale_);
                const int yEnd = std::min(height_, (y + 1) * scale_);
                for (int fy = y * scale_; fy < yEnd; ++fy) {
                    for (int fx = x * scale_; fx < xEnd; ++fx) {
                        const int encoded = encode(fineShape, {layer, fx, fy});
                        if ((alongY ? fy + 1 < height_ : fx + 1 < width_)) {
                            const auto cost = wireStepCost(fine, encoded);
                            if (cost < INF_DISTANCE) { wireSum += cost; ++wireCount; }
                        }
                        if (layer < geometry_.routingLayerEnd) {
                            const auto cost = viaStepCost(fine, encoded);
                            if (cost < INF_DISTANCE) { viaSum += cost; ++viaCount; }
                        }
                    }
                }
                wireCost_[coarse] = meanCost(wireSum, wireCount, scale_);
                viaCost_[coarse] = meanCost(viaSum, viaCount, 1);
                dirty_[coarse] = 0;
            }
        }
    }
    geometry_.horizontalWireLowerBound = geometry_.verticalWireLowerBound =
        geometry_.viaLowerBound = std::numeric_limits<int>::max();
    for (std::size_t node = 0; node < nodeCount(); ++node) {
        const auto point = decode(coarseShape, static_cast<int>(node));
        auto& bound = ((point.layer & 1) ^ geometry_.direction)
                          ? geometry_.verticalWireLowerBound : geometry_.horizontalWireLowerBound;
        if (wireCost_[node] > 0) bound = std::min(bound, wireCost_[node]);
        if (viaCost_[node] > 0) geometry_.viaLowerBound = std::min(geometry_.viaLowerBound, viaCost_[node]);
    }
}

CpuMazeSearchView CpuMazeCoarseGrid::view() const {
    auto result = geometry_;
    result.wireCost = &wireCost_;
    result.viaCost = &viaCost_;
    return result;
}

std::size_t CpuMazeCoarseGrid::bytes() const {
    return (wireCost_.capacity() + viaCost_.capacity()) * sizeof(int) + dirty_.capacity();
}

cpu_maze::CpuMazeCorridor CpuMazeCoarseGrid::corridor(const std::vector<int>& path, int radius) const {
    CpuMazeCorridor result{geometry_.xSize, scale_,
                          std::vector<uint8_t>(geometry_.xSize * geometry_.ySize, 0)};
    const auto shape = shapeOf(geometry_);
    for (int node : path) {
        const auto point = decode(shape, node);
        for (int y = std::max(0, point.y - radius); y <= std::min(shape.ySize - 1, point.y + radius); ++y)
            for (int x = std::max(0, point.x - radius); x <= std::min(shape.xSize - 1, point.x + radius); ++x)
                result.cells[y * shape.xSize + x] = 1;
    }
    return result;
}

cpu_maze::SearchPath guidedShortestPath(const CpuMazeSearchView& view, const GridShape& shape,
                                      const std::vector<int>& sources,
                                      const std::unordered_set<int>& targets,
                                      CpuMazeSearchScratch& scratch, CpuMazeSearchStats& stats) {
    std::size_t expanded = 0;
    const auto* grid = view.coarseGrid;
    if (grid != nullptr && grid->scale() > 1) {
        const auto coarseView = grid->view();
        const auto coarseShape = shapeOf(coarseView);
        auto coarseNode = [&](int node) {
            auto point = decode(shape, node);
            point.x /= grid->scale();
            point.y /= grid->scale();
            return encode(coarseShape, point);
        };
        std::vector<int> coarseSources;
        std::unordered_set<int> coarseTargets;
        for (int node : sources) if (coarseNode(node) >= 0) coarseSources.push_back(coarseNode(node));
        for (int node : targets) if (coarseNode(node) >= 0) coarseTargets.insert(coarseNode(node));
        const auto coarsePath = shortestPath(coarseView, coarseShape, coarseSources,
                                             coarseTargets, scratch.coarse);
        expanded += coarsePath.expandedStates;
        stats.coarseExpandedStates += coarsePath.expandedStates;
        if (!coarsePath.nodes.empty()) {
            for (int radius : {0, 1, 2}) {
                const auto corridor = grid->corridor(coarsePath.nodes, radius);
                auto finePath = shortestPath(view, shape, sources, targets, scratch.fine, &corridor);
                stats.fineExpandedStates += finePath.expandedStates;
                expanded += finePath.expandedStates;
                if (!finePath.nodes.empty()) {
                    finePath.expandedStates = expanded;
                    return finePath;
                }
                ++stats.corridorRetries;
            }
        }
        ++stats.fullSearchFallbacks;
    }
    auto path = shortestPath(view, shape, sources, targets, scratch.fine);
    stats.fineExpandedStates += path.expandedStates;
    path.expandedStates += expanded;
    return path;
}

}  // namespace gr
