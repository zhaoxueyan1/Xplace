#pragma once

#include "CpuMazeRouter.h"

#include <unordered_set>
#include <limits>

namespace gr {
namespace cpu_maze {

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

// A corridor is a union of coarse cells, open on every enabled routing layer.
struct CpuMazeCorridor {
    int width;
    int scale;
    std::vector<uint8_t> cells;
    bool contains(const CpuMazePoint& point) const;
};

std::size_t checkedGridSize(const CpuMazeSearchView& view);
void validateView(const CpuMazeSearchView& view);
GridShape shapeOf(const CpuMazeSearchView& view);
int coordinateOffset(const GridShape& shape, int layer, int x, int y);
int encode(const GridShape& shape, const CpuMazePoint& point);
CpuMazePoint decode(const GridShape& shape, int encoded);
bool layerEnabled(const CpuMazeSearchView& view, int layer);
bool validPoint(const CpuMazeSearchView& view, const GridShape& shape, const CpuMazePoint& point);
int64_t wireStepCost(const CpuMazeSearchView& view, int encoded);
int64_t viaStepCost(const CpuMazeSearchView& view, int encoded);
SearchPath shortestPath(const CpuMazeSearchView& view, const GridShape& shape,
                        const std::vector<int>& sources, const std::unordered_set<int>& targets,
                        CpuMazePathScratch& scratch, const CpuMazeCorridor* corridor = nullptr);

}  // namespace cpu_maze
}  // namespace gr
