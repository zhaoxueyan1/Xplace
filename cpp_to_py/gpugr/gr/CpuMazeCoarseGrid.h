#pragma once

#include "CpuMazeSearch.h"

namespace gr {

// Coarse costs are approximate guidance. Fine-grid legality and costs remain
// authoritative, including when earlier commits have changed this snapshot.
class CpuMazeCoarseGrid {
public:
    CpuMazeCoarseGrid(const CpuMazeSearchView& fine, int scale, int width, int height);
    void rebuild(const CpuMazeSearchView& fine);
    // Mark during serial commits; refresh only after all batch workers finish.
    void invalidate(int fineResource);
    void refresh(const CpuMazeSearchView& fine);
    CpuMazeSearchView view() const;
    int scale() const { return scale_; }
    std::size_t nodeCount() const;
    std::size_t bytes() const;
    cpu_maze::CpuMazeCorridor corridor(const std::vector<int>& path, int radius) const;

private:
    int scale_;
    int width_;
    int height_;
    CpuMazeSearchView geometry_;
    cpu_maze::GridShape fineShape_;
    std::vector<int> wireCost_;
    std::vector<int> viaCost_;
    std::vector<uint8_t> dirty_;
};

cpu_maze::SearchPath guidedShortestPath(const CpuMazeSearchView& view,
                                      const cpu_maze::GridShape& shape,
                                      const std::vector<int>& sources,
                                      const std::unordered_set<int>& targets,
                                      CpuMazeSearchScratch& scratch,
                                      CpuMazeSearchStats& stats);

}  // namespace gr
