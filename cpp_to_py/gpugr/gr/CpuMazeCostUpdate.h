#pragma once

#include <utility>
#include <vector>

namespace gr {

struct CpuMazeCostUpdate {
    std::vector<int> wires;
    std::vector<int> vias;
};

// Exact cost-map dependency footprint for the CPU routing cost model:
// wire costs read their own usage and vias at both endpoints; via costs
// read wire usage at the two adjacent layers plus their own maze usage.
CpuMazeCostUpdate cpuMazeCostUpdate(const std::vector<std::pair<int, int>>& wires,
                                    const std::vector<int>& vias,
                                    int layerCount,
                                    int gridSize);

}  // namespace gr
