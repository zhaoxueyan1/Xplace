#include "CpuMazeCostUpdate.h"

#include <algorithm>

namespace gr {

CpuMazeCostUpdate cpuMazeCostUpdate(const std::vector<std::pair<int, int>>& wires,
                                    const std::vector<int>& vias,
                                    int layerCount,
                                    int gridSize) {
    const int stride = gridSize * gridSize;
    CpuMazeCostUpdate update;
    auto addWire = [&](int index) {
        if (index >= 0 && index < layerCount * stride && index % gridSize + 1 < gridSize) {
            update.wires.push_back(index);
        }
    };
    auto addVia = [&](int index) {
        if (index >= 0 && index < (layerCount - 1) * stride) {
            update.vias.push_back(index);
        }
    };
    auto transposed = [&](int index, int layer) {
        const int offset = index % stride;
        return layer * stride + (offset % gridSize) * gridSize + offset / gridSize;
    };
    for (const auto& [start, span] : wires) {
        for (int wire = start; wire < start + span; ++wire) {
            addWire(wire);
            const int layer = wire / stride;
            for (int cell : {wire, wire + 1}) {
                addVia(cell);
                if (layer > 0) addVia(transposed(cell, layer - 1));
            }
        }
    }
    for (int via : vias) {
        addVia(via);
        const int layer = via / stride;
        for (int cell : {via, transposed(via, layer + 1)}) {
            addWire(cell);
            if (cell % gridSize > 0) addWire(cell - 1);
        }
    }
    for (auto* values : {&update.wires, &update.vias}) {
        std::sort(values->begin(), values->end());
        values->erase(std::unique(values->begin(), values->end()), values->end());
    }
    return update;
}

}  // namespace gr
