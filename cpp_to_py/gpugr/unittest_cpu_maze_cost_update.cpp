#include "gpugr/gr/CpuMazeCostUpdate.h"

#include <algorithm>
#include <cassert>
#include <vector>

int main() {
    // A layer-1 wire at (x=1,y=0), plus its second edge. Adjacent via
    // costs on both sides of the layer must observe its usage change.
    const auto wire = gr::cpuMazeCostUpdate({{30, 2}}, {}, 3, 5);
    assert(wire.wires == std::vector<int>({30, 31}));
    assert(wire.vias == std::vector<int>({1, 6, 11, 30, 31, 32}));
    // A via changes both neighboring wire layers, including the incoming
    // edge at each cell. Layer encodings alternate their contiguous axis.
    const auto via = gr::cpuMazeCostUpdate({}, {6}, 3, 5);
    assert(via.wires == std::vector<int>({5, 6, 30, 31}));
    assert(via.vias == std::vector<int>({6}));
    // Cells on a row boundary must not affect the preceding encoded row.
    const auto boundary = gr::cpuMazeCostUpdate({}, {0}, 3, 5);
    assert(boundary.wires == std::vector<int>({0, 25}));

    constexpr int layers = 4, size = 5, stride = size * size;
    auto transpose = [](int cell, int layer) {
        return layer * stride + cell % size * size + cell / size % size;
    };
    // Enumerate the dependency reads used by CPURouter's wire/via costs.
    // Check every single-resource mutation, independently of the update
    // implementation's traversal direction and candidate deduplication.
    for (int changed = 0; changed < layers * stride; ++changed) {
        for (bool changedVia : {false, true}) {
            if (changedVia && changed >= (layers - 1) * stride) continue;
            if (!changedVia && changed % size == size - 1) continue;
            const auto update = changedVia
                ? gr::cpuMazeCostUpdate({}, {changed}, layers, size)
                : gr::cpuMazeCostUpdate({{changed, 1}}, {}, layers, size);
            for (int resource = 0; resource < layers * stride; ++resource) {
                const int layer = resource / stride;
                bool wireReadsChange = !changedVia && changed == resource;
                if (changedVia && resource % size + 1 < size) {
                    for (int cell : {resource, resource + 1}) {
                        wireReadsChange |= layer + 1 < layers && changed == cell;
                        wireReadsChange |= layer > 0 && changed == transpose(cell, layer - 1);
                    }
                }
                if (resource % size + 1 < size && wireReadsChange) {
                    assert(std::binary_search(update.wires.begin(), update.wires.end(), resource));
                }
                if (layer + 1 >= layers) continue;
                bool viaReadsChange = changedVia && changed == resource;
                if (!changedVia) {
                    for (int cell : {resource, transpose(resource, layer + 1)}) {
                        viaReadsChange |= cell % size > 0 && changed == cell - 1;
                        viaReadsChange |= cell % size + 1 < size && changed == cell;
                    }
                }
                if (viaReadsChange) {
                    assert(std::binary_search(update.vias.begin(), update.vias.end(), resource));
                }
            }
        }
    }
}
