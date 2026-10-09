#include "gpugr/gr/CpuMazeCoarseGrid.h"

#include <algorithm>
#include <cassert>
#include <queue>
#include <vector>

namespace {
using namespace gr;
using namespace gr::cpu_maze;

struct Fixture {
    std::vector<int> wires, vias;
    std::vector<uint8_t> blockedWire, blockedVia;
    CpuMazeSearchView view;

    Fixture(int width, int height, int direction = 0)
        : wires(3 * width * height, 1000), vias(wires.size(), 1000),
          blockedWire(wires.size(), 0), blockedVia(wires.size(), 0) {
        view.layerCount = 3;
        view.xSize = width;
        view.ySize = height;
        view.direction = direction;
        view.routingLayerEnd = 2;
        view.wireCost = &wires;
        view.viaCost = &vias;
        view.blockedWire = &blockedWire;
        view.blockedVia = &blockedVia;
    }
};

void checkRoute(const CpuMazeNet& net, const CpuMazeSearchResult& result,
                const CpuMazeSearchView& view) {
    assert(result.succeeded());
    const auto shape = shapeOf(view);
    std::vector<std::vector<int>> adjacency(checkedGridSize(view));
    auto connect = [&](int from, int to) {
        adjacency[from].push_back(to);
        adjacency[to].push_back(from);
    };
    for (const auto& [start, span] : result.candidate.wireDeltas) {
        assert(span > 0);
        for (int node = start; node < start + span; ++node) {
            assert(wireStepCost(view, node) < INF_DISTANCE);
            auto p = decode(shape, node), q = decode(shape, node + 1);
            assert(p.layer == q.layer && std::abs(p.x - q.x) + std::abs(p.y - q.y) == 1);
            connect(node, node + 1);
        }
    }
    for (int via : result.candidate.viaDeltas) {
        assert(viaStepCost(view, via) < INF_DISTANCE);
        auto p = decode(shape, via);
        assert(p.layer < view.routingLayerEnd);
        ++p.layer;
        connect(via, encode(shape, p));
    }
    std::vector<uint8_t> visited(adjacency.size(), 0);
    std::queue<int> queue;
    queue.push(encode(shape, net.pinGroups.front().front()));
    visited[queue.front()] = 1;
    while (!queue.empty()) {
        const int node = queue.front();
        queue.pop();
        for (int next : adjacency[node]) if (!visited[next]) {
            visited[next] = 1;
            queue.push(next);
        }
    }
    for (const auto& group : net.pinGroups)
        for (const auto& pin : group) assert(visited[encode(shape, pin)]);
}
}  // namespace

int main() {
    CpuMazeRouter router;
    for (int direction : {0, 1}) {
        Fixture f(65, 49, direction);  // Partial edge blocks and rectangular, transposed storage.
        CpuMazeNet net{0, {{{0, 2, 3}}, {{0, 61, 45}}, {{0, 25, 20}, {1, 25, 20}}}};
        CpuMazeCoarseGrid grid(f.view, 8, 65, 49);
        grid.rebuild(f.view);
        auto guided = f.view;
        guided.coarseGrid = &grid;
        const auto full = router.route(net, f.view);
        const auto result = router.route(net, guided);
        checkRoute(net, result, f.view);
        assert(result.searchStats.coarseExpandedStates > 0);
        assert(result.searchStats.fineExpandedStates > 0);
        assert(result.expandedStates < full.expandedStates);
        CpuMazeSearchScratch scratch;
        assert(router.route(net, guided, scratch).candidate.rawRoute == result.candidate.rawRoute);
        const std::vector<CpuMazeNet> nets(9, net);
        const auto serial = router.routeBatch(nets, guided, 1);
        for (int workers : {2, 4, 8}) {
            const auto parallel = router.routeBatch(nets, guided, workers);
            for (std::size_t i = 0; i < nets.size(); ++i)
                assert(parallel.results[i].candidate.rawRoute == serial.results[i].candidate.rawRoute);
        }
        // Rebuilding costs must change the immutable snapshot used by the next batch.
        const auto previous = *grid.view().wireCost;
        std::fill(f.wires.begin(), f.wires.end(), 10000000);
        grid.rebuild(f.view);
        assert(*grid.view().wireCost != previous);
        checkRoute(net, router.route(net, guided, scratch), f.view);
        // A commit updates both wire and via costs in one block. Incremental
        // refresh must produce the exact same guide maps as a full rebuild.
        const auto fineShape = shapeOf(f.view);
        for (int layer : {0, 1, 2}) {
            const int node = encode(fineShape, {layer, 31, 17});
            f.wires[node] = 17;
            f.vias[node] = 23;
            grid.invalidate(node);
        }
        grid.refresh(f.view);
        const auto wires = *grid.view().wireCost;
        const auto vias = *grid.view().viaCost;
        grid.rebuild(f.view);
        assert(*grid.view().wireCost == wires && *grid.view().viaCost == vias);
        CpuMazeCoarseGrid disabled(f.view, 1, 65, 49);
        guided.coarseGrid = &disabled;
        assert(router.route(net, guided).candidate.rawRoute == router.route(net, f.view).candidate.rawRoute);
    }

    Fixture barrier(64, 48);
    const auto shape = shapeOf(barrier.view);
    for (int layer : {0, 2}) for (int y = 0; y < 46; ++y)
        barrier.blockedWire[encode(shape, {layer, 27, y})] = 1;
    CpuMazeNet crossing{1, {{{0, 20, 3}}, {{0, 35, 3}}}};
    CpuMazeCoarseGrid grid(barrier.view, 8, 64, 48);
    grid.rebuild(barrier.view);
    auto guided = barrier.view;
    guided.coarseGrid = &grid;
    const auto rescued = router.route(crossing, guided);
    checkRoute(crossing, rescued, barrier.view);
    assert(rescued.searchStats.corridorRetries == 3);
    assert(rescued.searchStats.fullSearchFallbacks == 1);

    // Coarse guidance can become stale within a batch; fine legality is authoritative.
    for (int layer : {0, 2}) for (int y = 46; y < 48; ++y)
        barrier.blockedWire[encode(shape, {layer, 27, y})] = 1;
    assert(router.route(crossing, guided).failureReason == CpuMazeFailureReason::Unreachable);
    return 0;
}
