#include "gpugr/gr/CpuMazeRouter.h"
#include "gpugr/gr/CpuRouteState.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

namespace {

gr::CpuMazeSearchView makeView(std::vector<int>& wireUsage,
                               std::vector<int>& viaUsage,
                               std::vector<int>& wireCost,
                               std::vector<int>& viaCost,
                               std::vector<float>& capacity,
                               std::vector<float>& fixedUsage,
                               std::vector<uint8_t>& blockedWire,
                               std::vector<uint8_t>& blockedVia) {
    gr::CpuMazeSearchView view;
    view.layerCount = 3;
    view.xSize = 5;
    view.ySize = 5;
    view.direction = 0;
    view.routingLayerBegin = 0;
    view.routingLayerEnd = 2;
    view.wireUsage = &wireUsage;
    view.viaUsage = &viaUsage;
    view.wireCost = &wireCost;
    view.viaCost = &viaCost;
    view.capacity = &capacity;
    view.fixedUsage = &fixedUsage;
    view.blockedWire = &blockedWire;
    view.blockedVia = &blockedVia;
    view.overflowPenalty = 1000;
    view.impassableCost = 10000000;
    return view;
}

bool usesWire(const gr::CpuRouteCandidate& candidate, int resource) {
    return std::any_of(candidate.wireDeltas.begin(), candidate.wireDeltas.end(),
                       [resource](const auto& delta) {
                           return resource >= delta.first && resource < delta.first + delta.second;
                       });
}

}  // namespace

int main() {
    constexpr int resourceCount = 3 * 5 * 5;
    std::vector<int> wireUsage(resourceCount, 0);
    std::vector<int> viaUsage(resourceCount, 0);
    std::vector<int> wireCost(resourceCount, 1);
    std::vector<int> viaCost(resourceCount, 1);
    std::vector<float> capacity(resourceCount, 1.0f);
    std::vector<float> fixedUsage(resourceCount, 0.0f);
    std::vector<uint8_t> blockedWire(resourceCount, 0);
    std::vector<uint8_t> blockedVia(resourceCount, 0);
    auto view = makeView(wireUsage,
                         viaUsage,
                         wireCost,
                         viaCost,
                         capacity,
                         fixedUsage,
                         blockedWire,
                         blockedVia);

    const gr::CpuMazeNet netA{0, {{{0, 0, 0}}, {{0, 2, 0}}}};
    const gr::CpuMazeNet netB{1, {{{0, 0, 0}}, {{0, 2, 0}}}};
    gr::CpuMazeRouter router;
    const auto initialA = router.route(netA, view);
    const auto initialB = router.route(netB, view);
    assert(initialA.succeeded() && initialB.succeeded());
    assert(initialA.candidate.rawRoute == initialB.candidate.rawRoute);

    gr::CpuRouteState state;
    state.resetResources(resourceCount);
    state.resetRoutes(2);
    state.commit(initialA.candidate);
    state.commit(initialB.candidate);
    assert(state.wires()[0] == 2 && state.wires()[1] == 2);

    const gr::CpuRouteSnapshot oldB = state.remove(1);
    assert(state.wires()[0] == 1 && state.wires()[1] == 1);
    view.wireUsage = &state.wires();
    view.viaUsage = &state.vias();
    const auto detour = router.route(netB, view);
    assert(detour.succeeded());
    assert(usesWire(detour.candidate, 50));
    assert(!usesWire(detour.candidate, 0));
    state.commit(detour.candidate);
    assert(state.wires()[0] == 1 && state.wires()[1] == 1);

    const auto best = state.checkpoint();
    const auto removedDetour = state.remove(1);
    assert(state.wires()[0] == 1 && state.wires()[1] == 1);
    blockedWire[0] = 1;
    blockedWire[1] = 1;
    std::fill(blockedVia.begin(), blockedVia.end(), 1);
    view.blockedVia = &blockedVia;
    view.blockedWire = &blockedWire;
    const auto impossible = router.route(netB, view);
    assert(!impossible.succeeded());
    assert(impossible.failureReason == gr::CpuMazeFailureReason::Unreachable);
    state.restore(1, removedDetour);
    state.restoreCheckpoint(best);
    assert(state.rawRoutes()[1] == best.rawRoutes[1]);
    assert(state.wires()[0] == 1 && state.wires()[1] == 1);

    state.remove(1);
    state.restore(1, oldB);
    assert(state.wires()[0] == 2 && state.wires()[1] == 2);
    return 0;
}
