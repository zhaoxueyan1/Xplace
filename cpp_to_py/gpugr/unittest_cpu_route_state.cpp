#include "gpugr/gr/CpuRouteState.h"

#include <cassert>
#include <numeric>
#include <random>
#include <algorithm>
#include <stdexcept>

namespace {

template <typename Exception, typename Function>
void assertThrows(Function&& function) {
    bool thrown = false;
    try {
        function();
    } catch (const Exception&) {
        thrown = true;
    }
    assert(thrown);
}

}  // namespace

int main() {
    gr::CpuRouteState state;
    state.resetResources(16);
    state.resetRoutes(2);

    gr::CpuRouteCandidate candidate;
    candidate.netId = 0;
    candidate.status = gr::CpuRouteStatus::Routed;
    candidate.rawRoute = {5, 2, 2, 7, -1};
    candidate.wireDeltas = {{2, 2}};
    candidate.viaDeltas = {7};
    state.commit(candidate);
    assert(state.wires()[2] == 1 && state.wires()[3] == 1);
    assert(state.vias()[7] == 1);
    assert(state.rawRoutes()[0] == candidate.rawRoute);

    const gr::CpuRouteSnapshot removed = state.remove(0);
    assert(state.wires()[2] == 0 && state.wires()[3] == 0);
    assert(state.vias()[7] == 0);
    assert(state.rawRoutes()[0] == std::vector<int>({1}));
    state.restore(0, removed);
    assert(state.wires()[2] == 1 && state.wires()[3] == 1);
    assert(state.vias()[7] == 1);
    assert(state.rawRoutes()[0] == candidate.rawRoute);

    assertThrows<std::runtime_error>([&] { state.restore(0, removed); });
    state.remove(0);
    assertThrows<std::runtime_error>([&] { state.remove(0); });

    gr::CpuRouteCandidate invalid;
    invalid.netId = 1;
    invalid.status = gr::CpuRouteStatus::Routed;
    invalid.rawRoute = {3, 0, 2};
    invalid.wireDeltas = {{15, 2}};
    assertThrows<std::runtime_error>([&] { state.commit(invalid); });
    for (int usage : state.wires()) {
        assert(usage >= 0);
    }

    state.setInactiveRoute(1);
    assert(state.rawRoutes()[1] == std::vector<int>({0}));

    gr::CpuRouteState atomic;
    atomic.resetResources(8);
    atomic.resetRoutes(1);
    gr::CpuRouteCandidate atomicCandidate;
    atomicCandidate.netId = 0;
    atomicCandidate.status = gr::CpuRouteStatus::Routed;
    atomicCandidate.rawRoute = {5, 2, 1, 5, 1};
    atomicCandidate.wireDeltas = {{2, 1}, {5, 1}};
    atomic.commit(atomicCandidate);
    atomic.wires()[2] = 0;
    assertThrows<std::runtime_error>([&] { atomic.remove(0); });
    assert(atomic.wires()[5] == 1);
    assert(atomic.snapshot(0).active);

    gr::CpuRouteState duplicate;
    duplicate.resetResources(8);
    duplicate.resetRoutes(1);
    gr::CpuRouteCandidate repeated;
    repeated.netId = 0;
    repeated.status = gr::CpuRouteStatus::Routed;
    repeated.rawRoute = {9, 2, 1, 2, 1, 5, -1, 5, -1};
    repeated.wireDeltas = {{2, 1}, {2, 1}};
    repeated.viaDeltas = {5, 5};
    duplicate.commit(repeated);
    duplicate.wires()[2] = 1;
    assertThrows<std::runtime_error>([&] { duplicate.remove(0); });
    assert(duplicate.wires()[2] == 1 && duplicate.vias()[5] == 2);
    assert(duplicate.snapshot(0).active);
    duplicate.wires()[2] = 2;
    duplicate.vias()[5] = 1;
    assertThrows<std::runtime_error>([&] { duplicate.remove(0); });
    assert(duplicate.wires()[2] == 2 && duplicate.vias()[5] == 1);

    gr::CpuRouteState mismatch;
    mismatch.resetResources(16);
    mismatch.resetRoutes(1);
    candidate.rawRoute[1] = 4;
    assertThrows<std::runtime_error>([&] { mismatch.commit(candidate); });
    assert(!mismatch.snapshot(0).active);
    for (int usage : mismatch.wires()) assert(usage == 0);

    gr::CpuRouteState randomized;
    randomized.resetResources(64);
    randomized.resetRoutes(100);
    std::mt19937 generator(73021);
    for (int net = 0; net < 100; ++net) {
        gr::CpuRouteCandidate route;
        route.netId = net;
        route.status = gr::CpuRouteStatus::Routed;
        const int start = generator() % 60, span = 1 + generator() % 4;
        const int via = generator() % 64;
        route.rawRoute = {5, start, span, via, -1};
        route.wireDeltas = {{start, span}};
        route.viaDeltas = {via};
        randomized.commit(route);
    }
    const auto saved = randomized.checkpoint();
    const auto wires = randomized.wires(), vias = randomized.vias();
    std::vector<int> order(100);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), generator);
    for (int net : order) randomized.remove(net);
    for (int usage : randomized.wires()) assert(usage == 0);
    for (int usage : randomized.vias()) assert(usage == 0);
    randomized.restoreCheckpoint(saved);
    assert(randomized.wires() == wires && randomized.vias() == vias);
    assert(randomized.rawRoutes() == saved.rawRoutes);

    const gr::CpuRouteCheckpoint checkpoint = state.checkpoint();
    state.resetResources(16);
    state.resetRoutes(2);
    state.restoreCheckpoint(checkpoint);
    assert(state.rawRoutes()[0] == std::vector<int>({1}));
    assert(state.wires()[2] == 0 && state.wires()[3] == 0);
    assert(state.vias()[7] == 0);
    return 0;
}
