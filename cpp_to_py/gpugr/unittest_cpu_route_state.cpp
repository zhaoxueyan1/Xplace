#include "gpugr/gr/CpuRouteState.h"

#include <cassert>
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
    return 0;
}
