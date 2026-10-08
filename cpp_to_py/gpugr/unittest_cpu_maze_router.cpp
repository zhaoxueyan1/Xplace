#include "gpugr/gr/CpuMazeRouter.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

namespace {

gr::CpuMazeSearchView makeView(int layers,
                               int width,
                               int height,
                               std::vector<int>& wireUsage,
                               std::vector<int>& viaUsage,
                               std::vector<int>& wireCost,
                               std::vector<int>& viaCost,
                               std::vector<float>& capacity,
                               std::vector<float>& fixedUsage,
                               std::vector<uint8_t>& blockedWire,
                               std::vector<uint8_t>& blockedVia) {
    gr::CpuMazeSearchView view;
    view.layerCount = layers;
    view.xSize = width;
    view.ySize = height;
    view.direction = 0;
    view.routingLayerBegin = 0;
    view.routingLayerEnd = layers - 1;
    view.wireUsage = &wireUsage;
    view.viaUsage = &viaUsage;
    view.wireCost = &wireCost;
    view.viaCost = &viaCost;
    view.capacity = &capacity;
    view.fixedUsage = &fixedUsage;
    view.blockedWire = &blockedWire;
    view.blockedVia = &blockedVia;
    view.overflowPenalty = 1000;
    return view;
}

void assertValidRoute(const gr::CpuRouteCandidate& candidate,
                      int layers,
                      int width,
                      int height,
                      int direction = 0) {
    assert(candidate.status == gr::CpuRouteStatus::Routed);
    assert(candidate.rawRoute[0] == static_cast<int>(candidate.rawRoute.size()));
    for (std::size_t index = 1; index + 1 < candidate.rawRoute.size(); index += 2) {
        const int point = candidate.rawRoute[index];
        const int span = candidate.rawRoute[index + 1];
        assert(point >= 0);
        if (span == -1) {
            assert(point < (layers - 1) * width * height);
        } else {
            assert(span > 0);
            assert(point + span <= layers * width * height);
            const int layer = point / (width * height);
            const int offset = point % (width * height);
            const bool contiguousY = ((layer & 1) ^ direction) != 0;
            const int axisSize = contiguousY ? height : width;
            const int mappedX = offset / axisSize;
            const int mappedY = offset % axisSize;
            assert(mappedX >= 0 && mappedX < (contiguousY ? width : height));
            if (contiguousY) {
                assert(mappedY + span < height);
            } else {
                assert(mappedY + span < width);
            }
        }
    }
}

}  // namespace

int main() {
    constexpr int layers = 3;
    constexpr int width = 5;
    constexpr int height = 5;
    constexpr int resourceCount = layers * width * height;
    std::vector<int> wireUsage(resourceCount, 0);
    std::vector<int> viaUsage(resourceCount, 0);
    std::vector<int> wireCost(resourceCount, 1);
    std::vector<int> viaCost(resourceCount, 1);
    std::vector<float> capacity(resourceCount, 10.0f);
    std::vector<float> fixedUsage(resourceCount, 0.0f);
    std::vector<uint8_t> blockedWire(resourceCount, 0);
    std::vector<uint8_t> blockedVia(resourceCount, 0);
    const gr::CpuMazeSearchView view = makeView(layers,
                                                width,
                                                height,
                                                wireUsage,
                                                viaUsage,
                                                wireCost,
                                                viaCost,
                                                capacity,
                                                fixedUsage,
                                                blockedWire,
                                                blockedVia);
    gr::CpuMazeRouter router;

    gr::CpuMazeNet direct{1, {{{0, 0, 0}}, {{0, 2, 0}}}};
    const auto directResult = router.route(direct, view);
    assert(directResult.succeeded());
    const std::vector<std::pair<int, int>> directWires{{0, 2}};
    assert(directResult.candidate.wireDeltas == directWires);
    assert(directResult.candidate.viaDeltas.empty());
    const std::vector<int> directRawRoute{3, 0, 2};
    assert(directResult.candidate.rawRoute == directRawRoute);
    assertValidRoute(directResult.candidate, layers, width, height);

    gr::CpuMazeNet viaNet{2, {{{0, 0, 0}}, {{1, 0, 2}}}};
    const auto viaResult = router.route(viaNet, view);
    assert(viaResult.succeeded());
    assert(viaResult.candidate.viaDeltas == std::vector<int>({0}));
    const std::vector<std::pair<int, int>> viaWires{{25, 2}};
    assert(viaResult.candidate.wireDeltas == viaWires);
    const std::vector<int> viaRawRoute{5, 0, -1, 25, 2};
    assert(viaResult.candidate.rawRoute == viaRawRoute);
    assertValidRoute(viaResult.candidate, layers, width, height);

    const auto repeated = router.route(viaNet, view);
    assert(repeated.candidate.rawRoute == viaResult.candidate.rawRoute);
    assert(repeated.candidate.wireDeltas == viaResult.candidate.wireDeltas);
    assert(repeated.candidate.viaDeltas == viaResult.candidate.viaDeltas);
    assert(repeated.expandedStates == viaResult.expandedStates);

    gr::CpuMazeNet accessNet{3, {{{0, 0, 0}, {1, 0, 0}}, {{1, 0, 2}}}};
    const auto accessResult = router.route(accessNet, view);
    assert(accessResult.succeeded());
    assert(accessResult.candidate.viaDeltas == std::vector<int>({0}));
    assertValidRoute(accessResult.candidate, layers, width, height);

    gr::CpuMazeSearchView transposedView = view;
    transposedView.direction = 1;
    gr::CpuMazeNet transposed{3, {{{0, 0, 0}}, {{0, 0, 2}}}};
    const auto transposedResult = router.route(transposed, transposedView);
    assert(transposedResult.succeeded());
    const std::vector<std::pair<int, int>> transposedWires{{0, 2}};
    assert(transposedResult.candidate.wireDeltas == transposedWires);
    assertValidRoute(transposedResult.candidate, layers, width, height, 1);

    std::vector<int> rectangularWireUsage(2 * 3 * 5, 0);
    std::vector<int> rectangularViaUsage(2 * 3 * 5, 0);
    std::vector<int> rectangularWireCost(2 * 3 * 5, 1);
    std::vector<int> rectangularViaCost(2 * 3 * 5, 1);
    std::vector<float> rectangularCapacity(2 * 3 * 5, 10.0f);
    std::vector<float> rectangularFixedUsage(2 * 3 * 5, 0.0f);
    std::vector<uint8_t> rectangularBlockedWire(2 * 3 * 5, 0);
    std::vector<uint8_t> rectangularBlockedVia(2 * 3 * 5, 0);
    const auto rectangularView = makeView(2,
                                          3,
                                          5,
                                          rectangularWireUsage,
                                          rectangularViaUsage,
                                          rectangularWireCost,
                                          rectangularViaCost,
                                          rectangularCapacity,
                                          rectangularFixedUsage,
                                          rectangularBlockedWire,
                                          rectangularBlockedVia);
    gr::CpuMazeNet rectangularNet{5, {{{0, 0, 0}}, {{0, 2, 0}}}};
    const auto rectangularResult = router.route(rectangularNet, rectangularView);
    assert(rectangularResult.succeeded());
    assert(rectangularResult.candidate.rawRoute == directRawRoute);
    assertValidRoute(rectangularResult.candidate, 2, 3, 5);

    gr::CpuMazeSearchView singleLayerView = view;
    singleLayerView.routingLayerEnd = 0;
    gr::CpuMazeNet layerLimited{6, {{{0, 0, 0}}, {{0, 0, 2}}}};
    const auto layerLimitedResult = router.route(layerLimited, singleLayerView);
    assert(!layerLimitedResult.succeeded());
    assert(layerLimitedResult.failureReason == gr::CpuMazeFailureReason::Unreachable);

    const std::vector<gr::CpuMazeNet> batchNets{direct, viaNet, accessNet};
    const auto serialBatch = router.routeBatch(batchNets, view, 1);
    assert(serialBatch.results.size() == batchNets.size());
    assert(serialBatch.effectiveWorkers == 1);
    for (int requestedWorkers : {2, 4, 8}) {
        const auto parallelBatch = router.routeBatch(batchNets, view, requestedWorkers);
        assert(parallelBatch.results.size() == batchNets.size());
        assert(parallelBatch.effectiveWorkers >= 1 &&
               parallelBatch.effectiveWorkers <= requestedWorkers);
        for (std::size_t index = 0; index < batchNets.size(); ++index) {
            assert(serialBatch.results[index].candidate.rawRoute ==
                   parallelBatch.results[index].candidate.rawRoute);
            assert(serialBatch.results[index].candidate.wireDeltas ==
                   parallelBatch.results[index].candidate.wireDeltas);
            assert(serialBatch.results[index].candidate.viaDeltas ==
                   parallelBatch.results[index].candidate.viaDeltas);
            assert(serialBatch.results[index].failureReason ==
                   parallelBatch.results[index].failureReason);
        }
    }

    std::fill(blockedWire.begin(), blockedWire.end(), 0);
    blockedWire[0] = 1;
    blockedWire[1] = 1;
    gr::CpuMazeNet detour{4, {{{0, 0, 0}}, {{0, 2, 0}}}};
    const auto detourResult = router.route(detour, view);
    assert(detourResult.succeeded());
    assert(detourResult.candidate.viaDeltas.size() >= 2);
    const std::pair<int, int> detourWire{50, 2};
    assert(std::find(detourResult.candidate.wireDeltas.begin(),
                     detourResult.candidate.wireDeltas.end(),
                     detourWire) != detourResult.candidate.wireDeltas.end());
    assertValidRoute(detourResult.candidate, layers, width, height);

    std::fill(blockedWire.begin(), blockedWire.end(), 1);
    const auto blocked = router.route(direct, view);
    assert(!blocked.succeeded());
    assert(blocked.failureReason == gr::CpuMazeFailureReason::Unreachable);
    assert(blocked.candidate.rawRoute == std::vector<int>({-1}));
    assert(gr::cpuMazeFailureReasonName(blocked.failureReason) == std::string("unreachable"));

    // Reusing scratch after a failure, a shape change and a cost/obstacle
    // change must be identical to a search with freshly initialized storage.
    gr::CpuMazeSearchScratch scratch;
    assert(router.route(direct, view, scratch).failureReason == blocked.failureReason);
    const auto resized = router.route(rectangularNet, rectangularView, scratch);
    assert(resized.candidate.rawRoute == rectangularResult.candidate.rawRoute);
    assert(resized.expandedStates == rectangularResult.expandedStates);
    std::fill(blockedWire.begin(), blockedWire.end(), 0);
    const auto reused = router.route(viaNet, view, scratch);
    const auto fresh = router.route(viaNet, view);
    assert(reused.candidate.rawRoute == fresh.candidate.rawRoute);
    assert(reused.expandedStates == fresh.expandedStates);

    gr::CpuMazeNet steiner{9, {{{0, 0, 0}}, {{0, 2, 0}}}};
    steiner.branches = {{{{0, 1, 0}, {1, 1, 0}, {2, 1, 0}}, false},
                        {{{0, 2, 0}}, true}};
    const auto steinerRoute = router.route(steiner, view);
    assert(steinerRoute.succeeded());
    assert(steinerRoute.candidate.viaDeltas.empty());
    assert(steinerRoute.candidate.rawRoute == directRawRoute);
    steiner.branches.pop_back();
    assert(router.route(steiner, view).failureReason == gr::CpuMazeFailureReason::Unreachable);
    return 0;
}
