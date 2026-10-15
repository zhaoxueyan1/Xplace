#include "CPURouter.h"
#include "CpuMazeCostPolicy.h"
#include "CpuMazeCoarseGrid.h"
#include "CpuMazeBatch.h"
#include "CpuMazeCostUpdate.h"
#include "PatternRoute.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <stdexcept>

namespace gr {
namespace {
using SteadyClock = std::chrono::steady_clock;
constexpr int INF = 10000000;
double elapsedSeconds(const SteadyClock::time_point& start) {
    return std::chrono::duration<double>(SteadyClock::now() - start).count();
}
}  // namespace

CpuMazeSearchView CPURouter::mazeSearchView() const {
    CpuMazeSearchView view;
    view.layerCount = LAYER;
    view.xSize = N;
    view.ySize = N;
    view.direction = DIRECTION;
    view.routingLayerBegin = routingLayerBegin;
    view.routingLayerEnd = routingLayerEnd;
    view.wireUsage = &routeState.wires();
    view.viaUsage = &routeState.vias();
    view.wireCost = &cost;
    view.viaCost = &viaCostMap;
    view.capacity = &capacity;
    view.fixedUsage = &fixed;
    // Native cost maps already include the CUDA-compatible congestion
    // schedule. The standalone maze fixture's extra penalty is not added.
    view.overflowPenalty = 0;
    view.blockedWire = &mazeBlockedWire;
    view.blockedVia = &mazeBlockedVia;
    view.horizontalWireLowerBound = mazeHorizontalLowerBound;
    view.verticalWireLowerBound = mazeVerticalLowerBound;
    view.viaLowerBound = mazeViaLowerBound;
    return view;
}

CpuMazeNet CPURouter::mazeNet(GrNet& net, int netId) const {
    CpuMazeNet maze;
    maze.netId = netId;
    std::map<std::pair<int, int>, std::vector<CpuMazePoint>> accessByLocation;
    for (const auto& pinGroup : net.getPins()) {
        std::vector<CpuMazePoint> group;
        group.reserve(pinGroup.size());
        for (int encoded : pinGroup) {
            const auto [layer, x, y] = decodeIndex(encoded);
            group.push_back({layer, x, y});
            accessByLocation[{x, y}].push_back({layer, x, y});
        }
        if (!group.empty()) {
            maze.pinGroups.push_back(std::move(group));
        }
    }
    // Reuse the initial pattern router's prepared Steiner DFS order. Real
    // access groups require their local layer ladder; a Steiner point can
    // connect on any legal layer and must not create an artificial ladder.
    const std::size_t pinCount = std::max<std::size_t>(1, net.getPins().size());
    std::vector<int> points(48 * pinCount + 2, 0), gbpoints(4 * pinCount, -1);
    int pinOffset = 0;
    double prepareSeconds = 0;
    const int length = prepare(prepareSeconds, net, points.data(), 0,
                               gbpoints.data(), pinOffset, points.size(), gbpoints.size(),
                               X, Y, N, LAYER, DIRECTION);
    for (int index = 2; index + 5 < length; index += 6) {
        const int x = points[index] / N, y = points[index] % N;
        CpuMazeBranch branch;
        const auto access = accessByLocation.find({x, y});
        if (access != accessByLocation.end()) {
            branch.targets = access->second;
            branch.pinAccess = true;
        } else {
            for (int layer = routingLayerBegin; layer <= routingLayerEnd; ++layer) {
                branch.targets.push_back({layer, x, y});
            }
        }
        maze.branches.push_back(std::move(branch));
    }
    return maze;
}

void CPURouter::routeMazeRrr(std::vector<GrNet>& nets, int rrrIters, int requestedWorkers) {
    if (rrrIters < 0) {
        throw std::invalid_argument("backend=" + diagnosticBackend +
                                    " operation=run_ggr rrr_iters must be non-negative");
    }
    const auto totalStart = SteadyClock::now();
    // The maze lane deliberately uses the existing deterministic parallel PR as
    // its initial state.  This also makes rrr_iters=0 an exact compatibility
    // path for cpu_pr_mt.
    routeParallel(nets, requestedWorkers);
    runStats.requestedRrrIters = rrrIters;
    runStats.completedRrrIters = 0;
    runStats.bestStateRestored = false;
    runStats.mazeRouteFailureCount = 0;
    runStats.conflictFallbackCount = 0;
    runStats.memoryAccounting = "conservative_router_storage_estimate_not_rss";
    runStats.peakMemoryBytes = routeState.bytes() + cost.size() * 64 +
                               runStats.scratchBytesPerWorker *
                               static_cast<std::size_t>(std::max(1, runStats.effectiveWorkers));
    runStats.mazeFailureReasons.clear();
    runStats.rrrIterations.clear();

    auto metric = [](const OverflowReport& report) {
        const auto& qor = report.qor;
        return std::tuple<double, int, int, int>(qor.overflowWireExcess + qor.overflowViaExcess,
                                                 qor.overflowWireResourceCount + qor.overflowViaResourceCount,
                                                 qor.wireLength, qor.viaCount);
    };
    runStats.hasRouteQor = true;
    runStats.initialRouteQor = overflowReport.qor;
    runStats.acceptedRouteQor = runStats.initialRouteQor;
    if (rrrIters == 0) {
        runStats.totalSeconds = elapsedSeconds(totalStart);
        return;
    }
    mazeBlockedWire.assign(cost.size(), 1);
    mazeBlockedVia.assign(viaCostMap.size(), 1);
    for (int layer = routingLayerBegin; layer <= routingLayerEnd; ++layer) {
        for (int x = 0; x < X; ++x) {
            for (int y = 0; y < Y; ++y) {
                const int resource = encodedIndex(layer, x, y);
                const bool vertical = ((layer & 1) ^ DIRECTION) != 0;
                if ((vertical ? y + 1 < Y : x + 1 < X) && capacity[resource] >= 0.01f) {
                    mazeBlockedWire[resource] = 0;
                }
                if (layer < routingLayerEnd) {
                    // Capacity is wire-track capacity, not via legality. Match
                    // CUDA's finite via cost even in zero-track pin-access cells.
                    // Layer and physical-grid restrictions still apply.
                    mazeBlockedVia[resource] = 0;
                }
            }
        }
    }
    auto bestMetric = metric(overflowReport);
    CpuMazeRouter mazeRouter;
    CpuMazeSearchScratch fallbackScratch;
    runStats.mazeBatchSizeLimit = CPU_MAZE_BATCH_SIZE;

    for (int iter = 1; iter <= rrrIters; ++iter) {
        const auto iterationStart = SteadyClock::now();
        OverflowReport before = overflowReport;
        std::sort(before.netIds.begin(), before.netIds.end(), [&](int lhs, int rhs) {
            const int lhsArea = nets[lhs].area();
            const int rhsArea = nets[rhs].area();
            if (lhsArea != rhsArea) return lhsArea > rhsArea;
            return lhs < rhs;
        });
        if (before.netIds.empty()) {
            break;
        }
        logger.info("CPU maze RRR iteration=%d selected_nets=%zu", iter, before.netIds.size());

        std::vector<CpuRouteSnapshot> oldRoutes;
        oldRoutes.reserve(before.netIds.size());
        for (int netId : before.netIds) {
            oldRoutes.push_back(routeState.remove(netId));
            setNetRouteFromRaw(nets[netId], {1});
        }

        const auto costStart = SteadyClock::now();
        const auto policy = CpuMazeCostPolicy::forIteration(iter, rrrIters);
        logisticSlope = static_cast<float>(policy.logisticSlope);
        unitViaMultiplier = static_cast<float>(policy.viaMultiplier);
        for (int layer = 0; layer < LAYER; ++layer) {
            unitShortCostDiscounted[layer] = static_cast<float>(
                unitShortCost[layer] * policy.shortViolationDiscount);
        }
        calculateCostMaps();
        mazeHorizontalLowerBound = mazeVerticalLowerBound = mazeViaLowerBound = INF;
        for (int resource = 0; resource < LAYER * N * N; ++resource) {
            const int layer = resource / N / N;
            if (!isRoutingLayerEnabled(layer)) continue;
            auto& wireBound = ((layer & 1) ^ DIRECTION)
                                  ? mazeVerticalLowerBound : mazeHorizontalLowerBound;
            if (!mazeBlockedWire[resource]) wireBound = std::min(wireBound, std::max(1, cost[resource]));
            if (!mazeBlockedVia[resource]) {
                mazeViaLowerBound = std::min(mazeViaLowerBound, std::max(1, viaCostMap[resource]));
            }
        }
        double costSeconds = elapsedSeconds(costStart);

        CpuMazeCoarseGrid coarseGrid(mazeSearchView(), mazeCoarseningScale, X, Y);
        auto guidedView = [&]() {
            auto view = mazeSearchView();
            view.coarseGrid = &coarseGrid;
            return view;
        };
        CpuMazeBatchRouter batchRouter(guidedView(), requestedWorkers, CPU_MAZE_BATCH_SIZE);
        auto sharedBytes = [&]() {
            auto bytes = [](const auto& values) { return values.capacity() * sizeof(values.front()); };
            // Include current/saved routes, changed-resource flags, coarse
            // search storage and the serial fallback scratch reservation.
            return routeState.bytes() + cpuRouteSnapshotsBytes(oldRoutes) +
                   bytes(capacity) + bytes(wireDist) + bytes(fixedLength) + bytes(fixed) +
                   bytes(fixUsage) + bytes(movUsage) + bytes(cost) + bytes(viaCostMap) +
                   bytes(costSum) + bytes(overflowWire) + bytes(overflowVia) +
                   bytes(mazeBlockedWire) + bytes(mazeBlockedVia) +
                   bytes(unitShortCost) + bytes(unitShortCostDiscounted) +
                   cost.size() * (2 + 256) + nets.size() * 16 +
                   coarseGrid.nodeCount() * 64;
        };
        double searchSeconds = 0.0, workerSearchSeconds = 0.0;
        double coarseBuildSeconds = 0.0;
        CpuMazeSearchStats searchStats;
        auto collectSearchStats = [&](const CpuMazeSearchResult& result) {
            searchStats.coarseExpandedStates += result.searchStats.coarseExpandedStates;
            searchStats.fineExpandedStates += result.searchStats.fineExpandedStates;
            searchStats.corridorRetries += result.searchStats.corridorRetries;
            searchStats.fullSearchFallbacks += result.searchStats.fullSearchFallbacks;
        };
        int parallelBatches = 0, maxActiveSearches = 0;

        std::vector<uint8_t> changedWireCost(cost.size(), 0);
        std::vector<uint8_t> changedViaCost(viaCostMap.size(), 0);
        auto refreshCosts = [&](const auto& route) {
            const auto start = SteadyClock::now();
            const auto update = cpuMazeCostUpdate(route.wireDeltas, route.viaDeltas, LAYER, N);
            for (int resource : update.wires) {
                const int next = calcWireCost(resource);
                if (next != cost[resource]) {
                    changedWireCost[resource] = 1;
                    coarseGrid.invalidate(resource);
                }
                cost[resource] = next;
            }
            for (int resource : update.vias) {
                const int next = calcViaCost(resource);
                if (next != viaCostMap[resource]) {
                    changedViaCost[resource] = 1;
                    coarseGrid.invalidate(resource);
                }
                viaCostMap[resource] = next;
            }
            costSeconds += elapsedSeconds(start);
        };
        auto candidateIsStale = [&](const CpuRouteCandidate& candidate) {
            for (const auto& [start, span] : candidate.wireDeltas) {
                for (int resource = start; resource < start + span; ++resource) {
                    if (changedWireCost[resource]) return true;
                }
            }
            for (int resource : candidate.viaDeltas) {
                if (changedViaCost[resource]) return true;
            }
            return false;
        };
        int rerouted = 0;
        int failed = 0;
        int conflictFallbacks = 0;
        double commitSeconds = 0.0;
        double rollbackSeconds = 0.0;
        double fallbackSearchSeconds = 0.0;
        for (std::size_t begin = 0; begin < before.netIds.size(); begin += CPU_MAZE_BATCH_SIZE) {
            const std::size_t end = std::min(before.netIds.size(), begin + CPU_MAZE_BATCH_SIZE);
            std::vector<CpuMazeNet> mazeNets;
            mazeNets.reserve(end - begin);
            for (std::size_t index = begin; index < end; ++index)
                mazeNets.push_back(mazeNet(nets[before.netIds[index]], before.netIds[index]));
            std::fill(changedWireCost.begin(), changedWireCost.end(), 0);
            std::fill(changedViaCost.begin(), changedViaCost.end(), 0);
            const auto coarseStart = SteadyClock::now();
            coarseGrid.refresh(mazeSearchView());
            coarseBuildSeconds += elapsedSeconds(coarseStart);
            const auto searchStart = SteadyClock::now();
            auto batch = batchRouter.search(mazeNets, guidedView(), sharedBytes());
            searchSeconds += elapsedSeconds(searchStart);
            workerSearchSeconds += batch.workerSearchSeconds;
            maxActiveSearches = std::max(maxActiveSearches, batch.maxActiveSearches);
            if (batch.effectiveWorkers > 1 && mazeNets.size() > 1) ++parallelBatches;
            runStats.effectiveWorkers = batch.effectiveWorkers;
            runStats.workerClampReason = batch.workerClampReason;
            runStats.scratchBytesPerWorker = std::max(runStats.scratchBytesPerWorker, batch.scratchBytesPerWorker);
            runStats.memoryBudgetBytes = batch.memoryBudgetBytes;
            runStats.mazeMaxBatchSize = std::max(runStats.mazeMaxBatchSize, static_cast<int>(mazeNets.size()));
            runStats.peakMemoryBytes = std::max(runStats.peakMemoryBytes,
                batchRouter.reservedBytes() + batch.scratchBytesPerWorker * batch.effectiveWorkers);
            for (std::size_t index = begin; index < end; ++index) {
                const int netId = before.netIds[index];
                CpuMazeSearchResult result = std::move(batch.results[index - begin]);
                collectSearchStats(result);
                if (!result.succeeded() || candidateIsStale(result.candidate)) {
                    ++conflictFallbacks;
                    const auto fallbackStart = SteadyClock::now();
                    result = mazeRouter.route(mazeNets[index - begin], guidedView(), fallbackScratch);
                    collectSearchStats(result);
                    fallbackSearchSeconds += elapsedSeconds(fallbackStart);
                }
                // Re-search observes current committed usage. Sharing a gcell
                // resource is legal in global routing and is priced by its cost;
                // it must not be rejected as another stale-snapshot conflict.
                if (!result.succeeded()) {
                    ++failed;
                    ++runStats.mazeRouteFailureCount;
                    ++runStats.mazeFailureReasons[cpuMazeFailureReasonName(result.failureReason)];
                    const auto rollbackStart = SteadyClock::now();
                    routeState.restore(netId, oldRoutes[index]);
                    setNetRouteFromRaw(nets[netId], oldRoutes[index].rawRoute);
                    refreshCosts(oldRoutes[index]);
                    rollbackSeconds += elapsedSeconds(rollbackStart);
                    continue;
                }
                const auto commitStart = SteadyClock::now();
                routeState.commit(result.candidate);
                setNetRouteFromRaw(nets[netId], result.candidate.rawRoute);
                commitSeconds += elapsedSeconds(commitStart);
                ++rerouted;
                refreshCosts(result.candidate);
            }
        }

        markOverflowNets(nets);
        OverflowReport after = overflowReport;
        CpuRouteRunStats::RrrIteration iteration = after.qor;
        iteration.iteration = iter;
        iteration.selectedNetCount = static_cast<int>(before.netIds.size());
        iteration.reroutedNetCount = rerouted;
        iteration.failedNetCount = failed;
        iteration.parallelBatchCount = parallelBatches;
        iteration.conflictFallbackCount = conflictFallbacks;
        iteration.searchSeconds = searchSeconds;
        iteration.fallbackSearchSeconds = fallbackSearchSeconds;
        iteration.maxActiveSearches = maxActiveSearches;
        iteration.workerSearchSeconds = workerSearchSeconds;
        iteration.coarseBuildSeconds = coarseBuildSeconds;
        iteration.mazeCoarseningScale = mazeCoarseningScale;
        iteration.mazeCoarseXSize = 1 + (X - 1) / mazeCoarseningScale;
        iteration.mazeCoarseYSize = 1 + (Y - 1) / mazeCoarseningScale;
        iteration.coarseExpandedStates = searchStats.coarseExpandedStates;
        iteration.fineExpandedStates = searchStats.fineExpandedStates;
        iteration.corridorRetries = searchStats.corridorRetries;
        iteration.fullSearchFallbacks = searchStats.fullSearchFallbacks;
        iteration.costRebuildSeconds = costSeconds;
        iteration.commitSeconds = commitSeconds;
        iteration.rollbackSeconds = rollbackSeconds;
        iteration.totalSeconds = elapsedSeconds(iterationStart);
        ++runStats.completedRrrIters;
        runStats.conflictFallbackCount += conflictFallbacks;

        if (metric(after) < bestMetric) {
            bestMetric = metric(after);
            iteration.bestState = true;
        }
        runStats.rrrIterations.push_back(iteration);
        runStats.acceptedRouteQor = iteration;
        logger.info("CPU maze commit iteration=%d rerouted=%d fallback=%d fallback_seconds=%.4f best=%d",
                    iter, rerouted, conflictFallbacks, fallbackSearchSeconds, iteration.bestState);
        // Later passes use different congestion/via penalties. A non-improving
        // intermediate pass must not skip the remaining cost schedule.
        if (after.netIds.empty()) {
            break;
        }
    }

    // GR parasitic evaluation uses the completed route, as the CUDA backend
    // does. bestState is diagnostic only; intermediate overflow minima do not
    // replace the result of the scheduled final congestion/via penalties.
    runStats.routeHash64 = hashRawRoutes(routeState.routes());
    runStats.wireMapHash64 = hashIntVector(routeState.wires());
    runStats.viaMapHash64 = hashIntVector(routeState.vias());
    runStats.totalSeconds = elapsedSeconds(totalStart);
    runStats.terminalStatus = "completed";
}

}  // namespace gr
