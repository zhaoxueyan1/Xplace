#include "CPURouter.h"
#include "CpuMazeCostPolicy.h"
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
    view.impassableCost = INF;
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
    runStats.peakMemoryBytes = runStats.scratchBytesPerWorker *
                               static_cast<std::size_t>(std::max(1, runStats.effectiveWorkers));
    runStats.mazeFailureReasons.clear();
    runStats.rrrIterations.clear();

    struct OverflowReport {
        std::vector<int> netIds;
        int wireResourceCount = 0;
        int viaResourceCount = 0;
        double wireExcess = 0.0;
        double viaExcess = 0.0;
        int routedNetCount = 0;
        int wireLength = 0;
        int viaCount = 0;
        uint64_t routeHash64 = 0;
        uint64_t wireMapHash64 = 0;
        uint64_t viaMapHash64 = 0;
    };

    auto makeReport = [&]() {
        OverflowReport report;
        for (int idx = 0; idx < LAYER * N * N; ++idx) {
            if (overflowWire[idx] != 0) {
                ++report.wireResourceCount;
                report.wireExcess += std::max(
                    0.0,
                    static_cast<double>(routeState.wires()[idx]) + fixed[idx] +
                        twoCellsViaUsage(idx) - capacity[idx]);
            }
        }
        for (int idx = 0; idx < (LAYER - 1) * N * N; ++idx) {
            if (overflowVia[idx] != 0) {
                ++report.viaResourceCount;
                const int layer = idx / N / N + 1;
                const int y = idx % (N * N) / N;
                const int x = idx % N;
                const int upper = layer * N * N + x * N + y;
                report.viaExcess += std::max(
                    0.0,
                    std::max(static_cast<double>(inCellUsedArea(idx) - capacity[idx]),
                             static_cast<double>(inCellUsedArea(upper) - capacity[upper])));
            }
        }
        for (int netId = 0; netId < static_cast<int>(nets.size()); ++netId) {
            const auto& net = nets[netId];
            const auto& rawRoute = routeState.rawRoutes()[netId];
            if (!net.noroute && net.exclusion == NetExclusion::None &&
                !rawRoute.empty() && rawRoute[0] > 0) {
                ++report.routedNetCount;
            }
            if (net.noroute || net.exclusion != NetExclusion::None || rawRoute.empty() ||
                rawRoute[0] <= 0) {
                continue;
            }
            const int end = std::min(static_cast<int>(rawRoute.size()), rawRoute[0]);
            bool overflow = false;
            for (int i = 1; i + 1 < end && !overflow; i += 2) {
                const int point = rawRoute[i];
                const int span = rawRoute[i + 1];
                if (span == -1) {
                    overflow = point >= 0 && point < static_cast<int>(overflowVia.size()) &&
                               overflowVia[point] != 0;
                } else if (span > 0) {
                    for (int offset = 0; offset < span; ++offset) {
                        const int resource = point + offset;
                        if (resource >= 0 && resource < static_cast<int>(overflowWire.size()) &&
                            overflowWire[resource] != 0) {
                            overflow = true;
                            break;
                        }
                    }
                }
            }
            if (overflow) {
                report.netIds.push_back(netId);
            }
            for (int i = 1; i + 1 < end; i += 2) {
                if (rawRoute[i + 1] == -1) {
                    ++report.viaCount;
                } else if (rawRoute[i + 1] > 0) {
                    report.wireLength += rawRoute[i + 1];
                }
            }
        }
        report.routeHash64 = hashRawRoutes(routeState.rawRoutes());
        report.wireMapHash64 = hashIntVector(routeState.wires());
        report.viaMapHash64 = hashIntVector(routeState.vias());
        return report;
    };

    auto metric = [](const OverflowReport& report) {
        return std::tuple<double, int, int, int>(report.wireExcess + report.viaExcess,
                                                 report.wireResourceCount + report.viaResourceCount,
                                                 report.wireLength,
                                                 report.viaCount);
    };

    OverflowReport bestReport = makeReport();
    auto reportQor = [](const OverflowReport& report) {
        CpuRouteRunStats::RrrIteration qor;
        qor.overflowNetCount = static_cast<int>(report.netIds.size());
        qor.overflowWireResourceCount = report.wireResourceCount;
        qor.overflowViaResourceCount = report.viaResourceCount;
        qor.overflowWireExcess = report.wireExcess;
        qor.overflowViaExcess = report.viaExcess;
        qor.wireLength = report.wireLength;
        qor.viaCount = report.viaCount;
        qor.routedNetCount = report.routedNetCount;
        qor.routeHash64 = report.routeHash64;
        qor.wireMapHash64 = report.wireMapHash64;
        qor.viaMapHash64 = report.viaMapHash64;
        return qor;
    };
    runStats.hasRouteQor = true;
    runStats.initialRouteQor = reportQor(bestReport);
    runStats.acceptedRouteQor = runStats.initialRouteQor;
    if (rrrIters == 0) {
        runStats.totalSeconds = elapsedSeconds(totalStart);
        return;
    }
    auto checkpoint = routeState.checkpoint();
    auto bestMetric = metric(bestReport);
    CpuMazeRouter mazeRouter;
    CpuMazeSearchScratch fallbackScratch;

    for (int iter = 1; iter <= rrrIters; ++iter) {
        const auto iterationStart = SteadyClock::now();
        OverflowReport before = makeReport();
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
            if (cost[resource] < INF) wireBound = std::min(wireBound, std::max(1, cost[resource]));
            if (layer < routingLayerEnd && viaCostMap[resource] < INF) {
                mazeViaLowerBound = std::min(mazeViaLowerBound, std::max(1, viaCostMap[resource]));
            }
        }
        double costSeconds = elapsedSeconds(costStart);

        std::vector<CpuMazeNet> mazeNets;
        mazeNets.reserve(before.netIds.size());
        for (int netId : before.netIds) mazeNets.push_back(mazeNet(nets[netId], netId));
        const auto searchStart = SteadyClock::now();
        const CpuMazeBatchResult batch = mazeRouter.routeBatch(mazeNets, mazeSearchView(), requestedWorkers);
        const double searchSeconds = elapsedSeconds(searchStart);
        logger.info("CPU maze batch search iteration=%d seconds=%.4f effective_workers=%d max_active=%d",
                    iter, searchSeconds, batch.effectiveWorkers, batch.maxActiveSearches);
        runStats.effectiveWorkers = batch.effectiveWorkers;
        runStats.workerClampReason = batch.workerClampReason;
        runStats.scratchBytesPerWorker = std::max(runStats.scratchBytesPerWorker, batch.scratchBytesPerWorker);
        runStats.memoryBudgetBytes = std::max(runStats.memoryBudgetBytes, batch.memoryBudgetBytes);
        runStats.peakMemoryBytes = std::max(
            runStats.peakMemoryBytes,
            batch.scratchBytesPerWorker * static_cast<std::size_t>(std::max(1, batch.effectiveWorkers)));

        std::vector<uint8_t> changedWireCost(cost.size(), 0);
        std::vector<uint8_t> changedViaCost(viaCostMap.size(), 0);
        auto refreshCosts = [&](const auto& route) {
            const auto start = SteadyClock::now();
            const auto update = cpuMazeCostUpdate(route.wireDeltas, route.viaDeltas, LAYER, N);
            for (int resource : update.wires) {
                cost[resource] = calcWireCost(resource);
                changedWireCost[resource] = 1;
            }
            for (int resource : update.vias) {
                viaCostMap[resource] = calcViaCost(resource);
                changedViaCost[resource] = 1;
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
        bool stateInvalidated = false;
        for (std::size_t index = 0; index < before.netIds.size(); ++index) {
            const int netId = before.netIds[index];
            CpuMazeSearchResult result = batch.results[index];
            if (stateInvalidated || !result.succeeded() || candidateIsStale(result.candidate)) {
                ++conflictFallbacks;
                const auto fallbackStart = SteadyClock::now();
                result = mazeRouter.route(mazeNets[index], mazeSearchView(), fallbackScratch);
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
                stateInvalidated = true;
                continue;
            }
            const auto commitStart = SteadyClock::now();
            routeState.commit(result.candidate);
            setNetRouteFromRaw(nets[netId], result.candidate.rawRoute);
            commitSeconds += elapsedSeconds(commitStart);
            ++rerouted;
            refreshCosts(result.candidate);
        }

        markOverflowNets(nets);
        OverflowReport after = makeReport();
        CpuRouteRunStats::RrrIteration iteration = reportQor(after);
        iteration.iteration = iter;
        iteration.selectedNetCount = static_cast<int>(before.netIds.size());
        iteration.reroutedNetCount = rerouted;
        iteration.failedNetCount = failed;
        iteration.parallelBatchCount = batch.effectiveWorkers > 1 && batch.results.size() > 1 ? 1 : 0;
        iteration.conflictFallbackCount = conflictFallbacks;
        iteration.searchSeconds = searchSeconds;
        iteration.fallbackSearchSeconds = fallbackSearchSeconds;
        iteration.maxActiveSearches = batch.maxActiveSearches;
        iteration.workerSearchSeconds = batch.workerSearchSeconds;
        iteration.costRebuildSeconds = costSeconds;
        iteration.commitSeconds = commitSeconds;
        iteration.rollbackSeconds = rollbackSeconds;
        iteration.totalSeconds = elapsedSeconds(iterationStart);
        ++runStats.completedRrrIters;
        runStats.conflictFallbackCount += conflictFallbacks;

        if (metric(after) < bestMetric) {
            bestMetric = metric(after);
            bestReport = after;
            checkpoint = routeState.checkpoint();
            iteration.bestState = true;
        }
        runStats.rrrIterations.push_back(iteration);
        logger.info("CPU maze commit iteration=%d rerouted=%d failed=%d fallback=%d fallback_seconds=%.4f best=%d",
                    iter, rerouted, failed, conflictFallbacks, fallbackSearchSeconds, iteration.bestState);
        if (!iteration.bestState) {
            break;
        }
        if (after.netIds.empty()) {
            break;
        }
    }

    const OverflowReport finalReport = makeReport();
    if (metric(finalReport) != bestMetric || finalReport.routeHash64 != bestReport.routeHash64) {
        routeState.restoreCheckpoint(checkpoint);
        for (int netId = 0; netId < static_cast<int>(nets.size()); ++netId) {
            setNetRouteFromRaw(nets[netId], routeState.rawRoutes()[netId]);
        }
        markOverflowNets(nets);
        runStats.bestStateRestored = true;
    }
    runStats.routeHash64 = hashRawRoutes(routeState.rawRoutes());
    runStats.wireMapHash64 = hashIntVector(routeState.wires());
    runStats.viaMapHash64 = hashIntVector(routeState.vias());
    runStats.acceptedRouteQor = reportQor(bestReport);
    runStats.totalSeconds = elapsedSeconds(totalStart);
    runStats.terminalStatus = "completed";
}

}  // namespace gr
