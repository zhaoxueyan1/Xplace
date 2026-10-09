#pragma once

#include "common/common.h"
#include "gpugr/db/GRDatabase.h"
#include "gpugr/db/GrNet.h"
#include "gpugr/gr/CpuRouteBatchScheduler.h"
#include "gpugr/gr/CpuMazeRouter.h"
#include "gpugr/gr/CpuRouteState.h"
#include "gpugr/gr/CpuRouteTypes.h"
#include <torch/extension.h>
#include <tuple>
#include <vector>

namespace gr {

class CpuRouteWorkerPool;

class CPURouter {
public:
    void initialize(const gr::GRDatabase& grdb, const std::string& backendName = "cpu_pr");
    void setMap(const std::vector<float>& cap,
                const std::vector<float>& wir,
                const std::vector<float>& fixedL,
                const std::vector<float>& fix,
                const std::vector<float>& fixRaw,
                const std::vector<float>& movRaw);
    void route(std::vector<GrNet>& nets);
    void routeParallel(std::vector<GrNet>& nets, int requestedWorkers);
    void routeMazeRrr(std::vector<GrNet>& nets, int rrrIters, int requestedWorkers);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> getDemandMap();
    torch::Tensor getCapacityMap();
    torch::Tensor getRawWireDemandMap();
    torch::Tensor getFixUsageMap();
    torch::Tensor getMovUsageMap();
    int getNumOvflNets() const { return numOvflNets; }
    std::vector<std::vector<int>> getRawRoutes() const { return routeState.rawRoutes(); }
    const CpuRouteRunStats& getRunStats() const { return runStats; }

private:
    int encodedIndex(int layer, int x, int y) const;
    std::tuple<int, int, int> decodeIndex(int encoded) const;
    bool isRoutingLayerEnabled(int layer) const {
        return routingLayerBegin <= layer && layer <= routingLayerEnd;
    }
    float wireCost(int encoded) const;
    float viaCost(int encoded) const;
    int calcWireCost(int encoded) const;
    int calcViaCost(int encoded) const;
    void calculateCostMaps();
    void calculateCostSum();
    void calculateCostMapsParallel(CpuRouteWorkerPool& workerPool, int workers);
    void calculateCostSumParallel(CpuRouteWorkerPool& workerPool);
    void routePreparedNet(const std::vector<int>& points,
                          CpuRouteScratch& scratch,
                          CpuRouteCandidate& candidate,
                          bool resetOnlyRouteRegion) const;
    void viaSweep(std::vector<int64_t>& map,
                  std::vector<int>& prev,
                  std::vector<int>& midScratch,
                  int x,
                  int y) const;
    void appendWire(CpuRouteCandidate& candidate, int from, int to) const;
    void appendVia(CpuRouteCandidate& candidate, int encoded) const;
    CpuRouteBatchSchedule buildRoutingOrder(const std::vector<GrNet>& nets,
                                            int conflictScanLimit,
                                            int batchSizeLimit,
                                            bool packConflictTail) const;
    void commitCandidate(GrNet& net, const CpuRouteCandidate& candidate);
    bool batchHasResourceConflict(const std::vector<GrNet>& nets,
                                  const std::vector<CpuRouteCandidate>& candidates,
                                  std::size_t begin,
                                  std::size_t end) const;
    void setNetRouteFromRaw(GrNet& net, const std::vector<int>& rawRoute) const;
    struct OverflowReport {
        std::vector<int> netIds;
        CpuRouteRunStats::RrrIteration qor;
        int legacyOverflowNetCount = 0;
    };
    OverflowReport overflowReport;
    void markOverflowNets(const std::vector<GrNet>& nets);
    CpuMazeSearchView mazeSearchView() const;
    CpuMazeNet mazeNet(GrNet& net, int netId) const;
    float twoCellsViaUsage(int encoded) const;
    float inCellViaUsage(int encoded) const;
    float inCellUsedArea(int encoded) const;
    float inCellViaCost(int encoded) const;
    torch::Tensor buildGridTensor(const std::vector<float>& values) const;
    torch::Tensor buildGridTensor(const std::vector<int>& values) const;
    [[noreturn]] void unsupported(const std::string& operation) const;

    int LAYER = 0;
    int N = 0;
    int X = 0;
    int Y = 0;
    int DIRECTION = 0;
    int routingLayerBegin = 0;
    int routingLayerEnd = -1;
    std::vector<float> capacity;
    std::vector<float> wireDist;
    std::vector<float> fixedLength;
    std::vector<float> fixed;
    std::vector<float> fixUsage;
    std::vector<float> movUsage;
    std::vector<int> cost;
    std::vector<int> viaCostMap;
    std::vector<int64_t> costSum;
    std::vector<int> overflowWire;
    std::vector<int> overflowVia;
    std::vector<uint8_t> mazeBlockedWire;
    std::vector<uint8_t> mazeBlockedVia;
    std::vector<double> unitShortCost;
    std::vector<float> unitShortCostDiscounted;
    float unitViaCost = 4.0f;
    float unitViaMultiplier = 1.0f;
    float logisticSlope = 1.0f;
    int mazeHorizontalLowerBound = 1;
    int mazeVerticalLowerBound = 1;
    int mazeViaLowerBound = 1;
    int mazeCoarseningScale = 1;
    int numOvflNets = 0;
    CpuRouteState routeState;
    CpuRouteRunStats runStats;
    std::string diagnosticBackend = "cpu_pr";
};

}  // namespace gr
