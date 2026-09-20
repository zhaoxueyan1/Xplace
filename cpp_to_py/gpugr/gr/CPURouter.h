#pragma once

#include "common/common.h"
#include "gpugr/db/GRDatabase.h"
#include "gpugr/db/GrNet.h"
#include <torch/extension.h>
#include <tuple>
#include <vector>

namespace gr {

class CPURouter {
public:
    void initialize(const gr::GRDatabase& grdb);
    void setMap(const std::vector<float>& cap,
                const std::vector<float>& wir,
                const std::vector<float>& fixedL,
                const std::vector<float>& fix,
                const std::vector<float>& fixRaw,
                const std::vector<float>& movRaw);
    void route(std::vector<GrNet>& nets);
    void loadCanonicalRoutes(const std::vector<GrNet>& nets,
                             const std::vector<std::vector<int>>& canonicalRawRoutes);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> getDemandMap();
    torch::Tensor getCapacityMap();
    torch::Tensor getRawWireDemandMap();
    torch::Tensor getFixUsageMap();
    torch::Tensor getMovUsageMap();
    int getNumOvflNets() const { return numOvflNets; }
    std::vector<std::vector<int>> getRawRoutes() const { return rawRoutes; }

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
    void routePreparedNet(const std::vector<int>& points, std::vector<int>& rawRoute);
    void viaSweep(std::vector<int64_t>& map,
                  std::vector<int>& prev,
                  std::vector<int>& midScratch,
                  int x,
                  int y) const;
    void appendWire(std::vector<int>& rawRoute, int from, int to);
    void appendVia(std::vector<int>& rawRoute, int encoded);
    std::vector<int> buildRoutingOrder(const std::vector<GrNet>& nets, std::vector<int>& batchSizes) const;
    void setNetRouteFromRaw(GrNet& net, const std::vector<int>& rawRoute) const;
    void markOverflowNets(const std::vector<GrNet>& nets);
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
    std::vector<int> wires;
    std::vector<int> vias;
    std::vector<int> cost;
    std::vector<int> viaCostMap;
    std::vector<int64_t> costSum;
    std::vector<int> overflowWire;
    std::vector<int> overflowVia;
    std::vector<float> unitShortCostDiscounted;
    float unitViaCost = 4.0f;
    float unitViaMultiplier = 1.0f;
    float logisticSlope = 1.0f;
    int numOvflNets = 0;
    std::vector<std::vector<int>> rawRoutes;
};

}  // namespace gr
