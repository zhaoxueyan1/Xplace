#include "CUGR2Router.h"

#include "common/db/Database.h"
#include "common/db/Layer.h"
#include "common/db/Net.h"
#include "gr/Cugr2Bridge.h"
#include "flute.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace gr {

namespace {

std::string resolveLutPath(const char* name) {
    const char* root = std::getenv("XPLACE_FLUTE_LUT_ROOT");
    if (root != nullptr && root[0] != '\0') return std::string(root) + "/" + name;
    return std::string("thirdparty/flute/") + name;
}

void xplaceFluteInit(const std::string& powvFile, const std::string& postFile) {
    Flute::readLUT(powvFile.empty() ? resolveLutPath("POWV9.dat") : powvFile,
                   postFile.empty() ? resolveLutPath("POST9.dat") : postFile);
}

cugr2_bridge::FluteTree xplaceFluteRoute(int degree, const int* xs, const int* ys, int accuracy) {
    std::vector<int> xValues(xs, xs + degree);
    std::vector<int> yValues(ys, ys + degree);
    Flute::Tree tree = Flute::flute(degree, xValues.data(), yValues.data(), accuracy);
    cugr2_bridge::FluteTree result;
    result.branches.reserve(static_cast<size_t>(std::max(0, tree.deg * 2 - 2)));
    for (int index = 0; index < tree.deg * 2 - 2; ++index) {
        result.branches.push_back({tree.branch[index].x, tree.branch[index].y, tree.branch[index].n});
    }
    free(tree.branch);
    return result;
}

cugr2_bridge::Grid buildGrid(const GRDatabase& grdb) {
    cugr2_bridge::Grid grid;
    grid.libDbu = grdb.microns;
    grid.m2Pitch = static_cast<std::int64_t>(grdb.m2pitch);
    grid.gridlines.resize(2);
    grid.gridlines[0].assign(grdb.gridlines[0].begin(), grdb.gridlines[0].end());
    grid.gridlines[1].assign(grdb.gridlines[1].begin(), grdb.gridlines[1].end());
    grid.nMaxGrid = grdb.nMaxGrid;
    grid.m1Direction = grdb.m1direction;
    grid.bottomRoutingLayer = grdb.routingLayerBegin;
    grid.topRoutingLayer = grdb.routingLayerEnd;
    grid.layerNames.reserve(grdb.nLayers);
    grid.layerDirections.reserve(grdb.nLayers);
    grid.layerMinLengths.reserve(grdb.nLayers);
    grid.unitLengthShortCosts.resize(grdb.nLayers, 0.0);

    const double pitch = std::max(1.0, grdb.m2pitch);
    const double unitWireCost = 0.5 * std::max(1, grdb.microns) / pitch;
    grid.unitLengthWireCost = unitWireCost;
    grid.unitViaCost = 4.0 / unitWireCost * std::max(1, grdb.microns);
    for (int layer = 0; layer < grdb.nLayers; ++layer) {
        const db::Layer* routingLayer = grdb.rawdb.getRLayer(layer);
        grid.layerNames.emplace_back(routingLayer->name());
        grid.layerDirections.emplace_back(routingLayer->direction == 'h' ? 0 : 1);
        const double width = std::max(1, grdb.layerWidth[layer]);
        const double area = routingLayer->area > 0 ? routingLayer->area : width * width;
        grid.layerMinLengths.emplace_back(static_cast<std::int64_t>(std::max(0.0, area / width - width)));
        grid.unitLengthShortCosts[layer] =
            500.0 * width * std::max(1, grdb.microns) / pitch / pitch / unitWireCost;
    }

    grid.edges.assign(grdb.nLayers,
                      std::vector<std::vector<cugr2_bridge::GraphEdge>>(
                          grdb.xSize, std::vector<cugr2_bridge::GraphEdge>(grdb.ySize)));
    for (int layer = 0; layer < grdb.nLayers; ++layer) {
        for (int x = 0; x < grdb.xSize; ++x) {
            for (int y = 0; y < grdb.ySize; ++y) {
                const int encoded = grdb.encodeId(layer, x, y);
                auto& edge = grid.edges[layer][x][y];
                edge.capacity = grdb.capacity[encoded];
                edge.initialDemand = grdb.fixedUsage.empty() ? 0.0 : grdb.fixedUsage[encoded];
            }
        }
    }
    return grid;
}

std::vector<cugr2_bridge::Net> buildNets(const GRDatabase& grdb) {
    std::vector<cugr2_bridge::Net> prepared;
    prepared.reserve(grdb.grNets.size());
    for (int netId = 0; netId < static_cast<int>(grdb.grNets.size()); ++netId) {
        const GrNet& source = grdb.grNets[netId];
        if (source.noroute || source.getPins().size() < 2) continue;
        cugr2_bridge::Net net;
        net.index = netId;
        net.name = grdb.rawdb.nets[netId]->name;
        for (const auto& pin : source.getPins()) {
            std::vector<cugr2_bridge::Pin> accessPoints;
            accessPoints.reserve(pin.size());
            for (int encoded : pin) {
                const int layer = encoded / grdb.nMaxGrid / grdb.nMaxGrid;
                int x = encoded % (grdb.nMaxGrid * grdb.nMaxGrid) / grdb.nMaxGrid;
                int y = encoded % grdb.nMaxGrid;
                if (!((layer & 1) ^ grdb.m1direction)) std::swap(x, y);
                if (layer >= grdb.routingLayerBegin && layer <= grdb.routingLayerEnd &&
                    x >= 0 && x < grdb.xSize && y >= 0 && y < grdb.ySize) {
                    accessPoints.push_back({layer, x, y});
                }
            }
            if (accessPoints.empty()) {
                throw std::runtime_error("backend=cugr2 operation=prepare empty pin access group for net " +
                                         std::to_string(netId));
            }
            net.pinAccessPoints.push_back(std::move(accessPoints));
        }
        prepared.push_back(std::move(net));
    }
    return prepared;
}

}  // namespace

void CUGR2Router::installFluteProvider() {
    cugr2_bridge::setFluteProvider(&xplaceFluteInit, &xplaceFluteRoute);
}

void CUGR2Router::initialize(const GRDatabase& grdb) {
    accumulator_.initialize(grdb, "cugr2");
    accumulator_.setMap(
        grdb.capacity,
        grdb.wireDist,
        grdb.fixedLength,
        grdb.fixedUsage,
        grdb.fixTmpUsage,
        grdb.movTmpUsage);
    flutePowvFile_ = resolveLutPath("POWV9.dat");
    flutePostFile_ = resolveLutPath("POST9.dat");
    installFluteProvider();
}

void CUGR2Router::route(GRDatabase& grdb) {
    const cugr2_bridge::Grid grid = buildGrid(grdb);
    const std::vector<cugr2_bridge::Net> nets = buildNets(grdb);
    const cugr2_bridge::Result result = cugr2_bridge::route(grid, nets, flutePowvFile_, flutePostFile_);

    std::vector<std::vector<int>> rawRoutes(grdb.grNets.size(), {0});
    for (const cugr2_bridge::Route& route : result.routes) {
        if (route.failure != cugr2_bridge::FailureKind::None) {
            throw std::runtime_error("backend=cugr2 operation=route failed for net " +
                                     std::to_string(route.netIndex) + ": " + route.failureMessage);
        }
        grdb.grNets[route.netIndex].setWires(route.wires);
        grdb.grNets[route.netIndex].setVias(route.vias);
        rawRoutes[route.netIndex] = route.rawRoute;
    }
    accumulator_.loadCanonicalRoutes(grdb.grNets, rawRoutes);
    numOvflNets_ = static_cast<int>(result.finalOverflowNets);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CUGR2Router::getDemandMap() {
    return accumulator_.getDemandMap();
}

torch::Tensor CUGR2Router::getCapacityMap() { return accumulator_.getCapacityMap(); }

torch::Tensor CUGR2Router::getRawWireDemandMap() { return accumulator_.getRawWireDemandMap(); }

torch::Tensor CUGR2Router::getFixUsageMap() { return accumulator_.getFixUsageMap(); }

torch::Tensor CUGR2Router::getMovUsageMap() { return accumulator_.getMovUsageMap(); }

}  // namespace gr
