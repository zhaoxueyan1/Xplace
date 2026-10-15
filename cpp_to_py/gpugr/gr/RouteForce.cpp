#include "RouteForce.h"
#include "common/db/Database.h"
#include "io_parser/gp/GPDatabase.h"
#include "gpugr/db/GRDatabase.h"

namespace gr {

namespace {

torch::Tensor buildGridGraphMapTensor(
    const std::vector<float>& values,
    int layer_count,
    int n_max_grid,
    int x_size,
    int y_size,
    int direction) {
    auto map = torch::zeros({layer_count, x_size, y_size}, torch::dtype(torch::kFloat32));
    auto accessor = map.accessor<float, 3>();
    for (int layer = 0; layer < layer_count; ++layer) {
        for (int x = 0; x < x_size; ++x) {
            for (int y = 0; y < y_size; ++y) {
                int mapped_x = x;
                int mapped_y = y;
                if (!(layer & 1) ^ direction) {
                    std::swap(mapped_x, mapped_y);
                }
                int idx = layer * n_max_grid * n_max_grid + mapped_x * n_max_grid + mapped_y;
                accessor[layer][x][y] = values[idx];
            }
        }
    }
    return map;
}

bool isCpuPatternBackend(GGRBackend backend) {
    return backend == GGRBackend::CPU_PR || backend == GGRBackend::CPU_PR_MT;
}

bool isCpuRouteBackend(GGRBackend backend) {
    return isCpuPatternBackend(backend) || backend == GGRBackend::CPU_PR_MAZE;
}

}  // namespace

RouteForce::RouteForce(std::shared_ptr<gr::GRDatabase> grdb_) : grdb(*grdb_) {}

[[noreturn]] void RouteForce::unsupported(const std::string& operation) {
    throw std::runtime_error(
        "backend=" + ggrBackendName(backend) + " operation=" + operation + " is unsupported");
}

GGRBackend RouteForce::resolveBackend() {
    if (grSetting.backend == GGRBackend::AUTO) {
#if ENABLE_CUDA
        backend = GGRBackend::CUDA;
#else
        backend = GGRBackend::CPU_PR;
#endif
    } else {
        backend = grSetting.backend;
    }
#if !ENABLE_CUDA
    if (backend == GGRBackend::CUDA) {
        throw std::runtime_error("backend=cuda operation=run_ggr requested, but CUDA support is unavailable");
    }
#endif
    if (isCpuPatternBackend(backend) && grSetting.rrrIters > 0) {
        throw std::runtime_error(
            "backend=" + ggrBackendName(backend) +
            " operation=run_ggr only supports rrr_iters=0; CPU RRR is unsupported");
    }
    grSetting.resolvedBackend = backend;
    return backend;
}

void RouteForce::run_ggr() {
    logger.enable_logger();
    utils::timer T_total;
    T_total.start();
    resolveBackend();
    if (isCpuRouteBackend(backend)) {
        cpuRouter.initialize(grdb, ggrBackendName(backend));
        cpuRouter.setMap(
            grdb.capacity,
            grdb.wireDist,
            grdb.fixedLength,
            grdb.fixedUsage,
            grdb.fixTmpUsage,
            grdb.movTmpUsage);
        utils::timer T;
        T.start();
        if (backend == GGRBackend::CPU_PR_MT) {
            cpuRouter.routeParallel(grdb.grNets, grSetting.requestedWorkers);
        } else if (backend == GGRBackend::CPU_PR_MAZE) {
            cpuRouter.routeMazeRrr(grdb.grNets, grSetting.rrrIters, grSetting.requestedWorkers);
        } else {
            cpuRouter.route(grdb.grNets);
        }
        logger.info("##### %s Pattern Routing Time: %.4f #####",
                    ggrBackendName(backend).c_str(),
                    T.elapsed());

        if (grSetting.routeGuideFile != "") {
            grdb.writeGuides(grSetting.routeGuideFile);
        }

        logger.info("Total %s GR Time: %.4f", ggrBackendName(backend).c_str(), T_total.elapsed());
        logger.reset_logger();
        return;
    }

#if ENABLE_CUDA
    // we only need the PR segment, our current data structure unsupport MR route force
    // if rrrIters > 0, this router can only be used for congestion map computation or solution evaluation
    int runMazeRouteTimes = grSetting.rrrIters;
    // Parameters
    int rrrIterLimit = 1 + runMazeRouteTimes;
    double _unitWireCostRaw = 0.5 * grdb.microns / grdb.m2pitch;
    double _unitViaCostRaw = 4;
    double _unitViaCost = _unitViaCostRaw / _unitWireCostRaw * grdb.microns;
    double _unitShortVioCostRaw = 500;
    double rrrInitVioCostDiscount = 0.1;

    router.initialize(grSetting.deviceId,
                      grdb.nLayers,
                      grdb.xSize,
                      grdb.ySize,
                      grdb.nMaxGrid,
                      grdb.cgxsize,
                      grdb.cgysize,
                      grdb.m1direction,
                      grdb.csrnScale,
                      grdb.routingLayerBegin,
                      grdb.routingLayerEnd);
    router.setMap(grdb.capacity, grdb.wireDist, grdb.fixedLength, grdb.fixedUsage);

    std::vector<float> _unitShortVioCost(grdb.nLayers), _unitShortVioCostDiscounted(grdb.nLayers);
    router.setFromNets(grdb.grNets, grdb.gpdb.getPins().size());
    router.setUnitViaCost(_unitViaCost);
    for (int i = 0; i < grdb.nLayers; ++i) {
        _unitShortVioCost[i] =
            _unitShortVioCostRaw * grdb.layerWidth[i] * grdb.microns / grdb.m2pitch / grdb.m2pitch / _unitWireCostRaw;
    }
    utils::timer routingTimer;
    routingTimer.start();
    for (;;) {
        try {
            for (int iter = 0; iter < rrrIterLimit; iter++) {
                router.setLogisticSlope(1 << iter);
                router.setUnitVioCost(_unitShortVioCost, 0.1);
                if (iter == 0) {
                    router.setUnitViaMultiplier(1);
                } else {
                    router.setUnitViaMultiplier(std::max(100 / pow(5, iter - 1), 4.0));
                    router.setUnitVioCost(_unitShortVioCost,
                                          rrrInitVioCostDiscount + (1.0 - rrrInitVioCostDiscount) / (rrrIterLimit - 1) * iter);
                }
                utils::timer T;
                T.start();
                router.route(grdb.grNets, iter);
                logger.info("##### GPU Routing Iter: %d Time: %.4f #####", iter, T.elapsed());
                // break;
            }
            break;
        } catch (const RouteCapacityExceeded& overflow) {
            router.growRouteBuffer(overflow.netIds);
        }
    }
    const double tot_time = routingTimer.elapsed();
    router.setToNets(grdb.grNets);
    logger.info("Total GPU Routing time: %.4f", tot_time);

    if (grSetting.routeGuideFile != "") {
        grdb.writeGuides(grSetting.routeGuideFile);
    }

    logger.info("Total GPU GR Time: %.4f", T_total.elapsed());
    logger.reset_logger();
#else
    unsupported("run_ggr");
#endif
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RouteForce::getDemandMap() {
    if (isCpuRouteBackend(backend)) return cpuRouter.getDemandMap();
#if ENABLE_CUDA
    return router.getDemandMap();
#else
    unsupported("dmd_map");
#endif
}

torch::Tensor RouteForce::getCapacityMap() {
    if (isCpuRouteBackend(backend)) return cpuRouter.getCapacityMap();
#if ENABLE_CUDA
    return router.getCapacityMap();
#else
    unsupported("cap_map");
#endif
}

torch::Tensor RouteForce::getRawWireDemandMap() {
    if (isCpuRouteBackend(backend)) return cpuRouter.getRawWireDemandMap();
#if ENABLE_CUDA
    return router.getRawWireDemandMap();
#else
    unsupported("raw_wire_dmd_map");
#endif
}

torch::Tensor RouteForce::getFixUsageMap() {
    if (isCpuRouteBackend(backend)) return cpuRouter.getFixUsageMap();
    return buildGridGraphMapTensor(
        grdb.fixTmpUsage,
        grdb.nLayers,
        grdb.nMaxGrid,
        grdb.xSize,
        grdb.ySize,
        grdb.m1direction);
}

torch::Tensor RouteForce::getMovUsageMap() {
    if (isCpuRouteBackend(backend)) return cpuRouter.getMovUsageMap();
    return buildGridGraphMapTensor(
        grdb.movTmpUsage,
        grdb.nLayers,
        grdb.nMaxGrid,
        grdb.xSize,
        grdb.ySize,
        grdb.m1direction);
}

torch::Tensor RouteForce::calcRouteGrad(torch::Tensor mask_map,
                                        torch::Tensor wire_dmd_map_2d,
                                        torch::Tensor via_dmd_map_2d,
                                        torch::Tensor cap_map_2d,
                                        torch::Tensor dist_weights,
                                        torch::Tensor wirelength_weights,
                                        torch::Tensor route_gradmat,
                                        torch::Tensor node2pin_list,
                                        torch::Tensor node2pin_list_end,
                                        float grad_weight,
                                        float unit_wire_cost,
                                        float unit_via_cost,
                                        int num_nodes) {
    if (isCpuRouteBackend(backend)) unsupported("route_grad");
#if ENABLE_CUDA
    return router.calcRouteGrad(mask_map,
                                wire_dmd_map_2d,
                                via_dmd_map_2d,
                                cap_map_2d,
                                dist_weights,
                                wirelength_weights,
                                route_gradmat,
                                node2pin_list,
                                node2pin_list_end,
                                grad_weight,
                                unit_wire_cost,
                                unit_via_cost,
                                num_nodes);
#else
    unsupported("route_grad");
#endif
};

torch::Tensor RouteForce::calcFillerRouteGrad(torch::Tensor filler_pos,
                                              torch::Tensor filler_size,
                                              torch::Tensor filler_weight,
                                              torch::Tensor expand_ratio,
                                              torch::Tensor grad_mat,
                                              float grad_weight,
                                              float unit_len_x,
                                              float unit_len_y,
                                              int num_bin_x,
                                              int num_bin_y,
                                              int num_fillers) {
    if (isCpuRouteBackend(backend)) unsupported("filler_route_grad");
#if ENABLE_CUDA
    return router.calcFillerRouteGrad(filler_pos,
                                      filler_size,
                                      filler_weight,
                                      expand_ratio,
                                      grad_mat,
                                      grad_weight,
                                      unit_len_x,
                                      unit_len_y,
                                      num_bin_x,
                                      num_bin_y,
                                      num_fillers);
#else
    unsupported("filler_route_grad");
#endif
}

torch::Tensor RouteForce::calcPseudoPinGrad(torch::Tensor node_pos, torch::Tensor pseudo_pin_pos, float gamma) {
    if (isCpuRouteBackend(backend)) unsupported("pseudo_grad");
#if ENABLE_CUDA
    return router.calcPseudoPinGrad(node_pos, pseudo_pin_pos, gamma);
#else
    unsupported("pseudo_grad");
#endif
}

torch::Tensor RouteForce::calcNodeInflateRatio(torch::Tensor node_pos,
                                               torch::Tensor node_size,
                                               torch::Tensor node_weight,
                                               torch::Tensor expand_ratio,
                                               torch::Tensor inflate_mat,
                                               float grad_weight,
                                               float unit_len_x,
                                               float unit_len_y,
                                               int num_bin_x,
                                               int num_bin_y,
                                               bool use_weighted_inflation) {
    if (isCpuRouteBackend(backend)) unsupported("inflate_ratio");
#if ENABLE_CUDA
    return router.calcNodeInflateRatio(node_pos,
                                       node_size,
                                       node_weight,
                                       expand_ratio,
                                       inflate_mat,
                                       grad_weight,
                                       unit_len_x,
                                       unit_len_y,
                                       num_bin_x,
                                       num_bin_y,
                                       use_weighted_inflation);
#else
    unsupported("inflate_ratio");
#endif
}

torch::Tensor RouteForce::calcInflatedPinRelCpos(torch::Tensor node_inflate_ratio,
                                                 torch::Tensor old_pin_rel_cpos,
                                                 torch::Tensor pin_id2node_id,
                                                 int num_movable_conn_nodes) {
    if (isCpuRouteBackend(backend)) unsupported("inflate_pin_rel_cpos");
#if ENABLE_CUDA
    return router.calcInflatedPinRelCpos(node_inflate_ratio, old_pin_rel_cpos, pin_id2node_id, num_movable_conn_nodes);
#else
    unsupported("inflate_pin_rel_cpos");
#endif
}

int RouteForce::getNumOvflNets() {
    if (isCpuRouteBackend(backend)) return cpuRouter.getNumOvflNets();
#if ENABLE_CUDA
    return router.getNumOvflNets();
#else
    return 0;
#endif
}

int RouteForce::getMicrons() { return grdb.microns; }

std::tuple<int, int> RouteForce::getGcellStep() { return {grdb.mainGcellStepX, grdb.mainGcellStepY}; }

std::vector<int> RouteForce::getLayerPitch() { return grdb.layerPitch; }

std::vector<int> RouteForce::getLayerWidth() { return grdb.layerWidth; }

std::tuple<int, int> RouteForce::getRoutingLayerRange() const {
    return {grdb.routingLayerBegin, grdb.routingLayerEnd};
}

std::vector<std::string> RouteForce::getRoutingLayerNames() const {
    return grdb.routingLayerNames();
}

std::vector<std::vector<int>> RouteForce::getRawRoutes() {
    if (isCpuRouteBackend(backend)) return cpuRouter.getRawRoutes();
#if ENABLE_CUDA
    return router.getRawRoutes();
#else
    return {};
#endif
}

CpuRouteRunStats RouteForce::getRunStats() const {
    if (isCpuRouteBackend(backend)) {
        return cpuRouter.getRunStats();
    }
    CpuRouteRunStats stats;
    stats.terminalStatus = "unsupported_backend";
    return stats;
}

gr::GRDatabase& RouteForce::getGRDatabase() { return grdb; }

}  // namespace gr
