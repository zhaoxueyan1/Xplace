#include "common/common.h"
#include "common/db/Database.h"
#include "common/db/Layer.h"
#include "common/db/Net.h"
#include "io_parser/gp/GPDatabase.h"
#include "gpugr/db/GRDatabase.h"
#include "gpugr/gr/RouteForce.h"
#include "flute.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace Xplace {

namespace {

std::tuple<int, int, int> decodeRoutePoint(const gr::GRDatabase& grdb, int p) {
    int l = p / grdb.nMaxGrid / grdb.nMaxGrid;
    int x = p % (grdb.nMaxGrid * grdb.nMaxGrid) / grdb.nMaxGrid;
    int y = p % grdb.nMaxGrid;
    if (!(l & 1) ^ grdb.m1direction) std::swap(x, y);
    return {l, x, y};
}

pybind11::dict decodeWireEntry(const gr::GRDatabase& grdb,
                               const std::vector<std::string>& rlayerNames,
                               int order,
                               int p,
                               int span) {
    auto [l, x, y] = decodeRoutePoint(grdb, p);
    bool vertical = ((l & 1) ^ grdb.m1direction);
    int xmin = x;
    int xmax = vertical ? x : x + span;
    int ymin = y;
    int ymax = vertical ? y + span : y;

    pybind11::dict entry;
    entry["order"] = order;
    entry["type"] = "wire";
    entry["raw_point"] = p;
    entry["raw_span"] = span;
    entry["layer_idx"] = l;
    entry["layer_name"] = rlayerNames[l];
    entry["orientation"] = vertical ? "V" : "H";
    entry["grid_x1"] = xmin;
    entry["grid_y1"] = ymin;
    entry["grid_x2"] = xmax;
    entry["grid_y2"] = ymax;
    entry["dbu_lx"] = grdb.gridlines[0][xmin];
    entry["dbu_ly"] = grdb.gridlines[1][ymin];
    entry["dbu_hx"] = grdb.gridlines[0][xmax + 1];
    entry["dbu_hy"] = grdb.gridlines[1][ymax + 1];
    entry["dbu_center_x1"] = grdb.gridCenters[0][xmin];
    entry["dbu_center_y1"] = grdb.gridCenters[1][ymin];
    entry["dbu_center_x2"] = grdb.gridCenters[0][xmax];
    entry["dbu_center_y2"] = grdb.gridCenters[1][ymax];
    return entry;
}

pybind11::dict decodeViaEntry(const gr::GRDatabase& grdb,
                              const std::vector<std::string>& rlayerNames,
                              int order,
                              int p) {
    auto [l, x, y] = decodeRoutePoint(grdb, p);

    pybind11::dict entry;
    entry["order"] = order;
    entry["type"] = "via";
    entry["raw_point"] = p;
    entry["raw_span"] = -1;
    entry["layer_idx"] = l;
    entry["layer_name"] = rlayerNames[l];
    entry["orientation"] = "VIA";
    entry["grid_x1"] = x;
    entry["grid_y1"] = y;
    entry["grid_x2"] = x;
    entry["grid_y2"] = y;
    entry["dbu_lx"] = grdb.gridlines[0][x];
    entry["dbu_ly"] = grdb.gridlines[1][y];
    entry["dbu_hx"] = grdb.gridlines[0][x + 1];
    entry["dbu_hy"] = grdb.gridlines[1][y + 1];
    entry["dbu_center_x1"] = grdb.gridCenters[0][x];
    entry["dbu_center_y1"] = grdb.gridCenters[1][y];
    entry["dbu_center_x2"] = grdb.gridCenters[0][x];
    entry["dbu_center_y2"] = grdb.gridCenters[1][y];
    return entry;
}

pybind11::list buildRouteEntries(gr::RouteForce& routeforce) {
    pybind11::list allNets;
    auto rawRoutes = routeforce.getRawRoutes();
    auto& grdb = routeforce.getGRDatabase();

    std::vector<std::string> rlayerNames(grdb.rawdb.getNumRLayers());
    for (int l = 0; l < grdb.rawdb.getNumRLayers(); l++) {
        rlayerNames[l] = grdb.rawdb.getRLayer(l)->name();
    }

    const auto& gpNets = grdb.gpdb.getNets();
    for (size_t netId = 0; netId < rawRoutes.size(); netId++) {
        pybind11::dict netInfo;
        const auto& rawRoute = rawRoutes[netId];
        bool routeFailed = rawRoute.empty() || rawRoute[0] < 0;

        int rawdbNetId = -1;
        std::string netName = "";
        if (netId < gpNets.size()) {
            rawdbNetId = gpNets[netId].getOriDBId();
            if (0 <= rawdbNetId && rawdbNetId < static_cast<int>(grdb.rawdb.nets.size())) {
                netName = grdb.rawdb.nets[rawdbNetId]->name;
            } else {
                netName = gpNets[netId].getName();
            }
        }

        netInfo["net_id"] = static_cast<int>(netId);
        netInfo["rawdb_net_id"] = rawdbNetId;
        netInfo["net_name"] = netName;
        netInfo["route_failed"] = routeFailed;
        netInfo["raw_route"] = rawRoute;

        pybind11::list entries;
        if (!routeFailed && !rawRoute.empty()) {
            int routeEnd = std::min<int>(rawRoute.size(), rawRoute[0]);
            for (int i = 1, order = 0; i + 1 < routeEnd; i += 2, order++) {
                int p = rawRoute[i];
                int span = rawRoute[i + 1];
                if (span > 0) {
                    entries.append(decodeWireEntry(grdb, rlayerNames, order, p, span));
                } else if (span == -1) {
                    entries.append(decodeViaEntry(grdb, rlayerNames, order, p));
                }
            }
        }
        netInfo["entries"] = entries;
        allNets.append(netInfo);
    }
    return allNets;
}

}  // namespace

bool loadGRParams(const pybind11::dict& kwargs) {
    gr::grSetting.reset();
    // ----- design related options -----

    if (kwargs.contains("device_id")) {
        gr::grSetting.deviceId = kwargs["device_id"].cast<int>();
    }

    if (kwargs.contains("route_xSize")) {
        gr::grSetting.routeXSize = kwargs["route_xSize"].cast<int>();
    }

    if (kwargs.contains("route_ySize")) {
        gr::grSetting.routeYSize = kwargs["route_ySize"].cast<int>();
    }

    if (kwargs.contains("csrn_scale")) {
        gr::grSetting.csrnScale = kwargs["csrn_scale"].cast<int>();
    }

    if (kwargs.contains("rrrIters")) {
        gr::grSetting.rrrIters = kwargs["rrrIters"].cast<int>();
    }

    if (kwargs.contains("route_guide")) {
        gr::grSetting.routeGuideFile = kwargs["route_guide"].cast<std::string>();
    }

    return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<gr::GRDatabase, std::shared_ptr<gr::GRDatabase>>(m, "GRDatabase")
        .def(pybind11::init<std::shared_ptr<db::Database>, std::shared_ptr<gp::GPDatabase>>())
        .def("report_gr_stat", &gr::GRDatabase::reportGRStat)
        .def("setup_capacity", &gr::GRDatabase::setupCapacity)
        .def("setup_wiredist", &gr::GRDatabase::setupWireDist)
        .def("setup_obs", &gr::GRDatabase::setupObs)
        .def("setup_grnets", &gr::GRDatabase::setupGrNets);
    pybind11::class_<gr::RouteForce, std::shared_ptr<gr::RouteForce>>(m, "RouteForce")
        .def(pybind11::init<std::shared_ptr<gr::GRDatabase>>())
        .def("run_ggr", &gr::RouteForce::run_ggr)
        .def("num_ovfl_nets", &gr::RouteForce::getNumOvflNets)
        .def("gcell_steps", &gr::RouteForce::getGcellStep)
        .def("microns", &gr::RouteForce::getMicrons)
        .def("layer_pitch", &gr::RouteForce::getLayerPitch)
        .def("layer_width", &gr::RouteForce::getLayerWidth)
        .def("raw_routes", &gr::RouteForce::getRawRoutes, py::return_value_policy::move)
        .def("route_entries", [](gr::RouteForce& routeforce) { return buildRouteEntries(routeforce); })
        .def("dmd_map", &gr::RouteForce::getDemandMap, py::return_value_policy::move)
        .def("cap_map", &gr::RouteForce::getCapacityMap, py::return_value_policy::move)
        .def("raw_wire_dmd_map", &gr::RouteForce::getRawWireDemandMap, py::return_value_policy::move)
        .def("fix_usage_map", &gr::RouteForce::getFixUsageMap, py::return_value_policy::move)
        .def("mov_usage_map", &gr::RouteForce::getMovUsageMap, py::return_value_policy::move)
        .def("route_grad", &gr::RouteForce::calcRouteGrad, py::return_value_policy::move)
        .def("filler_route_grad", &gr::RouteForce::calcFillerRouteGrad, py::return_value_policy::move)
        .def("pseudo_grad", &gr::RouteForce::calcPseudoPinGrad, py::return_value_policy::move)
        .def("inflate_ratio", &gr::RouteForce::calcNodeInflateRatio, py::return_value_policy::move)
        .def("inflate_pin_rel_cpos", &gr::RouteForce::calcInflatedPinRelCpos, py::return_value_policy::move);
    
    m.def("create_grdatabase", [](std::shared_ptr<db::Database> rawdb, std::shared_ptr<gp::GPDatabase> gpdb) {
        logger.enable_logger();
        std::shared_ptr<gr::GRDatabase> grdb = std::make_shared<gr::GRDatabase>(rawdb, gpdb);
        logger.reset_logger();
        return grdb;
    });
    m.def("create_routeforce", [](std::shared_ptr<gr::GRDatabase> grdb) {
        logger.enable_logger();
        std::shared_ptr<gr::RouteForce> routeforce = std::make_shared<gr::RouteForce>(grdb);
        logger.reset_logger();
        return routeforce;
    });
    m.def("load_gr_params", &loadGRParams, "Parse input args to DB and return graph information");
    m.def("read_flute", &Flute::readLUT, "Read Flute LUT");
}

}  // namespace Xplace
