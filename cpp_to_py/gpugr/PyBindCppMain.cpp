#include "common/common.h"
#include "common/db/Database.h"
#include "common/db/Layer.h"
#include "common/db/Net.h"
#include "io_parser/gp/GPDatabase.h"
#include "gpugr/db/GRDatabase.h"
#include "gpugr/gr/RouteForce.h"
#include "flute.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace Xplace {

namespace {

struct IntInterval {
    int lo = 0;
    int hi = 0;
};

struct PackedTopologyRecord {
    int net_id = -1;
    std::map<int, std::vector<IntInterval>> horizontal_by_row;
    std::map<int, std::vector<IntInterval>> vertical_by_col;
    int num_horizontal_intervals = 0;
    int num_vertical_intervals = 0;
    int total_horizontal_length = 0;
    int total_vertical_length = 0;
    int raw_wire_count = 0;
    int invalid_wire_count = 0;
    bool route_failed = false;
};

std::tuple<int, int, int> decodeRoutePoint(const gr::GRDatabase& grdb, int p) {
    int l = p / grdb.nMaxGrid / grdb.nMaxGrid;
    int x = p % (grdb.nMaxGrid * grdb.nMaxGrid) / grdb.nMaxGrid;
    int y = p % grdb.nMaxGrid;
    if (!(l & 1) ^ grdb.m1direction) std::swap(x, y);
    return {l, x, y};
}

inline void mergeNormalizedIntIntervals(std::vector<IntInterval>& intervals, int maxGap) {
    if (intervals.size() <= 1) {
        return;
    }
    std::sort(intervals.begin(), intervals.end(), [](const IntInterval& a, const IntInterval& b) {
        if (a.lo != b.lo) {
            return a.lo < b.lo;
        }
        return a.hi < b.hi;
    });

    std::vector<IntInterval> merged;
    merged.reserve(intervals.size());
    IntInterval current = intervals.front();
    for (size_t i = 1; i < intervals.size(); ++i) {
        const IntInterval& next = intervals[i];
        if (next.lo <= current.hi + maxGap) {
            if (next.hi > current.hi) {
                current.hi = next.hi;
            }
        } else {
            merged.push_back(current);
            current = next;
        }
    }
    merged.push_back(current);
    intervals.swap(merged);
}

inline int intervalTotalLength(const std::vector<IntInterval>& intervals) {
    int total = 0;
    for (const IntInterval& interval : intervals) {
        total += interval.hi - interval.lo;
    }
    return total;
}

inline bool finalizePackedTopologyRecord(PackedTopologyRecord& record, int maxGap) {
    for (auto it = record.horizontal_by_row.begin(); it != record.horizontal_by_row.end();) {
        mergeNormalizedIntIntervals(it->second, maxGap);
        if (it->second.empty()) {
            it = record.horizontal_by_row.erase(it);
        } else {
            record.num_horizontal_intervals += static_cast<int>(it->second.size());
            record.total_horizontal_length += intervalTotalLength(it->second);
            ++it;
        }
    }
    for (auto it = record.vertical_by_col.begin(); it != record.vertical_by_col.end();) {
        mergeNormalizedIntIntervals(it->second, maxGap);
        if (it->second.empty()) {
            it = record.vertical_by_col.erase(it);
        } else {
            record.num_vertical_intervals += static_cast<int>(it->second.size());
            record.total_vertical_length += intervalTotalLength(it->second);
            ++it;
        }
    }
    return record.num_horizontal_intervals > 0 || record.num_vertical_intervals > 0;
}

template <typename T>
pybind11::array_t<T> vectorToNumpy(const std::vector<T>& values) {
    pybind11::array_t<T> result(values.size());
    auto resultView = result.template mutable_unchecked<1>();
    for (size_t i = 0; i < values.size(); ++i) {
        resultView(i) = values[i];
    }
    return result;
}

inline float gridCenter(int coordIdx, float origin, float step) {
    return origin + (static_cast<float>(coordIdx) + 0.5f) * step;
}

int mappedNetId(const pybind11::dict& netNameToId, const std::string& netName, int fallbackNetId) {
    if (!netName.empty() && netNameToId.contains(pybind11::str(netName))) {
        return netNameToId[pybind11::str(netName)].cast<int>();
    }
    return fallbackNetId;
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

pybind11::dict packTopologyRecords(const std::map<int, PackedTopologyRecord>& records,
                                   double xl,
                                   double yl,
                                   double routeBinSizeX,
                                   double routeBinSizeY) {
    std::vector<int32_t> netIds;
    std::vector<int32_t> netIndexById;
    std::vector<int32_t> hSegOffsets;
    std::vector<int32_t> vSegOffsets;
    std::vector<float> hX1;
    std::vector<float> hY;
    std::vector<float> hX2;
    std::vector<float> vX;
    std::vector<float> vY1;
    std::vector<float> vY2;

    netIds.reserve(records.size());
    hSegOffsets.reserve(records.size() + 1);
    vSegOffsets.reserve(records.size() + 1);
    hSegOffsets.push_back(0);
    vSegOffsets.push_back(0);

    float xlF = static_cast<float>(xl);
    float ylF = static_cast<float>(yl);
    float routeBinSizeXF = static_cast<float>(routeBinSizeX);
    float routeBinSizeYF = static_cast<float>(routeBinSizeY);
    int maxNetId = -1;

    for (const auto& [netId, record] : records) {
        netIds.push_back(static_cast<int32_t>(netId));
        maxNetId = std::max(maxNetId, netId);

        int32_t hCount = 0;
        for (const auto& [rowIdx, intervals] : record.horizontal_by_row) {
            float segY = gridCenter(rowIdx, ylF, routeBinSizeYF);
            for (const IntInterval& interval : intervals) {
                hX1.push_back(gridCenter(interval.lo, xlF, routeBinSizeXF));
                hY.push_back(segY);
                hX2.push_back(gridCenter(interval.hi, xlF, routeBinSizeXF));
                ++hCount;
            }
        }
        hSegOffsets.push_back(hSegOffsets.back() + hCount);

        int32_t vCount = 0;
        for (const auto& [colIdx, intervals] : record.vertical_by_col) {
            float segX = gridCenter(colIdx, xlF, routeBinSizeXF);
            for (const IntInterval& interval : intervals) {
                vX.push_back(segX);
                vY1.push_back(gridCenter(interval.lo, ylF, routeBinSizeYF));
                vY2.push_back(gridCenter(interval.hi, ylF, routeBinSizeYF));
                ++vCount;
            }
        }
        vSegOffsets.push_back(vSegOffsets.back() + vCount);
    }

    if (maxNetId >= 0) {
        netIndexById.assign(static_cast<size_t>(maxNetId) + 1, static_cast<int32_t>(-1));
        for (size_t index = 0; index < netIds.size(); ++index) {
            int32_t netId = netIds[index];
            if (netId >= 0) {
                netIndexById[static_cast<size_t>(netId)] = static_cast<int32_t>(index);
            }
        }
    }

    pybind11::dict packed;
    packed["net_ids"] = vectorToNumpy(netIds);
    packed["net_index_by_id"] = vectorToNumpy(netIndexById);
    packed["h_seg_offsets"] = vectorToNumpy(hSegOffsets);
    packed["v_seg_offsets"] = vectorToNumpy(vSegOffsets);
    packed["h_x1"] = vectorToNumpy(hX1);
    packed["h_y"] = vectorToNumpy(hY);
    packed["h_x2"] = vectorToNumpy(hX2);
    packed["v_x"] = vectorToNumpy(vX);
    packed["v_y1"] = vectorToNumpy(vY1);
    packed["v_y2"] = vectorToNumpy(vY2);
    return packed;
}

pybind11::dict buildSameNetTopologyPack(gr::RouteForce& routeforce,
                                        pybind11::dict netNameToId,
                                        int maxGap,
                                        int routeNumBinsX,
                                        int routeNumBinsY,
                                        double xl,
                                        double yl,
                                        double routeBinSizeX,
                                        double routeBinSizeY) {
    auto rawRoutes = routeforce.getRawRoutes();
    auto& grdb = routeforce.getGRDatabase();
    const auto& gpNets = grdb.gpdb.getNets();

    std::map<int, PackedTopologyRecord> records;
    int numRouteEntryNets = 0;
    int numRouteFailedNets = 0;
    int totalSegmentsH = 0;
    int totalSegmentsV = 0;
    int totalLengthH = 0;
    int totalLengthV = 0;
    int totalEntryCount = 0;
    int totalWireEntryCount = 0;
    int invalidWireCount = 0;
    int maxIntervalsPerNet = 0;

    for (size_t netId = 0; netId < rawRoutes.size(); ++netId) {
        ++numRouteEntryNets;
        const auto& rawRoute = rawRoutes[netId];
        bool routeFailed = rawRoute.empty() || rawRoute[0] < 0;
        if (routeFailed) {
            ++numRouteFailedNets;
        }

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

        int targetNetId = mappedNetId(netNameToId, netName, static_cast<int>(netId));
        PackedTopologyRecord record;
        record.net_id = targetNetId;
        record.route_failed = routeFailed;

        if (!routeFailed && !rawRoute.empty()) {
            int routeEnd = std::min(static_cast<int>(rawRoute.size()), rawRoute[0]);
            for (int i = 1; i + 1 < routeEnd; i += 2) {
                int p = rawRoute[i];
                int span = rawRoute[i + 1];
                if (span > 0) {
                    ++totalEntryCount;
                    ++record.raw_wire_count;
                    auto [l, x, y] = decodeRoutePoint(grdb, p);
                    bool vertical = ((l & 1) ^ grdb.m1direction);
                    int x1 = x;
                    int y1 = y;
                    int x2 = vertical ? x : x + span;
                    int y2 = vertical ? y + span : y;

                    if (!vertical && y1 == y2) {
                        int lo = std::min(x1, x2);
                        int hi = std::max(x1, x2);
                        record.horizontal_by_row[y1].push_back({lo, hi});
                    } else if (vertical && x1 == x2) {
                        int lo = std::min(y1, y2);
                        int hi = std::max(y1, y2);
                        record.vertical_by_col[x1].push_back({lo, hi});
                    } else {
                        ++record.invalid_wire_count;
                    }
                } else if (span == -1) {
                    ++totalEntryCount;
                }
            }
        }

        totalWireEntryCount += record.raw_wire_count;
        if (!finalizePackedTopologyRecord(record, maxGap)) {
            continue;
        }

        records[targetNetId] = record;
        totalSegmentsH += record.num_horizontal_intervals;
        totalSegmentsV += record.num_vertical_intervals;
        totalLengthH += record.total_horizontal_length;
        totalLengthV += record.total_vertical_length;
        invalidWireCount += record.invalid_wire_count;
        maxIntervalsPerNet = std::max(
            maxIntervalsPerNet,
            record.num_horizontal_intervals + record.num_vertical_intervals);
    }

    pybind11::dict stats;
    stats["route_grid_shape"] = pybind11::make_tuple(routeNumBinsX, routeNumBinsY);
    stats["num_route_entry_nets"] = pybind11::int_(numRouteEntryNets);
    stats["num_nets_with_topology"] = pybind11::int_(static_cast<int>(records.size()));
    stats["num_route_failed_nets"] = pybind11::int_(numRouteFailedNets);
    stats["unknown_name_count"] = pybind11::int_(0);
    stats["num_segments_h"] = pybind11::int_(totalSegmentsH);
    stats["num_segments_v"] = pybind11::int_(totalSegmentsV);
    stats["total_horizontal_length"] = pybind11::int_(totalLengthH);
    stats["total_vertical_length"] = pybind11::int_(totalLengthV);
    stats["total_entry_count"] = pybind11::int_(totalEntryCount);
    stats["wire_entry_count"] = pybind11::int_(totalWireEntryCount);
    stats["invalid_wire_count"] = pybind11::int_(invalidWireCount);
    stats["max_intervals_per_net"] = pybind11::int_(maxIntervalsPerNet);
    stats["backend"] = pybind11::str("gpugr_packed");

    pybind11::dict cache;
    cache["route_grid_shape"] = pybind11::make_tuple(routeNumBinsX, routeNumBinsY);
    cache["net_topologies"] = pybind11::dict();
    cache["route_entry_meta"] = pybind11::dict();
    cache["num_nets_with_topology"] = pybind11::int_(static_cast<int>(records.size()));
    cache["num_segments_h"] = pybind11::int_(totalSegmentsH);
    cache["num_segments_v"] = pybind11::int_(totalSegmentsV);
    cache["compact_topology_cache"] = pybind11::bool_(true);
    cache["_same_net_topo_cpp_pack_key"] = pybind11::make_tuple(xl, yl, routeBinSizeX, routeBinSizeY);
    cache["_same_net_topo_cpp_packed"] = packTopologyRecords(records, xl, yl, routeBinSizeX, routeBinSizeY);

    pybind11::dict result;
    result["cache"] = cache;
    result["stats"] = stats;
    return result;
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
        .def(
            "same_net_topology_pack",
            [](gr::RouteForce& routeforce,
               pybind11::dict netNameToId,
               int maxGap,
               int routeNumBinsX,
               int routeNumBinsY,
               double xl,
               double yl,
               double routeBinSizeX,
               double routeBinSizeY) {
                return buildSameNetTopologyPack(
                    routeforce,
                    netNameToId,
                    maxGap,
                    routeNumBinsX,
                    routeNumBinsY,
                    xl,
                    yl,
                    routeBinSizeX,
                    routeBinSizeY);
            },
            pybind11::arg("net_name_to_id"),
            pybind11::arg("max_gap") = 1,
            pybind11::arg("route_num_bins_x") = 0,
            pybind11::arg("route_num_bins_y") = 0,
            pybind11::arg("xl") = 0.0,
            pybind11::arg("yl") = 0.0,
            pybind11::arg("route_bin_size_x") = 1.0,
            pybind11::arg("route_bin_size_y") = 1.0)
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
