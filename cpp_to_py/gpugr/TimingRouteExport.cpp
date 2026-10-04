#include "TimingRouteExport.h"
#include "common/db/Database.h"
#include "common/db/Geometry.h"
#include "common/db/Net.h"
#include "common/db/Pin.h"
#include "gpugr/db/GRDatabase.h"
#include "gpugr/gr/RouteForce.h"
#include "io_parser/gp/GPDatabase.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <vector>

namespace Xplace {
namespace {
namespace py = pybind11;
using Point = std::tuple<int, int, int>;  // layer, grid x, grid y

template <class T>
py::array_t<T> array(const std::vector<T>& values) {
    py::array_t<T> result(values.size());
    std::copy(values.begin(), values.end(), result.mutable_data());
    return result;
}

int checkedIndex(std::size_t count) {
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("timing route export exceeds int32 index domain");
    }
    return static_cast<int>(count);
}

Point decode(const gr::GRDatabase& db, int encoded) {
    if (encoded < 0) throw std::runtime_error("negative timing route point");
    const int layer = encoded / db.nMaxGrid / db.nMaxGrid;
    int x = encoded % (db.nMaxGrid * db.nMaxGrid) / db.nMaxGrid;
    int y = encoded % db.nMaxGrid;
    if (!((layer & 1) ^ db.m1direction)) std::swap(x, y);
    if (layer >= db.nLayers || x >= db.xSize || y >= db.ySize) {
        throw std::runtime_error("timing route point outside routing grid");
    }
    return {layer, x, y};
}
}  // namespace

py::dict exportTimingRoutes(gr::RouteForce& routeforce) {
    auto& db = routeforce.getGRDatabase();
    const auto routes = routeforce.getRawRoutes();
    const auto& pins = db.gpdb.getPins();
    if (routes.size() != db.grNets.size() || routes.size() != db.rawdb.nets.size()) {
        throw std::runtime_error("timing route/net domain mismatch");
    }
    if (db.microns <= 0) throw std::runtime_error("invalid timing route DBU units");

    std::vector<int> vertexNet, vertexLayer, vertexX, vertexY;
    std::vector<int> edgeFrom, edgeTo, edgeKind, netVertexStart{0}, netEdgeStart{0};
    std::vector<int> pinVertex(pins.size(), -1), pinLayer(pins.size(), -1);
    std::vector<int> pinPhysicalLayer(pins.size(), -1), pinNet(pins.size(), -1);
    std::vector<int> pinX(pins.size(), 0), pinY(pins.size(), 0);
    std::vector<int> pinShape(pins.size() * 4, 0), pinDriver(pins.size(), 0);
    std::vector<int> pinLocalPathMissing(pins.size(), 1), netStatus;
    std::vector<int> netExclusion, fanout;
    std::vector<std::string> netNames, pinNames;
    for (const auto& pin : pins) pinNames.push_back(pin.getName());
    int duplicateEdges = 0;
    int missingLocalPaths = 0;
    int clampedLayers = 0;
    int clockCount = 0, highFanoutCount = 0;

    for (std::size_t netId = 0; netId < routes.size(); ++netId) {
        const auto& net = db.grNets[netId];
        const auto* rawNet = db.rawdb.nets[netId];
        netNames.push_back(rawNet->name);
        const auto& raw = routes[netId];
        // 0=routed, 1=coincident/single group, 2=failed, 3=unrouted, 4=excluded.
        netStatus.push_back(net.exclusion != gr::NetExclusion::None ? 4 :
                            raw.empty() || raw[0] < 0 ? 2 : net.noroute ? 1 : raw[0] == 0 ? 3 : 0);
        netExclusion.push_back(static_cast<int>(net.exclusion));
        fanout.push_back(gr::netFanout(*rawNet));
        clockCount += net.exclusion == gr::NetExclusion::Clock;
        highFanoutCount += net.exclusion == gr::NetExclusion::HighFanout;
        std::map<Point, int> vertices;
        std::set<std::pair<int, int>> edges;
        auto vertex = [&](Point point) {
            auto it = vertices.find(point);
            if (it != vertices.end()) return it->second;
            const auto [layer, x, y] = point;
            if (layer < 0 || layer >= db.nLayers || x < 0 || x >= db.xSize || y < 0 || y >= db.ySize) {
                throw std::runtime_error("timing route vertex outside routing grid");
            }
            const int id = checkedIndex(vertexNet.size());
            vertices.emplace(point, id);
            vertexNet.push_back(checkedIndex(netId));
            vertexLayer.push_back(layer);
            vertexX.push_back(db.gridCenters[0][x]);
            vertexY.push_back(db.gridCenters[1][y]);
            return id;
        };
        auto edge = [&](Point a, Point b, int kind) {
            int u = vertex(a), v = vertex(b);
            auto key = std::minmax(u, v);
            if (!edges.emplace(key.first, key.second).second) {
                ++duplicateEdges;
                return;
            }
            checkedIndex(edgeFrom.size() + 1);
            edgeFrom.push_back(u);
            edgeTo.push_back(v);
            edgeKind.push_back(kind);  // 0=wire, 1=adjacent-layer via
        };
        if (!raw.empty() && raw[0] > 0) {
            if (raw[0] != checkedIndex(raw.size()) || raw[0] % 2 != 1) {
                throw std::runtime_error("malformed timing raw route");
            }
            for (int i = 1; i + 1 < raw[0]; i += 2) {
                auto [layer, x, y] = decode(db, raw[i]);
                const int span = raw[i + 1];
                if (span == -1) {
                    edge({layer, x, y}, {layer + 1, x, y}, 1);
                } else if (span > 0) {
                    const bool vertical = ((layer & 1) ^ db.m1direction);
                    for (int k = 0; k < span; ++k) {
                        edge({layer, x + (vertical ? 0 : k), y + (vertical ? k : 0)},
                             {layer, x + (vertical ? 0 : k + 1), y + (vertical ? k + 1 : 0)}, 0);
                    }
                } else {
                    throw std::runtime_error("unsupported timing route span for net " +
                                             rawNet->name + " at record " + std::to_string(i));
                }
            }
        }
        // Use each pin's recorded setup choice, including same-gcell pins.
        // Missing local access stays explicit; no wire or via is invented.
        for (const auto& access : net.pinAccess) {
            const int p = access.gpdbPinId;
            if (p < 0 || p >= checkedIndex(pins.size()) || pinVertex[p] != -1) {
                throw std::runtime_error("invalid or duplicate timing pin access");
            }
            pinVertex[p] = vertex({access.layer, access.x, access.y});
            pinNet[p] = checkedIndex(netId);
            pinLayer[p] = access.layer;
            pinPhysicalLayer[p] = access.physicalLayer;
            // The timing estimator uses the selected physical shape's center;
            // the router's actual gcell/layer choice remains unchanged.
            pinX[p] = access.lx + (static_cast<int64_t>(access.hx) - access.lx) / 2;
            pinY[p] = access.ly + (static_cast<int64_t>(access.hy) - access.ly) / 2;
            const int cx = db.gridCenters[0][access.x], cy = db.gridCenters[1][access.y];
            const bool onShape = access.layer == access.physicalLayer &&
                                 access.lx <= cx && cx <= access.hx &&
                                 access.ly <= cy && cy <= access.hy;
            pinLocalPathMissing[p] = !onShape;
            missingLocalPaths += !onShape;
            clampedLayers += access.layer != access.physicalLayer;
            const int bounds[] = {access.lx, access.ly, access.hx, access.hy};
            std::copy(bounds, bounds + 4, pinShape.begin() + 4 * p);
        }
        for (const auto* pin : rawNet->pins) {
            if (pin->gpdb_id < 0) continue;
            const int p = pin->gpdb_id;
            if (p >= checkedIndex(pins.size())) throw std::runtime_error("raw timing pin ID out of range");
            pinNet[p] = checkedIndex(netId);
            // The DEF parser already reverses port direction into the net
            // domain (a chip INPUT is an electrical output).
            pinDriver[p] = pin->type->direction() == 'o';
        }
        netVertexStart.push_back(checkedIndex(vertexNet.size()));
        netEdgeStart.push_back(checkedIndex(edgeFrom.size()));
    }
    py::dict result;
    result["vertex_net"] = array(vertexNet);
    result["vertex_layer"] = array(vertexLayer);
    result["vertex_x_dbu"] = array(vertexX);
    result["vertex_y_dbu"] = array(vertexY);
    result["edge_from"] = array(edgeFrom);
    result["edge_to"] = array(edgeTo);
    result["edge_kind"] = array(edgeKind);
    result["net_vertex_start"] = array(netVertexStart);
    result["net_edge_start"] = array(netEdgeStart);
    result["net_status"] = array(netStatus);
    result["net_exclusion_reason"] = array(netExclusion);
    result["net_fanout"] = array(fanout);
    result["excluded_clock_net_count"] = clockCount;
    result["excluded_high_fanout_net_count"] = highFanoutCount;
    result["max_gr_fanout"] = gr::MAX_GR_FANOUT;
    result["pin_vertex"] = array(pinVertex);
    result["pin_net"] = array(pinNet);
    result["pin_is_driver"] = array(pinDriver);
    result["pin_access_layer"] = array(pinLayer);
    result["pin_physical_layer"] = array(pinPhysicalLayer);
    result["pin_x_dbu"] = array(pinX);
    result["pin_y_dbu"] = array(pinY);
    result["pin_shape_dbu"] = array(pinShape);
    result["pin_local_path_missing"] = array(pinLocalPathMissing);
    result["pin_names"] = pinNames;
    result["net_names"] = netNames;
    result["layer_names"] = db.routingLayerNames();
    result["dbu_per_micron"] = db.microns;
    result["duplicate_edge_count"] = duplicateEdges;
    result["missing_local_path_count"] = missingLocalPaths;
    result["clamped_pin_layer_count"] = clampedLayers;
    result["pin_access_model"] = "router_selected_gcell_with_physical_shape_evidence";
    return result;
}
}  // namespace Xplace
