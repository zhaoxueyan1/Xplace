#include "common/common.h"
#include "common/db/Database.h"
#include "common/db/Layer.h"
#include "common/db/Net.h"
#include "io_parser/gp/GPDatabase.h"
#include "gpugr/db/GRDatabase.h"
#include "gpugr/gr/RouteForce.h"
#include "flute.h"
#include "TimingRouteExport.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
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

pybind11::list buildGRDatabaseSetupProfile(const gr::GRDatabase& grdb) {
    pybind11::list records;
    for (const auto& record : grdb.getSetupProfileRecords()) {
        pybind11::dict item;
        item["name"] = record.name;
        item["elapsed_ms"] = record.elapsedMs;
        for (const auto& field : record.fields) {
            item[pybind11::str(field.first)] = field.second;
        }
        records.append(item);
    }
    return records;
}

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

std::vector<int> intArrayToVector(pybind11::handle value, const std::string& fieldName) {
    auto array = pybind11::array_t<int64_t, pybind11::array::c_style | pybind11::array::forcecast>::ensure(value);
    if (!array) {
        throw std::runtime_error("GGR L-shape topology pack field '" + fieldName + "' must be an integer array");
    }
    if (array.ndim() != 1) {
        throw std::runtime_error("GGR L-shape topology pack field '" + fieldName + "' must be 1D");
    }
    std::vector<int> values(static_cast<size_t>(array.shape(0)));
    auto view = array.unchecked<1>();
    for (pybind11::ssize_t i = 0; i < array.shape(0); ++i) {
        values[static_cast<size_t>(i)] = static_cast<int>(view(i));
    }
    return values;
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

int mappedLShapeNetId(const pybind11::dict& netNameToId, const std::string& netName) {
    if (!netName.empty() && netNameToId.contains(pybind11::str(netName))) {
        return netNameToId[pybind11::str(netName)].cast<int>();
    }
    return -1;
}

int mappedPinId(const pybind11::dict& pinNameToId, const gp::GPPin& pin) {
    const std::string pinName = pin.getName();
    if (!pinName.empty() && pinNameToId.contains(pybind11::str(pinName))) {
        return pinNameToId[pybind11::str(pinName)].cast<int>();
    }
    const std::string macroName = pin.getMacroName();
    if (!macroName.empty() && pinNameToId.contains(pybind11::str(macroName))) {
        return pinNameToId[pybind11::str(macroName)].cast<int>();
    }
    throw std::runtime_error("GGR L-shape topology pack cannot map gpdb pin '" + pinName + "'");
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

pybind11::dict buildGRNetSummary(gr::GRDatabase& grdb) {
    pybind11::dict summary;
    uint64_t routeableCount = 0;
    uint64_t norouteCount = 0;
    uint64_t pinGroupCount = 0;
    uint64_t pinGridCount = 0;
    uint64_t pinGpdbRefCount = 0;
    uint64_t bboxChecksum = 1469598103934665603ULL;
    uint64_t pinChecksum = 1099511628211ULL;
    uint64_t norouteChecksum = 7809847782465536322ULL;

    for (size_t netId = 0; netId < grdb.grNets.size(); ++netId) {
        auto& net = grdb.grNets[netId];
        if (net.noroute) {
            norouteCount++;
            norouteChecksum = norouteChecksum * 1099511628211ULL + static_cast<uint64_t>(netId + 1) * 97ULL;
        } else {
            routeableCount++;
        }

        bboxChecksum = bboxChecksum * 1099511628211ULL + static_cast<uint64_t>(netId + 1) * 1315423911ULL;
        bboxChecksum = bboxChecksum * 1099511628211ULL + static_cast<uint64_t>(net.lowerx) * 31ULL;
        bboxChecksum = bboxChecksum * 1099511628211ULL + static_cast<uint64_t>(net.lowery) * 37ULL;
        bboxChecksum = bboxChecksum * 1099511628211ULL + static_cast<uint64_t>(net.upperx) * 41ULL;
        bboxChecksum = bboxChecksum * 1099511628211ULL + static_cast<uint64_t>(net.uppery) * 43ULL;
        bboxChecksum = bboxChecksum * 1099511628211ULL + static_cast<uint64_t>(net.noroute) * 47ULL;

        const auto& pins = net.getPins();
        pinGroupCount += static_cast<uint64_t>(pins.size());
        for (size_t pinGroupId = 0; pinGroupId < pins.size(); ++pinGroupId) {
            const auto& pinGroup = pins[pinGroupId];
            pinGridCount += static_cast<uint64_t>(pinGroup.size());
            pinChecksum = pinChecksum * 1099511628211ULL + static_cast<uint64_t>(netId + 1) * 53ULL;
            pinChecksum = pinChecksum * 1099511628211ULL + static_cast<uint64_t>(pinGroupId + 1) * 59ULL;
            for (int gridId : pinGroup) {
                pinChecksum = pinChecksum * 1099511628211ULL + static_cast<uint64_t>(gridId + 1) * 61ULL;
            }
        }
        for (const auto& gpdbPins : net.pin2gpdbPinIds) {
            pinGpdbRefCount += static_cast<uint64_t>(gpdbPins.size());
            for (int gpdbPinId : gpdbPins) {
                pinChecksum = pinChecksum * 1099511628211ULL + static_cast<uint64_t>(gpdbPinId + 1) * 67ULL;
            }
        }
    }

    summary["num_gr_nets"] = pybind11::int_(static_cast<uint64_t>(grdb.grNets.size()));
    summary["routeable_net_count"] = pybind11::int_(routeableCount);
    summary["noroute_net_count"] = pybind11::int_(norouteCount);
    summary["pin_group_count"] = pybind11::int_(pinGroupCount);
    summary["pin_grid_count"] = pybind11::int_(pinGridCount);
    summary["pin_gpdb_ref_count"] = pybind11::int_(pinGpdbRefCount);
    summary["bbox_checksum"] = pybind11::int_(bboxChecksum);
    summary["pin_checksum"] = pybind11::int_(pinChecksum);
    summary["noroute_checksum"] = pybind11::int_(norouteChecksum);
    return summary;
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
                                   float xl,
                                   float yl,
                                   float routeBinSizeX,
                                   float routeBinSizeY) {
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

    int maxNetId = -1;

    for (const auto& [netId, record] : records) {
        netIds.push_back(static_cast<int32_t>(netId));
        maxNetId = std::max(maxNetId, netId);

        int32_t hCount = 0;
        for (const auto& [rowIdx, intervals] : record.horizontal_by_row) {
            float segY = gridCenter(rowIdx, yl, routeBinSizeY);
            for (const IntInterval& interval : intervals) {
                hX1.push_back(gridCenter(interval.lo, xl, routeBinSizeX));
                hY.push_back(segY);
                hX2.push_back(gridCenter(interval.hi, xl, routeBinSizeX));
                ++hCount;
            }
        }
        hSegOffsets.push_back(hSegOffsets.back() + hCount);

        int32_t vCount = 0;
        for (const auto& [colIdx, intervals] : record.vertical_by_col) {
            float segX = gridCenter(colIdx, xl, routeBinSizeX);
            for (const IntInterval& interval : intervals) {
                vX.push_back(segX);
                vY1.push_back(gridCenter(interval.lo, yl, routeBinSizeY));
                vY2.push_back(gridCenter(interval.hi, yl, routeBinSizeY));
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
                                        float xl,
                                        float yl,
                                        float routeBinSizeX,
                                        float routeBinSizeY) {
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

struct PinAccessGrid {
    int x = 0;
    int y = 0;
};

int gridIndexFromDbu(const std::vector<int>& lines, float p) {
    const int ip = static_cast<int>(p);
    int idx = static_cast<int>(std::lower_bound(lines.begin(), lines.end(), ip) - lines.begin()) - 1;
    return std::max(0, std::min(idx, static_cast<int>(lines.size()) - 2));
}

// Pins skipped by GR setup (no access points on the enabled layers) never
// enter accessByGpdbPin. Approximate their access with the gcell containing
// the pin's absolute position so the net keeps its L-shape topology edges.
PinAccessGrid pinAccessOrPinGrid(const gr::GRDatabase& grdb,
                                 const std::unordered_map<int, PinAccessGrid>& accessByGpdbPin,
                                 int gpdbPinId,
                                 const gp::point_type& pinPos,
                                 int& fallbackCount) {
    auto it = accessByGpdbPin.find(gpdbPinId);
    if (it != accessByGpdbPin.end()) {
        return it->second;
    }
    ++fallbackCount;
    return PinAccessGrid{gridIndexFromDbu(grdb.gridlines[0], pinPos.first),
                         gridIndexFromDbu(grdb.gridlines[1], pinPos.second)};
}

struct LShapeEdgeRecord {
    int from = -1;
    int to = -1;
    int direction = 0;
};

int intervalOverlapLength(const std::vector<IntInterval>& intervals, int lo, int hi) {
    if (hi < lo) {
        std::swap(lo, hi);
    }
    int total = 0;
    for (const IntInterval& interval : intervals) {
        const int overlapLo = std::max(lo, interval.lo);
        const int overlapHi = std::min(hi, interval.hi);
        if (overlapHi > overlapLo) {
            total += overlapHi - overlapLo;
        }
    }
    return total;
}

int routeOverlapForLShape(const PackedTopologyRecord& routeRecord,
                          int x1,
                          int y1,
                          int x2,
                          int y2,
                          bool horizontalFirst) {
    int overlap = 0;
    if (horizontalFirst) {
        auto hIt = routeRecord.horizontal_by_row.find(y1);
        if (hIt != routeRecord.horizontal_by_row.end()) {
            overlap += intervalOverlapLength(hIt->second, x1, x2);
        }
        auto vIt = routeRecord.vertical_by_col.find(x2);
        if (vIt != routeRecord.vertical_by_col.end()) {
            overlap += intervalOverlapLength(vIt->second, y1, y2);
        }
    } else {
        auto vIt = routeRecord.vertical_by_col.find(x1);
        if (vIt != routeRecord.vertical_by_col.end()) {
            overlap += intervalOverlapLength(vIt->second, y1, y2);
        }
        auto hIt = routeRecord.horizontal_by_row.find(y2);
        if (hIt != routeRecord.horizontal_by_row.end()) {
            overlap += intervalOverlapLength(hIt->second, x1, x2);
        }
    }
    return overlap;
}

PackedTopologyRecord buildRouteRecordForNet(const gr::GRDatabase& grdb,
                                            const std::vector<int>& rawRoute,
                                            int targetNetId,
                                            bool routeFailed,
                                            int maxGap) {
    PackedTopologyRecord record;
    record.net_id = targetNetId;
    record.route_failed = routeFailed;
    if (!routeFailed && !rawRoute.empty()) {
        int routeEnd = std::min(static_cast<int>(rawRoute.size()), rawRoute[0]);
        for (int i = 1; i + 1 < routeEnd; i += 2) {
            int p = rawRoute[i];
            int span = rawRoute[i + 1];
            if (span <= 0) {
                continue;
            }
            ++record.raw_wire_count;
            auto [l, x, y] = decodeRoutePoint(grdb, p);
            bool vertical = ((l & 1) ^ grdb.m1direction);
            int x1 = x;
            int y1 = y;
            int x2 = vertical ? x : x + span;
            int y2 = vertical ? y + span : y;
            if (!vertical && y1 == y2) {
                record.horizontal_by_row[y1].push_back({std::min(x1, x2), std::max(x1, x2)});
            } else if (vertical && x1 == x2) {
                record.vertical_by_col[x1].push_back({std::min(y1, y2), std::max(y1, y2)});
            } else {
                ++record.invalid_wire_count;
            }
        }
    }
    finalizePackedTopologyRecord(record, maxGap);
    return record;
}

int chooseLShapeDirection(const PackedTopologyRecord& routeRecord,
                          const gp::point_type& pinPosA,
                          const gp::point_type& pinPosB,
                          const PinAccessGrid& accessA,
                          const PinAccessGrid& accessB) {
    constexpr float kCoordEps = 1e-3f;
    if (std::abs(pinPosA.first - pinPosB.first) <= kCoordEps ||
        std::abs(pinPosA.second - pinPosB.second) <= kCoordEps) {
        return 2;
    }
    if (accessA.y == accessB.y) {
        return 0;
    }
    if (accessA.x == accessB.x) {
        return 1;
    }
    const int hOverlap = routeOverlapForLShape(
        routeRecord,
        accessA.x,
        accessA.y,
        accessB.x,
        accessB.y,
        true);
    const int vOverlap = routeOverlapForLShape(
        routeRecord,
        accessA.x,
        accessA.y,
        accessB.x,
        accessB.y,
        false);
    return hOverlap >= vOverlap ? 0 : 1;
}

pybind11::dict buildLShapeTopologyPack(gr::RouteForce& routeforce,
                                       pybind11::dict pinNameToId,
                                       pybind11::dict netNameToId,
                                       pybind11::handle flatNet2PinMapValue,
                                       pybind11::handle flatNet2PinStartValue,
                                       int numPins,
                                       int numNets,
                                       int routeNumBinsX,
                                       int routeNumBinsY,
                                       float xl,
                                       float yl,
                                       float routeBinSizeX,
                                       float routeBinSizeY) {
    if (numPins <= 0 || numNets < 0) {
        throw std::runtime_error("GGR L-shape topology pack requires positive num_pins and non-negative num_nets");
    }
    std::vector<int> flatNet2PinMap = intArrayToVector(flatNet2PinMapValue, "flat_net2pin_map");
    std::vector<int> flatNet2PinStart = intArrayToVector(flatNet2PinStartValue, "flat_net2pin_start_map");
    if (static_cast<int>(flatNet2PinStart.size()) != numNets + 1) {
        throw std::runtime_error("GGR L-shape topology pack flat_net2pin_start_map size must equal num_nets + 1");
    }
    if (flatNet2PinStart.empty() || flatNet2PinStart.front() != 0) {
        throw std::runtime_error("GGR L-shape topology pack flat_net2pin_start_map must start at zero");
    }
    if (flatNet2PinStart.back() != static_cast<int>(flatNet2PinMap.size())) {
        throw std::runtime_error("GGR L-shape topology pack flat_net2pin_start_map end must equal flat_net2pin_map size");
    }
    if (static_cast<int>(flatNet2PinMap.size()) != numPins) {
        throw std::runtime_error("GGR L-shape topology pack requires flat_net2pin_map to cover every AutoDMP pin");
    }
    if (netNameToId.empty()) {
        throw std::runtime_error("GGR L-shape topology pack requires a non-empty AutoDMP net name map");
    }
    std::vector<char> flatPinSeen(static_cast<size_t>(numPins), 0);
    for (int netId = 0; netId < numNets; ++netId) {
        if (flatNet2PinStart[netId] > flatNet2PinStart[netId + 1]) {
            throw std::runtime_error("GGR L-shape topology pack flat_net2pin_start_map must be monotonic");
        }
        for (int idx = flatNet2PinStart[netId]; idx < flatNet2PinStart[netId + 1]; ++idx) {
            const int pinId = flatNet2PinMap[static_cast<size_t>(idx)];
            if (pinId < 0 || pinId >= numPins) {
                throw std::runtime_error("GGR L-shape topology pack flat_net2pin_map contains pin id out of range");
            }
            if (flatPinSeen[static_cast<size_t>(pinId)]) {
                throw std::runtime_error("GGR L-shape topology pack flat_net2pin_map contains duplicate pin id");
            }
            flatPinSeen[static_cast<size_t>(pinId)] = 1;
        }
    }

    auto rawRoutes = routeforce.getRawRoutes();
    auto& grdb = routeforce.getGRDatabase();
    const auto& gpNets = grdb.gpdb.getNets();
    const auto& gpPins = grdb.gpdb.getPins();

    std::unordered_map<int, PinAccessGrid> accessByGpdbPin;
    for (size_t netId = 0; netId < grdb.grNets.size(); ++netId) {
        auto& grNet = grdb.grNets[netId];
        const auto& grPins = grNet.getPins();
        for (size_t grPinIdx = 0; grPinIdx < grPins.size(); ++grPinIdx) {
            if (grPins[grPinIdx].empty()) {
                continue;
            }
            auto [l, x, y] = decodeRoutePoint(grdb, grPins[grPinIdx].front());
            if (grPinIdx >= grNet.pin2gpdbPinIds.size()) {
                continue;
            }
            for (int gpdbPinId : grNet.pin2gpdbPinIds[grPinIdx]) {
                accessByGpdbPin.emplace(gpdbPinId, PinAccessGrid{x, y});
            }
        }
    }

    std::vector<LShapeEdgeRecord> edgeRecords;
    int routeFailedCount = 0;
    int skippedUnmappedNetCount = 0;
    int unmappedPinCount = 0;
    const std::vector<int> emptyRawRoute;

    for (size_t gpNetId = 0; gpNetId < gpNets.size(); ++gpNetId) {
        const auto& gpNet = gpNets[gpNetId];
        int rawdbNetId = gpNet.getOriDBId();
        std::string netName = "";
        if (0 <= rawdbNetId && rawdbNetId < static_cast<int>(grdb.rawdb.nets.size())) {
            netName = grdb.rawdb.nets[rawdbNetId]->name;
        } else {
            netName = gpNet.getName();
        }
        const int targetNetId = mappedLShapeNetId(netNameToId, netName);
        if (targetNetId < 0) {
            ++skippedUnmappedNetCount;
            continue;
        }
        if (targetNetId < 0 || targetNetId >= numNets) {
            throw std::runtime_error("GGR L-shape topology pack net id out of range for net '" + netName + "'");
        }

        const std::vector<int>& rawRoute = gpNetId < rawRoutes.size() ? rawRoutes[gpNetId] : emptyRawRoute;
        const bool routeFailed = rawRoute.empty() || rawRoute[0] < 0;
        if (routeFailed) {
            ++routeFailedCount;
        }
        PackedTopologyRecord routeRecord = buildRouteRecordForNet(
            grdb,
            rawRoute,
            targetNetId,
            routeFailed,
            1);

        std::vector<int> mappedGpdbPins;
        std::vector<int> mappedGlobalPins;
        std::set<int> seenGlobalPins;
        for (int gpdbPinId : gpNet.pins()) {
            if (gpdbPinId < 0 || gpdbPinId >= static_cast<int>(gpPins.size())) {
                continue;
            }
            const int globalPinId = mappedPinId(pinNameToId, gpPins[gpdbPinId]);
            if (globalPinId < 0 || globalPinId >= numPins) {
                throw std::runtime_error("GGR L-shape topology pack pin id out of range");
            }
            if (seenGlobalPins.insert(globalPinId).second) {
                mappedGpdbPins.push_back(gpdbPinId);
                mappedGlobalPins.push_back(globalPinId);
            }
        }

        if (mappedGlobalPins.size() < 2) {
            continue;
        }
        for (size_t i = 1; i < mappedGlobalPins.size(); ++i) {
            const int gpdbA = mappedGpdbPins[i - 1];
            const int gpdbB = mappedGpdbPins[i];
            const gp::point_type pinPosA = grdb.gpdb.getAbsolutePinPos(gpPins[gpdbA]);
            const gp::point_type pinPosB = grdb.gpdb.getAbsolutePinPos(gpPins[gpdbB]);
            const PinAccessGrid accessA =
                pinAccessOrPinGrid(grdb, accessByGpdbPin, gpdbA, pinPosA, unmappedPinCount);
            const PinAccessGrid accessB =
                pinAccessOrPinGrid(grdb, accessByGpdbPin, gpdbB, pinPosB, unmappedPinCount);
            const int direction = chooseLShapeDirection(
                routeRecord, pinPosA, pinPosB, accessA, accessB);
            edgeRecords.push_back({
                mappedGlobalPins[i - 1],
                mappedGlobalPins[i],
                direction,
            });
        }
    }

    std::vector<int32_t> pinRelateX(static_cast<size_t>(numPins));
    std::vector<int32_t> pinRelateY(static_cast<size_t>(numPins));
    std::vector<int32_t> pinFa(static_cast<size_t>(numPins), static_cast<int32_t>(-1));
    for (int pinId = 0; pinId < numPins; ++pinId) {
        pinRelateX[pinId] = pinId;
        pinRelateY[pinId] = pinId;
    }

    std::vector<int32_t> outDegree(static_cast<size_t>(numPins), static_cast<int32_t>(0));
    for (const LShapeEdgeRecord& edge : edgeRecords) {
        if (edge.from < 0 || edge.from >= numPins || edge.to < 0 || edge.to >= numPins) {
            throw std::runtime_error("GGR L-shape topology pack generated edge index out of range");
        }
        ++outDegree[static_cast<size_t>(edge.from)];
        if (pinFa[edge.to] < 0) {
            pinFa[edge.to] = edge.from;
        }
    }

    std::vector<int32_t> flatPinToStart(static_cast<size_t>(numPins) + 1, static_cast<int32_t>(0));
    for (int pinId = 0; pinId < numPins; ++pinId) {
        flatPinToStart[static_cast<size_t>(pinId) + 1] =
            flatPinToStart[static_cast<size_t>(pinId)] + outDegree[static_cast<size_t>(pinId)];
    }

    const size_t numEdges = edgeRecords.size();
    std::vector<int32_t> flatPinFrom(numEdges);
    std::vector<int32_t> flatPinTo(numEdges);
    std::vector<int32_t> edgeDirections(numEdges);
    std::vector<int32_t> cursor(flatPinToStart.begin(), flatPinToStart.end() - 1);
    for (const LShapeEdgeRecord& edge : edgeRecords) {
        const size_t writeIdx = static_cast<size_t>(cursor[static_cast<size_t>(edge.from)]++);
        flatPinFrom[writeIdx] = static_cast<int32_t>(edge.from);
        flatPinTo[writeIdx] = static_cast<int32_t>(edge.to);
        edgeDirections[writeIdx] = static_cast<int32_t>(edge.direction);
    }

    std::vector<int32_t> netFlatTopoSort;
    std::vector<int32_t> netFlatTopoSortStart;
    std::vector<int32_t> netVertexStart;
    netFlatTopoSort.reserve(flatNet2PinMap.size());
    for (int pinId : flatNet2PinMap) {
        netFlatTopoSort.push_back(static_cast<int32_t>(pinId));
    }
    netFlatTopoSortStart.reserve(flatNet2PinStart.size());
    netVertexStart.reserve(flatNet2PinStart.size());
    for (int start : flatNet2PinStart) {
        netFlatTopoSortStart.push_back(static_cast<int32_t>(start));
        netVertexStart.push_back(static_cast<int32_t>(start));
    }

    std::vector<int32_t> netSteinerStart(static_cast<size_t>(numNets) + 1, static_cast<int32_t>(numPins));

    pybind11::dict metadata;
    metadata["schema_version"] = pybind11::int_(1);
    metadata["num_pins"] = pybind11::int_(numPins);
    metadata["num_vertices"] = pybind11::int_(numPins);
    metadata["num_edges"] = pybind11::int_(static_cast<int>(flatPinFrom.size()));
    metadata["num_nets"] = pybind11::int_(numNets);
    metadata["route_grid_shape"] = pybind11::make_tuple(routeNumBinsX, routeNumBinsY);
    metadata["route_failed_count"] = pybind11::int_(routeFailedCount);
    metadata["skipped_unmapped_net_count"] = pybind11::int_(skippedUnmappedNetCount);
    metadata["unmapped_pin_access_count"] = pybind11::int_(unmappedPinCount);
    metadata["xl"] = pybind11::float_(xl);
    metadata["yl"] = pybind11::float_(yl);
    metadata["route_bin_size_x"] = pybind11::float_(routeBinSizeX);
    metadata["route_bin_size_y"] = pybind11::float_(routeBinSizeY);

    pybind11::dict pack;
    pack["pin_relate_x"] = vectorToNumpy(pinRelateX);
    pack["pin_relate_y"] = vectorToNumpy(pinRelateY);
    pack["net_vertex_start"] = vectorToNumpy(netVertexStart);
    pack["net_steiner_start"] = vectorToNumpy(netSteinerStart);
    pack["pin_fa"] = vectorToNumpy(pinFa);
    pack["flat_pin_to"] = vectorToNumpy(flatPinTo);
    pack["flat_pin_from"] = vectorToNumpy(flatPinFrom);
    pack["flat_pin_to_start"] = vectorToNumpy(flatPinToStart);
    pack["net_flat_topo_sort"] = vectorToNumpy(netFlatTopoSort);
    pack["net_flat_topo_sort_start"] = vectorToNumpy(netFlatTopoSortStart);
    pack["edge_l_directions"] = vectorToNumpy(edgeDirections);
    pack["metadata"] = metadata;
    return pack;
}

}  // namespace

bool loadGRParams(const pybind11::dict& kwargs) {
    gr::grSetting.reset();
    if (kwargs.contains("clock_net_names")) {
        const auto names = kwargs["clock_net_names"].cast<std::vector<std::string>>();
        gr::grSetting.clockNetNames.insert(names.begin(), names.end());
    }
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

    if (kwargs.contains("bottom_routing_layer")) {
        gr::grSetting.bottomRoutingLayer = kwargs["bottom_routing_layer"].cast<std::string>();
    }

    if (kwargs.contains("top_routing_layer")) {
        gr::grSetting.topRoutingLayer = kwargs["top_routing_layer"].cast<std::string>();
    }

    if (kwargs.contains("rrrIters")) {
        gr::grSetting.rrrIters = kwargs["rrrIters"].cast<int>();
    }

    if (kwargs.contains("threads")) {
        gr::grSetting.requestedWorkers = kwargs["threads"].cast<int>();
    }
    if (gr::grSetting.requestedWorkers <= 0) {
        throw std::runtime_error("gpugr threads must be positive");
    }

    if (kwargs.contains("route_guide")) {
        gr::grSetting.routeGuideFile = kwargs["route_guide"].cast<std::string>();
    }

    if (kwargs.contains("backend")) {
        gr::grSetting.backend = gr::parseGGRBackend(kwargs["backend"].cast<std::string>());
    }

    if (gr::grSetting.backend == gr::GGRBackend::CPU_PR && gr::grSetting.rrrIters > 0) {
        throw std::runtime_error(
            "backend=cpu_pr operation=load_gr_params only supports rrr_iters=0; CPU RRR is unsupported");
    }
    if (gr::grSetting.backend == gr::GGRBackend::CPU_PR_MT && gr::grSetting.rrrIters > 0) {
        throw std::runtime_error(
            "backend=cpu_pr_mt operation=load_gr_params only supports rrr_iters=0; CPU RRR is unsupported");
    }
    if (gr::grSetting.rrrIters < 0) {
        throw std::runtime_error("gpugr rrr_iters must be non-negative");
    }
#if !ENABLE_CUDA
    if (gr::grSetting.backend == gr::GGRBackend::CUDA) {
        throw std::runtime_error(
            "backend=cuda operation=load_gr_params requested, but CUDA support is unavailable");
    }
#endif

    return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<gr::GRDatabase, std::shared_ptr<gr::GRDatabase>>(m, "GRDatabase")
        .def(pybind11::init<std::shared_ptr<db::Database>, std::shared_ptr<gp::GPDatabase>>())
        .def("report_gr_stat", &gr::GRDatabase::reportGRStat)
        .def("grnet_summary", [](gr::GRDatabase& grdb) { return buildGRNetSummary(grdb); })
        .def("setup_profile", [](const gr::GRDatabase& grdb) { return buildGRDatabaseSetupProfile(grdb); })
        .def("setup_capacity", &gr::GRDatabase::setupCapacity)
        .def("setup_wiredist", &gr::GRDatabase::setupWireDist)
        .def("setup_obs", &gr::GRDatabase::setupObs)
        .def("setup_grnets", &gr::GRDatabase::setupGrNets)
        .def("routing_layer_range", &gr::GRDatabase::routingLayerRange)
        .def("routing_layer_names", &gr::GRDatabase::routingLayerNames);
    pybind11::class_<gr::RouteForce, std::shared_ptr<gr::RouteForce>>(m, "RouteForce")
        .def(pybind11::init<std::shared_ptr<gr::GRDatabase>>(), py::keep_alive<1, 2>())
        .def("run_ggr", &gr::RouteForce::run_ggr)
        .def("run_stats", [](const gr::RouteForce& routeforce) {
            const auto stats = routeforce.getRunStats();
            pybind11::dict result;
            result["schema_version"] = stats.schemaVersion;
            result["requested_workers"] = stats.requestedWorkers;
            result["effective_workers"] = stats.effectiveWorkers;
            result["batch_count"] = stats.batchCount;
            result["parallel_batch_count"] = stats.parallelBatchCount;
            result["conflict_batch_count"] = stats.conflictBatchCount;
            result["serial_fallback_batch_count"] = stats.serialFallbackBatchCount;
            result["max_batch_size"] = stats.maxBatchSize;
            result["batch_size_limit"] = stats.batchSizeLimit;
            result["batch_conflict_scan_limit"] = stats.batchConflictScanLimit;
            result["candidate_batch_count"] = stats.candidateBatchCount;
            result["candidate_batch_size_limit"] = stats.candidateBatchSizeLimit;
            result["packed_conflict_batch_count"] = stats.packedConflictBatchCount;
            result["scratch_bytes_per_worker"] = stats.scratchBytesPerWorker;
            result["memory_budget_bytes"] = stats.memoryBudgetBytes;
            result["route_hash64"] = stats.routeHash64;
            result["wire_map_hash64"] = stats.wireMapHash64;
            result["via_map_hash64"] = stats.viaMapHash64;
            result["worker_clamp_reason"] = stats.workerClampReason;
            result["terminal_status"] = stats.terminalStatus;
            result["flute_parallel"] = stats.fluteParallel;
            result["requested_rrr_iters"] = stats.requestedRrrIters;
            result["completed_rrr_iters"] = stats.completedRrrIters;
            result["best_state_restored"] = stats.bestStateRestored;
            result["maze_route_failure_count"] = stats.mazeRouteFailureCount;
            result["maze_failure_reasons"] = stats.mazeFailureReasons;
            result["conflict_fallback_count"] = stats.conflictFallbackCount;
            result["peak_memory_bytes"] = stats.peakMemoryBytes;
            result["memory_accounting"] = stats.memoryAccounting;
            result["maze_batch_size_limit"] = stats.mazeBatchSizeLimit;
            result["maze_max_batch_size"] = stats.mazeMaxBatchSize;
            pybind11::list rrrIterations;
            auto iterationStats = [](const auto& iteration) {
                pybind11::dict item;
                item["iteration"] = iteration.iteration;
                item["selected_net_count"] = iteration.selectedNetCount;
                item["rerouted_net_count"] = iteration.reroutedNetCount;
                item["failed_net_count"] = iteration.failedNetCount;
                item["overflow_net_count"] = iteration.overflowNetCount;
                item["overflow_wire_resource_count"] = iteration.overflowWireResourceCount;
                item["overflow_via_resource_count"] = iteration.overflowViaResourceCount;
                item["overflow_wire_excess"] = iteration.overflowWireExcess;
                item["overflow_via_excess"] = iteration.overflowViaExcess;
                item["wire_length"] = iteration.wireLength;
                item["via_count"] = iteration.viaCount;
                item["routed_net_count"] = iteration.routedNetCount;
                item["failed_route_count"] = iteration.failedRouteCount;
                item["unrouted_net_count"] = iteration.unroutedNetCount;
                item["route_hash64"] = iteration.routeHash64;
                item["wire_map_hash64"] = iteration.wireMapHash64;
                item["via_map_hash64"] = iteration.viaMapHash64;
                item["best_state"] = iteration.bestState;
                item["parallel_batch_count"] = iteration.parallelBatchCount;
                item["conflict_fallback_count"] = iteration.conflictFallbackCount;
                item["search_seconds"] = iteration.searchSeconds;
                item["fallback_search_seconds"] = iteration.fallbackSearchSeconds;
                item["max_active_searches"] = iteration.maxActiveSearches;
                item["worker_search_seconds"] = iteration.workerSearchSeconds;
                item["coarse_build_seconds"] = iteration.coarseBuildSeconds;
                item["maze_coarsening_scale"] = iteration.mazeCoarseningScale;
                item["maze_coarse_x_size"] = iteration.mazeCoarseXSize;
                item["maze_coarse_y_size"] = iteration.mazeCoarseYSize;
                item["coarse_expanded_states"] = iteration.coarseExpandedStates;
                item["fine_expanded_states"] = iteration.fineExpandedStates;
                item["corridor_retries"] = iteration.corridorRetries;
                item["full_search_fallbacks"] = iteration.fullSearchFallbacks;
                item["cost_rebuild_seconds"] = iteration.costRebuildSeconds;
                item["commit_seconds"] = iteration.commitSeconds;
                item["rollback_seconds"] = iteration.rollbackSeconds;
                item["total_seconds"] = iteration.totalSeconds;
                return item;
            };
            if (stats.hasRouteQor) {
                result["initial_route_qor"] = iterationStats(stats.initialRouteQor);
                result["accepted_route_qor"] = iterationStats(stats.acceptedRouteQor);
            }
            for (const auto& iteration : stats.rrrIterations) {
                rrrIterations.append(iterationStats(iteration));
            }
            result["rrr_iterations"] = rrrIterations;
            result["cost_map_seconds"] = stats.costMapSeconds;
            result["batch_schedule_seconds"] = stats.batchScheduleSeconds;
            result["prepare_seconds"] = stats.prepareSeconds;
            result["flute_wait_seconds"] = stats.fluteWaitSeconds;
            result["flute_wall_seconds"] = stats.fluteWallSeconds;
            result["candidate_seconds"] = stats.candidateSeconds;
            result["commit_seconds"] = stats.commitSeconds;
            result["serial_fallback_seconds"] = stats.serialFallbackSeconds;
            result["overflow_mark_seconds"] = stats.overflowMarkSeconds;
            result["total_seconds"] = stats.totalSeconds;
            return result;
        })
        .def("num_ovfl_nets", &gr::RouteForce::getNumOvflNets)
        .def("gcell_steps", &gr::RouteForce::getGcellStep)
        .def("microns", &gr::RouteForce::getMicrons)
        .def("layer_pitch", &gr::RouteForce::getLayerPitch)
        .def("layer_width", &gr::RouteForce::getLayerWidth)
        .def("routing_layer_range", &gr::RouteForce::getRoutingLayerRange)
        .def("routing_layer_names", &gr::RouteForce::getRoutingLayerNames)
        .def("raw_routes", &gr::RouteForce::getRawRoutes, py::return_value_policy::move)
        .def("route_entries", [](gr::RouteForce& routeforce) { return buildRouteEntries(routeforce); })
        .def("timing_route_pack", &exportTimingRoutes)
        .def(
            "same_net_topology_pack",
            [](gr::RouteForce& routeforce,
               pybind11::dict netNameToId,
               int maxGap,
               int routeNumBinsX,
               int routeNumBinsY,
               float xl,
               float yl,
               float routeBinSizeX,
               float routeBinSizeY) {
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
        .def(
            "l_shape_topology_pack",
            [](gr::RouteForce& routeforce,
               pybind11::dict pinNameToId,
               pybind11::dict netNameToId,
               pybind11::object flatNet2PinMap,
               pybind11::object flatNet2PinStart,
               int numPins,
               int numNets,
               int routeNumBinsX,
               int routeNumBinsY,
               float xl,
               float yl,
               float routeBinSizeX,
               float routeBinSizeY) {
                return buildLShapeTopologyPack(
                    routeforce,
                    pinNameToId,
                    netNameToId,
                    flatNet2PinMap,
                    flatNet2PinStart,
                    numPins,
                    numNets,
                    routeNumBinsX,
                    routeNumBinsY,
                    xl,
                    yl,
                    routeBinSizeX,
                    routeBinSizeY);
            },
            pybind11::arg("pin_name_to_id"),
            pybind11::arg("net_name_to_id"),
            pybind11::arg("flat_net2pin_map"),
            pybind11::arg("flat_net2pin_start_map"),
            pybind11::arg("num_pins"),
            pybind11::arg("num_nets"),
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
    }, py::keep_alive<0, 1>());
    m.def("load_gr_params", &loadGRParams, "Parse input args to DB and return graph information");
    m.def("cuda_enabled", []() { return static_cast<bool>(ENABLE_CUDA); });
    m.def("resolved_backend", []() { return gr::ggrBackendName(gr::grSetting.resolvedBackend); });
    m.def("read_flute", &Flute::readLUT, "Read Flute LUT");
}

}  // namespace Xplace
