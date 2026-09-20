#include "cugr_api.h"

#include <mutex>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "db/Database.h"
#include "db/Setting.h"
#include "flute/flute.h"
#include "gr_db/GrDatabase.h"
#include "multi_net/Router.h"
#include "rsyn/io/reader/ISPD2018Reader.h"
#include "rsyn/session/Session.h"
#include "utils/log.h"

double LARGE_NUM = 100000000;

namespace cugr_api {
namespace {

std::mutex session_mutex;
bool session_active = false;
bool lut_loaded = false;
bool session_initialized = false;
std::string session_key;

class ActiveSessionGuard {
public:
    ActiveSessionGuard() {
        std::lock_guard<std::mutex> lock(session_mutex);
        if (session_active) {
            throw std::runtime_error("CUGR supports one active session per process");
        }
        session_active = true;
    }

    ~ActiveSessionGuard() {
        std::lock_guard<std::mutex> lock(session_mutex);
        session_active = false;
    }
};

template <typename Function>
auto run_phase(const char* phase, Function&& function) -> decltype(function()) {
    try {
        return std::forward<Function>(function)();
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("CUGR ") + phase + " phase failed: " + error.what());
    } catch (...) {
        throw std::runtime_error(std::string("CUGR ") + phase + " phase failed with an unknown exception");
    }
}

void validate_options(const RunOptions& options) {
    if (options.lefs.empty()) {
        throw std::invalid_argument("CUGR requires at least one LEF file");
    }
    if (options.def_file.empty()) {
        throw std::invalid_argument("CUGR requires a DEF file");
    }
    for (const auto& lef : options.lefs) {
        if (!std::filesystem::is_regular_file(lef)) {
            throw std::invalid_argument("CUGR LEF file does not exist: " + lef);
        }
    }
    if (!std::filesystem::is_regular_file(options.def_file)) {
        throw std::invalid_argument("CUGR DEF file does not exist: " + options.def_file);
    }
    if (options.threads <= 0) {
        throw std::invalid_argument("CUGR threads must be positive");
    }
    if (options.total_passes <= 0) {
        throw std::invalid_argument("CUGR total_passes must be positive");
    }
    if ((options.route_x_size == 0) != (options.route_y_size == 0) ||
        options.route_x_size < 0 || options.route_y_size < 0) {
        throw std::invalid_argument("CUGR route grid dimensions must both be zero or both be positive");
    }
}

void load_lut_once(const RunOptions& options) {
    if (lut_loaded) {
        return;
    }
    if (options.powv_file.empty() || options.post_file.empty()) {
        throw std::invalid_argument("CUGR requires explicit FLUTE POWV9.dat and POST9.dat paths");
    }
    if (!std::filesystem::is_regular_file(options.powv_file) ||
        !std::filesystem::is_regular_file(options.post_file)) {
        throw std::invalid_argument("CUGR FLUTE LUT files do not exist");
    }
    readLUTFromFiles(options.powv_file.c_str(), options.post_file.c_str());
    lut_loaded = true;
}

std::string normalize_layer_name(std::string value);

std::string make_session_key(const RunOptions& options) {
    std::ostringstream key;
    key << "lefs=";
    for (const auto& lef : options.lefs) key << lef.size() << ':' << lef << ';';
    key << "grid=" << options.route_x_size << 'x' << options.route_y_size;
    key << "|bottom=" << normalize_layer_name(options.bottom_routing_layer);
    key << "|top=" << normalize_layer_name(options.top_routing_layer);
    key << "|design=" << options.design_identity;
    return key.str();
}

void configure_settings(const RunOptions& options) {
    db::setting.numThreads = options.threads;
    db::setting.rrrIterLimit = options.total_passes;
    db::setting.routeXSize = options.route_x_size;
    db::setting.routeYSize = options.route_y_size;
    db::setting.multiNetVerbose = db::VerboseLevelT::LOW;
    db::setting.dbVerbose = db::VerboseLevelT::LOW;
}

void reset_state_unlocked() {
    grDatabase = gr::GrDatabase();
    database = db::Database();
    Rsyn::Session::reset();
    session_initialized = false;
    session_key.clear();
}

std::string normalize_layer_name(std::string value) {
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char c) {
        return !std::isspace(c);
    }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char c) {
                    return !std::isspace(c);
                }).base(),
                value.end());
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return value;
}

int resolve_layer_name(const std::string& requested, int fallback, const char* bound_name) {
    const std::string normalized = normalize_layer_name(requested);
    if (normalized.empty()) return fallback;
    for (int layer = 0; layer < static_cast<int>(database.getLayerNum()); ++layer) {
        if (normalize_layer_name(database.getLayer(layer).name) == normalized) return layer;
    }
    throw std::invalid_argument("Unknown CUGR " + std::string(bound_name) + " routing layer '" + requested + "'");
}

void resolve_layer_window(const RunOptions& options) {
    const int layer_count = static_cast<int>(database.getLayerNum());
    if (layer_count <= 0) throw std::runtime_error("CUGR requires at least one routing layer");
    db::setting.routeLayerBegin = resolve_layer_name(options.bottom_routing_layer, 0, "bottom");
    db::setting.routeLayerEnd = resolve_layer_name(options.top_routing_layer, layer_count - 1, "top");
    if (db::setting.routeLayerBegin > db::setting.routeLayerEnd) {
        throw std::invalid_argument("CUGR bottom routing layer is above top routing layer");
    }
}

TopologyNode node_value(const std::shared_ptr<gr::GrSteiner>& node) {
    return {node->layerIdx, node->x, node->y, node->pinIdx, -1};
}

void collect_tree(const std::shared_ptr<gr::GrSteiner>& node,
                  NetResult& result,
                  std::unordered_map<const gr::GrSteiner*, int>& node_indices) {
    const int parent_index = node->parent ? node_indices.at(node->parent.get()) : -1;
    const int node_index = static_cast<int>(result.nodes.size());
    node_indices.emplace(node.get(), node_index);
    TopologyNode value = node_value(node);
    value.parent_index = parent_index;
    result.nodes.push_back(value);

    for (const auto& child : node->children) {
        RouteEntry entry;
        if (child->layerIdx == node->layerIdx) {
            entry.type = 0;
            entry.layer = node->layerIdx;
        } else {
            entry.type = 1;
            entry.layer = std::min(node->layerIdx, child->layerIdx);
        }
        entry.x1 = node->x;
        entry.y1 = node->y;
        entry.x2 = child->x;
        entry.y2 = child->y;
        result.entries.push_back(entry);
        collect_tree(child, result, node_indices);
    }
}

RunResult collect_result(int pass0_worker_count, bool cache_hit, const std::string& mode) {
    RunResult result;
    result.grid_x = grDatabase.getNumGrPoint(X);
    result.grid_y = grDatabase.getNumGrPoint(Y);
    result.num_layers = static_cast<int>(database.getLayerNum());
    result.threads = db::setting.numThreads;
    result.pass0_worker_count = pass0_worker_count;
    result.session_cache_hit = cache_hit;
    result.session_mode = mode;
    result.total_passes = db::setting.rrrIterLimit;
    result.routing_layer_begin = db::setting.routeLayerBegin;
    result.routing_layer_end = db::setting.routeLayerEnd;
    result.invalid_layer_warning_count = db::invalidLayerWarningCount();

    result.x_lines.reserve(static_cast<size_t>(result.grid_x + 1));
    result.y_lines.reserve(static_cast<size_t>(result.grid_y + 1));
    for (int x = 0; x <= result.grid_x; ++x) result.x_lines.push_back(grDatabase.getCoor(x, X));
    for (int y = 0; y <= result.grid_y; ++y) result.y_lines.push_back(grDatabase.getCoor(y, Y));

    result.layer_names.reserve(result.num_layers);
    result.layer_directions.reserve(result.num_layers);
    result.layer_edge_offsets.push_back(0);
    for (int layer = 0; layer < result.num_layers; ++layer) {
        const auto& layer_data = database.getLayer(layer);
        result.layer_names.push_back(layer_data.name);
        const int direction = database.getLayerDir(layer);
        result.layer_directions.push_back(direction);
        const int gridline_count = grDatabase.getNumGrPoint(direction);
        const int edge_count = grDatabase.getNumGrEdge(layer);
        for (int gridline = 0; gridline < gridline_count; ++gridline) {
            for (int cp = 0; cp < edge_count; ++cp) {
                result.capacity.push_back(static_cast<float>(grDatabase.getWireCapacityValue(layer, gridline, cp)));
                result.fixed_usage.push_back(static_cast<float>(grDatabase.getFixedUsageValue(layer, gridline, cp)));
                result.movable_usage.push_back(
                    static_cast<float>(grDatabase.getMovableUsageValue(layer, gridline, cp)));
                result.wire_usage.push_back(static_cast<float>(grDatabase.getWireUsageValue(layer, gridline, cp)));
            }
        }
        result.layer_edge_offsets.push_back(static_cast<int32_t>(result.capacity.size()));
    }

    for (int layer = 0; layer < result.num_layers; ++layer) {
        for (int x = 0; x < result.grid_x; ++x) {
            for (int y = 0; y < result.grid_y; ++y) {
                result.via_usage.push_back(static_cast<float>(grDatabase.getViaUsageValue(layer, x, y)));
            }
        }
    }

    result.nets.reserve(grDatabase.nets.size());
    for (const auto& net : grDatabase.nets) {
        NetResult net_result;
        net_result.name = net.getName();
        net_result.routed = !net.gridTopo.empty();
        net_result.route_failed = net.routeFailed();
        net_result.route_failed_reason = net.routeFailureReason();
        for (const auto& pin : net.pinAccessBoxes) {
            if (pin.empty()) {
                net_result.pin_access.push_back({-1, -1, -1, -1, -1});
            } else {
                const auto& point = pin.front();
                net_result.pin_access.push_back({point.layerIdx, point.x, point.y, -1, -1});
            }
        }
        std::unordered_map<const gr::GrSteiner*, int> node_indices;
        for (const auto& tree : net.gridTopo) collect_tree(tree, net_result, node_indices);
        result.wirelength_dbu += net.getWirelength();
        for (const auto& entry : net_result.entries) {
            if (entry.type == 1) ++result.via_count;
        }
        if (grDatabase.hasVio(net)) ++result.overflow_net_count;
        result.nets.push_back(std::move(net_result));
    }
    result.short_vio_area = grDatabase.lastShortVioArea();
    return result;
}

}  // namespace

RunResult run(const RunOptions& options) {
    validate_options(options);
    ActiveSessionGuard active_session;
    const auto total_start = std::chrono::steady_clock::now();
    const std::string requested_key = make_session_key(options);
    bool cache_hit = false;
    std::string session_mode = "full_reparse_reset";
    double parse_sec = 0.0;
    double refresh_sec = 0.0;
    double route_sec = 0.0;
    double pack_sec = 0.0;

    try {
        if (!session_initialized || session_key != requested_key) {
            if (session_initialized) reset_state_unlocked();
            const auto parse_start = std::chrono::steady_clock::now();
            run_phase("parse", [&]() {
                Rsyn::Session::init();
                db::setting = db::Setting();
                configure_settings(options);
                load_lut_once(options);

                Rsyn::Json params;
                params["lefFiles"] = options.lefs;
                params["defFile"] = options.def_file;
                Rsyn::ISPD2018Reader reader;
                if (!reader.load(params)) {
                    throw std::runtime_error("failed to parse the supplied LEF/DEF");
                }
                database.init();
                resolve_layer_window(options);
                grDatabase = gr::GrDatabase();
                grDatabase.init();
            });
            const auto parse_end = std::chrono::steady_clock::now();
            parse_sec = std::chrono::duration<double>(parse_end - parse_start).count();
            session_key = requested_key;
            session_initialized = true;
            session_mode = "full_reparse_reset";
        } else {
            cache_hit = true;
            session_mode = "coordinate_refresh";
            const auto refresh_start = std::chrono::steady_clock::now();
            run_phase("refresh", [&]() {
                configure_settings(options);
                database.refreshPlacementFromDef(options.def_file);
                resolve_layer_window(options);
                grDatabase = gr::GrDatabase();
                grDatabase.init();
            });
            const auto refresh_end = std::chrono::steady_clock::now();
            refresh_sec = std::chrono::duration<double>(refresh_end - refresh_start).count();
        }

        int pass0_worker_count = 0;
        const auto route_start = std::chrono::steady_clock::now();
        run_phase("route", [&]() {
            Router router(false);
            router.run();
            pass0_worker_count = router.pass0WorkerCount();
        });
        const auto route_end = std::chrono::steady_clock::now();
        route_sec = std::chrono::duration<double>(route_end - route_start).count();
        const auto pack_start = std::chrono::steady_clock::now();
        RunResult result = run_phase("pack", [&]() {
            return collect_result(pass0_worker_count, cache_hit, session_mode);
        });
        const auto pack_end = std::chrono::steady_clock::now();
        pack_sec = std::chrono::duration<double>(pack_end - pack_start).count();
        result.parse_sec = parse_sec;
        result.refresh_sec = refresh_sec;
        result.route_sec = route_sec;
        result.pack_sec = pack_sec;
        result.total_sec = std::chrono::duration<double>(pack_end - total_start).count();
        result.peak_rss_mb = utils::mem_use::get_peak();
        return result;
    } catch (...) {
        reset_state_unlocked();
        throw;
    }
}

void reset() {
    std::lock_guard<std::mutex> lock(session_mutex);
    if (session_active) {
        throw std::runtime_error("cannot reset CUGR while a session is active");
    }
    reset_state_unlocked();
}

const char* source_commit() {
#ifdef CUGR_SOURCE_COMMIT
    return CUGR_SOURCE_COMMIT;
#else
    return "unknown";
#endif
}

bool source_dirty() {
#ifdef CUGR_SOURCE_DIRTY
    return CUGR_SOURCE_DIRTY != 0;
#else
    return true;
#endif
}

}  // namespace cugr_api
