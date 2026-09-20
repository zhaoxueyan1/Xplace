#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cugr_api {

struct RunOptions {
    std::vector<std::string> lefs;
    std::string def_file;
    std::string powv_file;
    std::string post_file;
    std::string design_identity;
    int threads = 1;
    int total_passes = 1;
    int route_x_size = 0;
    int route_y_size = 0;
    std::string bottom_routing_layer;
    std::string top_routing_layer;
};

struct RouteEntry {
    int type = 0;  // 0: planar wire, 1: via
    int layer = -1;
    int x1 = -1;
    int y1 = -1;
    int x2 = -1;
    int y2 = -1;
};

struct TopologyNode {
    int layer = -1;
    int x = -1;
    int y = -1;
    int pin_index = -1;
    int parent_index = -1;
};

struct NetResult {
    std::string name;
    bool routed = false;
    bool route_failed = false;
    std::string route_failed_reason;
    std::vector<TopologyNode> nodes;
    std::vector<RouteEntry> entries;
    std::vector<TopologyNode> pin_access;
};

struct RunResult {
    int grid_x = 0;
    int grid_y = 0;
    int num_layers = 0;
    std::vector<int> x_lines;
    std::vector<int> y_lines;
    std::vector<std::string> layer_names;
    std::vector<int> layer_directions;
    std::vector<int32_t> layer_edge_offsets;
    std::vector<float> capacity;
    std::vector<float> fixed_usage;
    std::vector<float> movable_usage;
    std::vector<float> wire_usage;
    std::vector<float> via_usage;
    std::vector<NetResult> nets;
    double wirelength_dbu = 0.0;
    double short_vio_area = 0.0;
    int via_count = 0;
    int overflow_net_count = 0;
    int invalid_layer_warning_count = 0;
    int total_passes = 0;
    int threads = 0;
    int pass0_worker_count = 0;
    bool session_cache_hit = false;
    std::string session_mode;
    int routing_layer_begin = 0;
    int routing_layer_end = -1;
    double parse_sec = 0.0;
    double refresh_sec = 0.0;
    double route_sec = 0.0;
    double pack_sec = 0.0;
    double total_sec = 0.0;
    double peak_rss_mb = 0.0;
};

RunResult run(const RunOptions& options);
void reset();

const char* source_commit();
bool source_dirty();

}  // namespace cugr_api
