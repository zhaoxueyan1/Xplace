#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <set>
#include <stdexcept>
#include <string>

#include "cugr_api.h"

namespace py = pybind11;

namespace {

template <typename T>
bool has(const py::dict& kwargs, const char* name, T& value) {
    const py::str key(name);
    if (!kwargs.contains(key)) return false;
    value = kwargs[key].cast<T>();
    return true;
}

cugr_api::RunOptions parse_options(const py::dict& kwargs) {
    static const std::set<std::string> known_keys = {
        "lefs", "lef", "def", "powv_file", "post_file", "design_identity",
        "threads", "total_passes", "rrr_iters", "route_num_bins_x",
        "route_num_bins_y", "bottom_routing_layer", "top_routing_layer",
    };
    for (const auto& item : kwargs) {
        if (!py::isinstance<py::str>(item.first)) {
            throw std::invalid_argument("CUGR run option keys must be strings");
        }
        const std::string key = item.first.cast<std::string>();
        if (!known_keys.count(key)) {
            throw std::invalid_argument("CUGR run received unknown option '" + key + "'");
        }
    }

    cugr_api::RunOptions options;
    const bool has_lefs = has(kwargs, "lefs", options.lefs);
    if (has_lefs && kwargs.contains("lef")) {
        throw std::invalid_argument("CUGR run accepts either 'lefs' or 'lef', not both");
    }
    if (!has_lefs) {
        std::string lef;
        if (!has(kwargs, "lef", lef)) {
            throw std::invalid_argument("CUGR run requires 'lefs' or 'lef'");
        }
        options.lefs.push_back(lef);
    }
    if (!has(kwargs, "def", options.def_file)) {
        throw std::invalid_argument("CUGR run requires 'def'");
    }
    has(kwargs, "powv_file", options.powv_file);
    has(kwargs, "post_file", options.post_file);
    has(kwargs, "design_identity", options.design_identity);
    has(kwargs, "threads", options.threads);
    has(kwargs, "route_num_bins_x", options.route_x_size);
    has(kwargs, "route_num_bins_y", options.route_y_size);
    has(kwargs, "bottom_routing_layer", options.bottom_routing_layer);
    has(kwargs, "top_routing_layer", options.top_routing_layer);

    int extra_rrr_iters = 0;
    if (kwargs.contains("total_passes") && kwargs.contains("rrr_iters")) {
        throw std::invalid_argument("CUGR run accepts either 'total_passes' or 'rrr_iters', not both");
    }
    if (has(kwargs, "total_passes", options.total_passes)) {
        if (options.total_passes <= 0) {
            throw std::invalid_argument("CUGR total_passes must be positive");
        }
    } else {
        has(kwargs, "rrr_iters", extra_rrr_iters);
        if (extra_rrr_iters < 0) {
            throw std::invalid_argument("CUGR rrr_iters must be non-negative");
        }
        options.total_passes = extra_rrr_iters + 1;
    }
    return options;
}

py::dict result_to_dict(const cugr_api::RunResult& result) {
    py::dict output;
    output["grid_x"] = result.grid_x;
    output["grid_y"] = result.grid_y;
    output["num_layers"] = result.num_layers;
    output["x_lines"] = result.x_lines;
    output["y_lines"] = result.y_lines;
    output["layer_names"] = result.layer_names;
    output["layer_directions"] = result.layer_directions;
    output["layer_edge_offsets"] = result.layer_edge_offsets;
    output["capacity"] = result.capacity;
    output["fixed_usage"] = result.fixed_usage;
    output["movable_usage"] = result.movable_usage;
    output["wire_usage"] = result.wire_usage;
    output["via_usage"] = result.via_usage;
    output["wirelength_dbu"] = result.wirelength_dbu;
    output["short_vio_area"] = result.short_vio_area;
    output["via_count"] = result.via_count;
    output["overflow_net_count"] = result.overflow_net_count;
    output["invalid_layer_warning_count"] = result.invalid_layer_warning_count;
    output["total_passes"] = result.total_passes;
    output["threads"] = result.threads;
    output["pass0_worker_count"] = result.pass0_worker_count;
    output["session_cache_hit"] = result.session_cache_hit;
    output["session_mode"] = result.session_mode;
    output["routing_layer_begin"] = result.routing_layer_begin;
    output["routing_layer_end"] = result.routing_layer_end;
    output["parse_sec"] = result.parse_sec;
    output["refresh_sec"] = result.refresh_sec;
    output["route_sec"] = result.route_sec;
    output["pack_sec"] = result.pack_sec;
    output["total_sec"] = result.total_sec;
    output["peak_rss_mb"] = result.peak_rss_mb;

    py::list nets;
    for (const auto& net : result.nets) {
        py::dict net_dict;
        net_dict["name"] = net.name;
        net_dict["routed"] = net.routed;
        net_dict["route_failed"] = net.route_failed;
        net_dict["route_failed_reason"] = net.route_failed_reason;
        py::list pins;
        for (const auto& pin : net.pin_access) {
            pins.append(py::make_tuple(pin.layer, pin.x, pin.y));
        }
        net_dict["pin_access"] = pins;
        py::list nodes;
        for (const auto& node : net.nodes) {
            nodes.append(py::make_tuple(
                node.layer, node.x, node.y, node.pin_index, node.parent_index));
        }
        net_dict["nodes"] = nodes;
        py::list entries;
        for (const auto& entry : net.entries) {
            entries.append(py::make_tuple(
                entry.type, entry.layer, entry.x1, entry.y1, entry.x2, entry.y2));
        }
        net_dict["entries"] = entries;
        nets.append(net_dict);
    }
    output["nets"] = nets;
    return output;
}

}  // namespace

PYBIND11_MODULE(cugr, module) {
    module.doc() = "CPU CUGR runtime for the Xplace routability backend";
    module.def("available", []() { return true; });
    module.def("backend_name", []() { return "cugr"; });
    module.def("source_commit", []() { return cugr_api::source_commit(); });
    module.def("source_dirty", []() { return cugr_api::source_dirty(); });
    module.def("build_metadata", []() {
        py::dict metadata;
        metadata["backend"] = "cugr";
        metadata["source_commit"] = cugr_api::source_commit();
        metadata["source_dirty"] = cugr_api::source_dirty();
        metadata["cuda_enabled"] = false;
        return metadata;
    });
    module.def("run", [](const py::dict& kwargs) {
        return result_to_dict(cugr_api::run(parse_options(kwargs)));
    });
    module.def("reset", &cugr_api::reset);
}
