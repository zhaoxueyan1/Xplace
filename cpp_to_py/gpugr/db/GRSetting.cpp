#include "GRSetting.h"
#include <algorithm>
#include <stdexcept>

namespace gr {

GGRBackend parseGGRBackend(const std::string& value) {
    std::string normalized = value;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (normalized == "cuda") {
        return GGRBackend::CUDA;
    }
    if (normalized == "cpu_pr") {
        return GGRBackend::CPU_PR;
    }
    if (normalized == "auto" || normalized.empty()) {
        return GGRBackend::AUTO;
    }
    throw std::runtime_error("Unsupported gpugr backend '" + value + "'. Supported backends: cuda, cpu_pr, auto");
}

std::string ggrBackendName(GGRBackend backend) {
    switch (backend) {
        case GGRBackend::CUDA:
            return "cuda";
        case GGRBackend::CPU_PR:
            return "cpu_pr";
        case GGRBackend::AUTO:
            return "auto";
    }
    return "unknown";
}

void GRSetting::reset() {
    deviceId = 0;

    routeXSize = 0;
    routeYSize = 0;
    csrnScale = 0;

    rrrIters = 0;
    backend = GGRBackend::AUTO;
    resolvedBackend = GGRBackend::AUTO;

    routeGuideFile = ""; 
}

GRSetting grSetting;

}  //   namespace gr
