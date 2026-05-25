#pragma once
#include <string>

namespace gr {

enum class GGRBackend { CUDA, CPU_PR, AUTO };

GGRBackend parseGGRBackend(const std::string& value);
std::string ggrBackendName(GGRBackend backend);

class GRSetting {
public:
    // 1. SystemSetting
    int deviceId = 0;

    // 2. Gridgraph setting
    int routeXSize = 0;
    int routeYSize = 0;
    int csrnScale = 0;

    // 3. The number of Rip-up and Reroute iterations (if 0, only PR is invoked)
    int rrrIters = 0;

    GGRBackend backend = GGRBackend::AUTO;
    GGRBackend resolvedBackend = GGRBackend::AUTO;

    std::string routeGuideFile = "";

    void reset();
};

extern GRSetting grSetting;
}  //   namespace gr
