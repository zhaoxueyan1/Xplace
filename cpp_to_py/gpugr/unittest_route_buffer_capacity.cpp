#include "gr/RouteBufferCapacity.h"
#include <cassert>

int main() {
    const std::vector<int64_t> initial{0, 256, 640, 641};
    const auto grown = gr::grownRouteOffsets(initial.data(), 3, {0, 0});
    assert((grown == std::vector<int64_t>{0, 1024, 1408, 1409}));
    const auto again = gr::grownRouteOffsets(grown.data(), 3, {0, 1});
    assert((again == std::vector<int64_t>{0, 4096, 5632, 5633}));
    const std::vector<int64_t> limit{0, std::numeric_limits<int>::max() / 4 + 1};
    bool failed = false;
    try {
        gr::grownRouteOffsets(limit.data(), 1, {0});
    } catch (const std::runtime_error&) {
        failed = true;
    }
    assert(failed);
}
