#include "gr/RouteBufferCapacity.h"
#include <cassert>

int main() {
    const std::vector<int64_t> initial{0, 256, 640, 641};
    const auto grown = gr::grownRouteOffsets(initial.data(), 3, {0, 0});
    assert((grown == std::vector<int64_t>{0, 512, 896, 897}));
    const auto again = gr::grownRouteOffsets(grown.data(), 3, {0, 1});
    assert((again == std::vector<int64_t>{0, 1024, 1792, 1793}));
    const std::vector<int64_t> limit{0, std::numeric_limits<int>::max() / 2 + 1};
    bool failed = false;
    try {
        gr::grownRouteOffsets(limit.data(), 1, {0});
    } catch (const std::runtime_error&) {
        failed = true;
    }
    assert(failed);
}
