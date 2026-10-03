#pragma once
#include <climits>
#include <cstdint>
#include <cstdio>

namespace gr {

__device__ __forceinline__ bool appendMazeRoute(int *routes,
                                                int64_t *routeLength,
                                                int64_t routeCapacity,
                                                int *routeOverflow,
                                                int value) {
    const int64_t cursor = *routeLength;
    if (cursor < 1 || cursor >= routeCapacity || cursor >= static_cast<int64_t>(INT_MAX)) {
        if (cursor >= routeCapacity) {
            printf("Maze route record capacity exceeded: cursor=%lld capacity=%lld\n",
                   static_cast<long long>(cursor), static_cast<long long>(routeCapacity));
        }
        *routeLength = -1;
        routes[0] = -1;
        if (routeOverflow != nullptr) *routeOverflow = 1;
        return false;
    }
    routes[cursor] = value;
    *routeLength = cursor + 1;
    routes[0] = static_cast<int>(*routeLength);
    return true;
}

// Fine-grid traceback often visits one grid edge at a time. Keep a continuous
// straight wire in one record without changing its edges or demand updates.
__device__ __forceinline__ bool appendMazeWire(int *routes,
                                               int64_t *routeLength,
                                               int64_t routeCapacity,
                                               int *routeOverflow,
                                               int point,
                                               int span,
                                               int stride) {
    const int64_t cursor = *routeLength;
    if (cursor >= 3 && routes[cursor - 1] > 0 &&
        routes[cursor - 2] / stride == point / stride) {
        const int previousPoint = routes[cursor - 2];
        const int previousSpan = routes[cursor - 1];
        if (previousPoint + previousSpan == point || point + span == previousPoint) {
            routes[cursor - 2] = min(previousPoint, point);
            routes[cursor - 1] += span;
            return true;
        }
    }
    return appendMazeRoute(routes, routeLength, routeCapacity, routeOverflow, point) &&
           appendMazeRoute(routes, routeLength, routeCapacity, routeOverflow, span);
}

// All layers of a pin group become zero-distance sources. Record the vias
// that make this interval physically connected, just as pattern routing does.
// The caller runs on one thread and supplies the setup's ordered layer list.
__device__ __forceinline__ void appendMazePinVias(int *routes,
                                                 int64_t *routeLength,
                                                 int64_t routeCapacity,
                                                 int *routeOverflow,
                                                 const int *accessPoints,
                                                 int count,
                                                 int *vias) {
    for (int pin = 0; pin + 1 < count && *routeLength > 0; ++pin) {
        const int via = accessPoints[pin];
        bool recorded = false;
        for (int64_t i = 1; i + 1 < *routeLength; i += 2) {
            if (routes[i] == via && routes[i + 1] == -1) {
                recorded = true;
                break;
            }
        }
        if (!recorded &&
            appendMazeRoute(routes, routeLength, routeCapacity, routeOverflow, via) &&
            appendMazeRoute(routes, routeLength, routeCapacity, routeOverflow, -1)) {
            atomicAdd(vias + via, 1);
        }
    }
}

}  // namespace gr
