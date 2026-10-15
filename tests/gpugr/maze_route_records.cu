// Build independently with cmake -S tests/gpugr -B <build>.
#include <cuda_runtime.h>
#include "gr/MazeRouteRecords.cuh"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
        std::abort();
    }
}

__global__ void connectPins(int *routes, int64_t *length, int capacity,
                            int *overflow, const int *pins, int count,
                            int *vias, bool existingVia) {
    if (existingVia) {
        gr::appendMazeRoute(routes, length, capacity, overflow, pins[0]);
        gr::appendMazeRoute(routes, length, capacity, overflow, -1);
        atomicAdd(vias + pins[0], 1);
    }
    gr::appendMazePinVias(routes, length, capacity, overflow, pins, count, vias);
    gr::appendMazePinVias(routes, length, capacity, overflow, pins, count, vias);
}

__global__ void appendWires(int *routes, int64_t *length, int *overflow, bool longWire) {
    if (longWire) {
        for (int point = 4096 + 1036; point >= 4096 + 37; --point) {
            gr::appendMazeWire(routes, length, 16, overflow, point, 1, 2048);
        }
    } else {
        gr::appendMazeWire(routes, length, 16, overflow, 100, 3, 128);
        gr::appendMazeWire(routes, length, 16, overflow, 97, 3, 128);
        gr::appendMazeWire(routes, length, 16, overflow, 103, 2, 128);
        gr::appendMazeRoute(routes, length, 16, overflow, 97);
        gr::appendMazeRoute(routes, length, 16, overflow, -1);
        gr::appendMazeWire(routes, length, 16, overflow, 105, 1, 128);
    }
}

int main() {
    int *routes, *pins, *vias, *overflow;
    int64_t *length;
    checkCuda(cudaMallocManaged(&routes, 16 * sizeof(int)));
    checkCuda(cudaMallocManaged(&pins, 3 * sizeof(int)));
    checkCuda(cudaMallocManaged(&vias, 256 * sizeof(int)));
    checkCuda(cudaMallocManaged(&overflow, sizeof(int)));
    checkCuda(cudaMallocManaged(&length, sizeof(int64_t)));
    // Same (x=3, y=7) on three layers, with alternating index orientation.
    const std::vector<int> access{95, 187, 223};
    std::copy(access.begin(), access.end(), pins);
    for (int scenario = 0; scenario < 4; ++scenario) {
        std::fill(routes, routes + 16, 0);
        std::fill(vias, vias + 256, 0);
        routes[0] = 1;
        *length = 1;
        *overflow = 0;
        const int count = scenario == 0 ? 1 : 3;
        const int capacity = scenario == 3 ? 4 : 16;
        connectPins<<<1, 1>>>(routes, length, capacity, overflow, pins, count,
                              vias, scenario == 2);
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
        std::vector<int> expectedRoutes(16, 0), expectedVias(256, 0);
        if (scenario == 0) {
            expectedRoutes[0] = 1;
            assert(*length == 1 && *overflow == 0);
        } else if (scenario == 3) {
            expectedRoutes[0] = -1;
            expectedRoutes[1] = 95;
            expectedRoutes[2] = -1;
            expectedRoutes[3] = 187;
            expectedVias[95] = 1;
            assert(*length == -1 && *overflow == 1);
        } else {
            std::copy_n(std::vector<int>{5, 95, -1, 187, -1}.begin(), 5,
                        expectedRoutes.begin());
            expectedVias[95] = expectedVias[187] = 1;
            assert(*length == 5 && *overflow == 0);
        }
        assert(std::vector<int>(routes, routes + 16) == expectedRoutes);
        assert(std::vector<int>(vias, vias + 256) == expectedVias);
    }
    for (bool longWire : {false, true}) {
        std::fill(routes, routes + 16, 0);
        routes[0] = 1;
        *length = 1;
        *overflow = 0;
        appendWires<<<1, 1>>>(routes, length, overflow, longWire);
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
        const std::vector<int> records = longWire ? std::vector<int>{3, 4133, 1000}
                                                 : std::vector<int>{7, 97, 8, 97, -1, 105, 1};
        std::vector<int> expected(16, 0);
        std::copy(records.begin(), records.end(), expected.begin());
        assert(std::vector<int>(routes, routes + 16) == expected);
        assert(*length == records.size() && *overflow == 0);
    }
    checkCuda(cudaFree(routes));
    checkCuda(cudaFree(pins));
    checkCuda(cudaFree(vias));
    checkCuda(cudaFree(overflow));
    checkCuda(cudaFree(length));
    std::puts("6 CUDA route-record cases passed");
}
