#include "GPURouter.h"
#include <algorithm>
#include <stdexcept>

namespace gr {

void GPURouter::growRouteBuffer(const std::vector<int>& netIds) {
    const auto offsets = grownRouteOffsets(routesOffsetCPU, NET_NUM, netIds);
    int* replacement = nullptr;
    const size_t bytes = static_cast<size_t>(offsets.back()) * sizeof(int);
    const cudaError_t allocation = cudaMalloc(&replacement, bytes);
    if (allocation != cudaSuccess) {
        throw std::runtime_error(std::string("failed to grow GPU route buffer: ") +
                                 cudaGetErrorString(allocation));
    }
    cudaFree(routes);
    routes = replacement;
    std::copy(offsets.begin(), offsets.end(), routesOffsetCPU);
    cudaMemcpy(routesOffset, offsets.data(), offsets.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

    // A failed traceback may have updated demand before running out of space.
    // Discard the whole routing attempt and restart PR/RRR from empty demand.
    cudaMemset(routes, 0, bytes);
    cudaMemset(routeLengths, 0, NET_NUM * sizeof(int64_t));
    cudaMemset(isOverflowNet, 0, NET_NUM * sizeof(int));
    cudaMemset(wires, 0, static_cast<size_t>(LAYER) * N * N * sizeof(int));
    cudaMemset(vias, 0, static_cast<size_t>(LAYER) * N * N * sizeof(int));
    cudaMemset(gbpinRoutes, 0, static_cast<size_t>(6) * numGbPin * sizeof(int));
    const cudaError_t reset = cudaDeviceSynchronize();
    if (reset != cudaSuccess) {
        throw std::runtime_error(std::string("failed to reset GPU routing after growth: ") +
                                 cudaGetErrorString(reset));
    }
    logger.info("GPU route buffer grew for %zu nets; total records %lld; restarting PR/RRR",
                netIds.size(), static_cast<long long>(offsets.back()));
}

}  // namespace gr
