#include "CPURouter.h"

#include "gpugr/gr/CpuRouteWorkerPool.h"
#include "gpugr/gr/PatternRoute.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace gr {

namespace {

constexpr int INF = 10000000;
constexpr int MAX_COST = 10000000;
constexpr int64_t PR_INF = (1LL << 60);
constexpr int64_t PR_FINITE_LIMIT = PR_INF / 2;
// Match the CUDA route-buffer capacity when constructing route-offset identities.
constexpr int MAX_ROUTE_LEN_PER_PIN = 160;

inline bool isFinitePatternCost(int64_t value) {
    return value >= 0 && value < PR_FINITE_LIMIT;
}

inline int boundedExp2(float x) {
    int shift = static_cast<int>(x);
    if (shift < 0) {
        return 0;
    }
    return 1 << std::min(30, shift);
}

using SteadyClock = std::chrono::steady_clock;

double elapsedSeconds(const SteadyClock::time_point& start) {
    return std::chrono::duration<double>(SteadyClock::now() - start).count();
}

std::size_t checkedBufferSize(std::size_t count, std::size_t scale, std::size_t extra = 0) {
    if (count > (std::numeric_limits<std::size_t>::max() - extra) / scale) {
        throw std::overflow_error("CPU pattern-routing preparation buffer size overflow");
    }
    return count * scale + extra;
}

void hashWord(uint64_t value, uint64_t& hash) {
    constexpr uint64_t FNV_PRIME = 1099511628211ULL;
    for (int byte = 0; byte < 8; ++byte) {
        hash ^= (value >> (byte * 8)) & 0xffULL;
        hash *= FNV_PRIME;
    }
}

uint64_t hashIntVector(const std::vector<int>& values) {
    uint64_t hash = 1469598103934665603ULL;
    hashWord(values.size(), hash);
    for (int value : values) {
        hashWord(static_cast<uint64_t>(static_cast<int64_t>(value)), hash);
    }
    return hash;
}

uint64_t hashRawRoutes(const std::vector<std::vector<int>>& routes) {
    uint64_t hash = 1469598103934665603ULL;
    hashWord(routes.size(), hash);
    for (const auto& route : routes) {
        hashWord(route.size(), hash);
        for (int value : route) {
            hashWord(static_cast<uint64_t>(static_cast<int64_t>(value)), hash);
        }
    }
    return hash;
}

}  // namespace

void CPURouter::initialize(const gr::GRDatabase& grdb, const std::string& backendName) {
    diagnosticBackend = backendName;
    LAYER = grdb.nLayers;
    N = grdb.nMaxGrid;
    X = grdb.xSize;
    Y = grdb.ySize;
    DIRECTION = grdb.m1direction;
    routingLayerBegin = grdb.routingLayerBegin;
    routingLayerEnd = grdb.routingLayerEnd;
    const int gridGraphSize = LAYER * N * N;
    routeState.resetResources(static_cast<std::size_t>(gridGraphSize));
    cost.assign(gridGraphSize, INF);
    viaCostMap.assign(gridGraphSize, 0);
    costSum.assign(gridGraphSize, 0);
    overflowWire.assign(gridGraphSize, 0);
    overflowVia.assign(gridGraphSize, 0);
    routeState.resetRoutes(0);
    numOvflNets = 0;

    double unitWireCostRaw = 0.5 * grdb.microns / grdb.m2pitch;
    double unitViaCostRaw = 4;
    unitViaCost = static_cast<float>(unitViaCostRaw / unitWireCostRaw * grdb.microns);
    unitViaMultiplier = 1.0f;
    logisticSlope = 1.0f;
    unitShortCostDiscounted.assign(LAYER, 0.0f);
    double unitShortVioCostRaw = 500;
    for (int layer = 0; layer < LAYER; ++layer) {
        double unitShortCost =
            unitShortVioCostRaw * grdb.layerWidth[layer] * grdb.microns / grdb.m2pitch / grdb.m2pitch / unitWireCostRaw;
        unitShortCostDiscounted[layer] = static_cast<float>(unitShortCost * 0.1);
    }
}

void CPURouter::setMap(const std::vector<float>& cap,
                       const std::vector<float>& wir,
                       const std::vector<float>& fixedL,
                       const std::vector<float>& fix,
                       const std::vector<float>& fixRaw,
                       const std::vector<float>& movRaw) {
    const int gridGraphSize = LAYER * N * N;
    if (static_cast<int>(cap.size()) < gridGraphSize ||
        static_cast<int>(wir.size()) < gridGraphSize ||
        static_cast<int>(fixedL.size()) < gridGraphSize ||
        static_cast<int>(fix.size()) < gridGraphSize ||
        static_cast<int>(fixRaw.size()) < gridGraphSize ||
        static_cast<int>(movRaw.size()) < gridGraphSize) {
        throw std::runtime_error("backend=" + diagnosticBackend +
                                 " operation=setMap missing required GR database map data");
    }
    capacity = cap;
    wireDist = wir;
    fixedLength = fixedL;
    fixed = fix;
    fixUsage = fixRaw;
    movUsage = movRaw;
}

int CPURouter::encodedIndex(int layer, int x, int y) const {
    if (layer < 0 || layer >= LAYER || x < 0 || x >= X || y < 0 || y >= Y) {
        throw std::runtime_error("backend=" + diagnosticBackend +
                                 " operation=route generated segment outside legal grid");
    }
    if (!((layer & 1) ^ DIRECTION)) {
        std::swap(x, y);
    }
    return layer * N * N + x * N + y;
}

std::tuple<int, int, int> CPURouter::decodeIndex(int encoded) const {
    int layer = encoded / N / N;
    int x = encoded % (N * N) / N;
    int y = encoded % N;
    if (!((layer & 1) ^ DIRECTION)) {
        std::swap(x, y);
    }
    return {layer, x, y};
}

float CPURouter::inCellViaUsage(int encoded) const {
    int layer = encoded / N / N - 1;
    int y = encoded / N % N;
    int x = encoded % N;
    float usage = 0.0f;
    if (layer + 2 < LAYER) {
        usage += static_cast<float>(routeState.vias()[encoded]);
    }
    if (layer >= 0) {
        usage += static_cast<float>(routeState.vias()[layer * N * N + x * N + y]);
    }
    return usage;
}

float CPURouter::twoCellsViaUsage(int encoded) const {
    return std::sqrt(0.5f * (inCellViaUsage(encoded) + inCellViaUsage(encoded + 1))) * 1.5f;
}

float CPURouter::inCellUsedArea(int encoded) const {
    float used = 0.0f;
    if (encoded % N > 0) {
        used += fixed[encoded - 1] + static_cast<float>(routeState.wires()[encoded - 1]);
    }
    if (encoded % N + 1 < N) {
        used += fixed[encoded] + static_cast<float>(routeState.wires()[encoded]);
    }
    return used / 2.0f;
}

float CPURouter::inCellViaCost(int encoded) const {
    return 1.0f / (1.0f + static_cast<float>(boundedExp2(logisticSlope * (capacity[encoded] - inCellUsedArea(encoded)))));
}

float CPURouter::wireCost(int encoded) const {
    return static_cast<float>(calcWireCost(encoded));
}

float CPURouter::viaCost(int encoded) const {
    return static_cast<float>(calcViaCost(encoded));
}

int CPURouter::calcWireCost(int encoded) const {
    if (!(encoded < LAYER * N * N && encoded % N + 1 < N)) {
        return INF;
    }
    const int layer = encoded / N / N;
    if (!isRoutingLayerEnabled(layer)) {
        return INF;
    }
    if (capacity[encoded] < 0.01f) {
        return INF;
    }
    int expectedLen = static_cast<int>(
        (fixed[encoded] * fixedLength[encoded] + static_cast<float>(routeState.wires()[encoded]) * wireDist[encoded]) /
        capacity[encoded]);
    float remain = capacity[encoded] - (fixed[encoded] + static_cast<float>(routeState.wires()[encoded]) + 1.0f +
                                        twoCellsViaUsage(encoded));
    float result =
        wireDist[encoded] +
        static_cast<float>(expectedLen) / (1.0f + std::exp(logisticSlope * remain)) *
            unitShortCostDiscounted[encoded / N / N];
    int rounded = static_cast<int>(result);
    return rounded > MAX_COST ? MAX_COST : rounded;
}

int CPURouter::calcViaCost(int encoded) const {
    if (encoded >= (LAYER - 1) * N * N) {
        return 0;
    }
    const int lowerLayer = encoded / N / N;
    const int upperLayer = lowerLayer + 1;
    if (!isRoutingLayerEnabled(lowerLayer) || !isRoutingLayerEnabled(upperLayer)) {
        return INF;
    }
    int layer = upperLayer;
    int y = encoded % (N * N) / N;
    int x = encoded % N;
    int upper = layer * N * N + x * N + y;
    float result = unitViaCost * (unitViaMultiplier + inCellViaCost(encoded) + inCellViaCost(upper));
    int rounded = static_cast<int>(result);
    return rounded > MAX_COST ? MAX_COST : rounded;
}

void CPURouter::calculateCostMaps() {
    const int gridGraphSize = LAYER * N * N;
    for (int idx = 0; idx < gridGraphSize; ++idx) {
        cost[idx] = calcWireCost(idx);
        viaCostMap[idx] = calcViaCost(idx);
    }
}

void CPURouter::calculateCostSum() {
    std::fill(costSum.begin(), costSum.end(), 0);
    for (int layer = 1; layer < LAYER; ++layer) {
        for (int row = 0; row < N; ++row) {
            int64_t running = 0;
            for (int col = N - 1; col >= 0; --col) {
                int idx = layer * N * N + row * N + col;
                running += cost[idx];
                costSum[idx] = running;
            }
        }
    }
}

void CPURouter::calculateCostMapsParallel(CpuRouteWorkerPool& workerPool, int workers) {
    const int gridGraphSize = LAYER * N * N;
    const std::size_t jobCount = static_cast<std::size_t>(std::max(1, workers)) * 8;
    const std::size_t chunkSize =
        (static_cast<std::size_t>(gridGraphSize) + jobCount - 1) / jobCount;
    workerPool.run(jobCount, [&](int, std::size_t jobId) {
        const std::size_t begin = jobId * chunkSize;
        const std::size_t end = std::min(
            static_cast<std::size_t>(gridGraphSize), begin + chunkSize);
        for (std::size_t idx = begin; idx < end; ++idx) {
            cost[idx] = calcWireCost(static_cast<int>(idx));
            viaCostMap[idx] = calcViaCost(static_cast<int>(idx));
        }
    });
}

void CPURouter::calculateCostSumParallel(CpuRouteWorkerPool& workerPool) {
    const std::size_t rowCount = static_cast<std::size_t>(std::max(0, LAYER - 1)) * N;
    workerPool.run(rowCount, [&](int, std::size_t rowId) {
        const int layer = static_cast<int>(rowId / static_cast<std::size_t>(N)) + 1;
        const int row = static_cast<int>(rowId % static_cast<std::size_t>(N));
        int64_t running = 0;
        for (int col = N - 1; col >= 0; --col) {
            const int idx = layer * N * N + row * N + col;
            running += cost[idx];
            costSum[idx] = running;
        }
    });
}

void CPURouter::viaSweep(std::vector<int64_t>& map,
                         std::vector<int>& prev,
                         std::vector<int>& midScratch,
                         int x,
                         int y) const {
    const int stride = LAYER * N * N;
    const int firstLayer = std::max(1, routingLayerBegin);
    const int lastLayer = std::min(LAYER - 1, routingLayerEnd);
    if (firstLayer > lastLayer) {
        return;
    }
    for (int layer = firstLayer; layer <= lastLayer; ++layer) {
        int cur = layer * N * N + (((layer & 1) ^ DIRECTION) ? x * N + y : y * N + x);
        midScratch[cur] = cur;
    }
    for (int layer = firstLayer + 1; layer <= lastLayer; ++layer) {
        int cur = layer * N * N + (((layer & 1) ^ DIRECTION) ? x * N + y : y * N + x);
        int last = (layer - 1) * N * N + (((layer & 1) ^ DIRECTION) ? y * N + x : x * N + y);
        const int64_t previous = map[4 * stride + last];
        const int64_t viaCost = viaCostMap[last];
        const int64_t candidate = previous + viaCost;
        if (isFinitePatternCost(previous) && viaCost < INF &&
            candidate < map[4 * stride + cur] && isFinitePatternCost(candidate)) {
            map[4 * stride + cur] = candidate;
            prev[4 * stride + cur] = prev[4 * stride + last];
            midScratch[cur] = last;
        }
    }
    for (int layer = lastLayer - 1; layer >= firstLayer; --layer) {
        int cur = layer * N * N + (((layer & 1) ^ DIRECTION) ? x * N + y : y * N + x);
        int last = (layer + 1) * N * N + (((layer & 1) ^ DIRECTION) ? y * N + x : x * N + y);
        const int64_t previous = map[4 * stride + last];
        const int64_t viaCost = viaCostMap[cur];
        const int64_t candidate = previous + viaCost;
        if (isFinitePatternCost(previous) && viaCost < INF &&
            candidate < map[4 * stride + cur] && isFinitePatternCost(candidate)) {
            map[4 * stride + cur] = candidate;
            prev[4 * stride + cur] = prev[4 * stride + last];
            midScratch[cur] = last;
        }
    }
}

void CPURouter::appendWire(CpuRouteCandidate& candidate, int from, int to) const {
    if (from == to) {
        return;
    }
    const int fromLayer = from / N / N;
    const int toLayer = to / N / N;
    if (fromLayer != toLayer || !isRoutingLayerEnabled(fromLayer)) {
        throw std::runtime_error("backend=" + diagnosticBackend +
                                 " operation=route generated wire outside GPUGR routing layer range");
    }
    int lo = std::min(from, to);
    int hi = std::max(from, to);
    candidate.rawRoute.push_back(lo);
    candidate.rawRoute.push_back(hi - lo);
    candidate.wireDeltas.emplace_back(lo, hi - lo);
}

void CPURouter::appendVia(CpuRouteCandidate& candidate, int encoded) const {
    const int lowerLayer = encoded / N / N;
    if (!isRoutingLayerEnabled(lowerLayer) || !isRoutingLayerEnabled(lowerLayer + 1)) {
        throw std::runtime_error("backend=" + diagnosticBackend +
                                 " operation=route generated via outside GPUGR routing layer range");
    }
    candidate.rawRoute.push_back(encoded);
    candidate.rawRoute.push_back(-1);
    candidate.viaDeltas.push_back(encoded);
}

void CPURouter::routePreparedNet(const std::vector<int>& pointsBuffer,
                                 CpuRouteScratch& scratch,
                                 CpuRouteCandidate& candidate,
                                 bool resetOnlyRouteRegion) const {
    candidate.status = CpuRouteStatus::NoRoute;
    candidate.rawRoute.clear();
    candidate.wireDeltas.clear();
    candidate.viaDeltas.clear();
    candidate.rawRoute.push_back(1);
    if (pointsBuffer.size() < 2) {
        candidate.status = CpuRouteStatus::Routed;
        return;
    }

    const int firstRoutingLayer = std::max(1, routingLayerBegin);
    const int lastRoutingLayer = std::min(LAYER - 1, routingLayerEnd);
    if (firstRoutingLayer > lastRoutingLayer) {
        candidate.rawRoute[0] = -1;
        return;
    }

    const int stride = LAYER * N * N;
    int nodeCnt = pointsBuffer[1];
    if (nodeCnt <= 0 || static_cast<std::size_t>(nodeCnt) >
                            (pointsBuffer.size() - 2) / static_cast<std::size_t>(6)) {
        throw std::runtime_error("backend=" + diagnosticBackend +
                                 " operation=route invalid prepared point count");
    }
    const int* points = pointsBuffer.data() + 2;
    if (resetOnlyRouteRegion) {
        int xl = X;
        int xh = -1;
        int yl = Y;
        int yh = -1;
        for (int node = 0; node < nodeCnt; ++node) {
            const int location = points[node * 6];
            const int x = location / N;
            const int y = location % N;
            if (x < 0 || x >= X || y < 0 || y >= Y) {
                throw std::runtime_error("backend=" + diagnosticBackend +
                                         " operation=route prepared point outside route grid");
            }
            xl = std::min(xl, x);
            xh = std::max(xh, x);
            yl = std::min(yl, y);
            yh = std::max(yh, y);
        }
        scratch.resetRegion(LAYER, N, DIRECTION, xl, xh, yl, yh, PR_INF);
    } else {
        scratch.reset(static_cast<std::size_t>(stride), PR_INF);
    }
    std::vector<int64_t>& map = scratch.map;
    std::vector<int>& prev = scratch.prev;
    std::vector<int>& prev2 = prev;
    std::vector<int>& midScratch = scratch.mid;

    for (int t = 6 * (nodeCnt - 1); t >= 0; t -= 6) {
        int tox = points[t] / N;
        int toy = points[t] % N;
        int minlayer = points[t + 1];
        int maxlayer = points[t + 2];

        for (int j = 1; j <= 3; ++j) {
            if (points[t + 2 + j] == -1) {
                continue;
            }
            int childPos = points[points[t + 2 + j]];
            int fromx = childPos / N;
            int fromy = childPos % N;
            if (fromx != tox && fromy != toy) {
                for (int a = std::min(fromx, tox); a <= std::max(fromx, tox); ++a) {
                    for (int layer = firstRoutingLayer; layer <= lastRoutingLayer; ++layer) {
                        if ((layer & 1) ^ DIRECTION) {
                            map[4 * stride + layer * N * N + a * N + toy] = PR_INF;
                            map[4 * stride + layer * N * N + a * N + fromy] = PR_INF;
                        } else {
                            map[4 * stride + layer * N * N + toy * N + a] = PR_INF;
                            map[4 * stride + layer * N * N + fromy * N + a] = PR_INF;
                        }
                    }
                }
                for (int b = std::min(fromy, toy); b <= std::max(fromy, toy); ++b) {
                    for (int layer = firstRoutingLayer; layer <= lastRoutingLayer; ++layer) {
                        if ((layer & 1) ^ DIRECTION) {
                            map[4 * stride + layer * N * N + tox * N + b] = PR_INF;
                            map[4 * stride + layer * N * N + fromx * N + b] = PR_INF;
                        } else {
                            map[4 * stride + layer * N * N + b * N + tox] = PR_INF;
                            map[4 * stride + layer * N * N + b * N + fromx] = PR_INF;
                        }
                    }
                }
                for (int layer = firstRoutingLayer; layer <= lastRoutingLayer; ++layer) {
                    if ((layer & 1) ^ DIRECTION) {
                        int last = layer * N * N + fromx * N + fromy;
                        for (int b = std::min(fromy, toy); b <= std::max(fromy, toy); ++b) {
                            if (b == fromy) {
                                continue;
                            }
                            int cur = layer * N * N + fromx * N + b;
                            int64_t wire = last < cur ? costSum[last] - costSum[cur] : costSum[cur] - costSum[last];
                            map[4 * stride + cur] = map[last] + wire;
                            prev[4 * stride + cur] = last;
                        }
                    } else {
                        int last = layer * N * N + fromy * N + fromx;
                        for (int b = std::min(fromx, tox); b <= std::max(fromx, tox); ++b) {
                            if (b == fromx) {
                                continue;
                            }
                            int cur = layer * N * N + fromy * N + b;
                            int64_t wire = last < cur ? costSum[last] - costSum[cur] : costSum[cur] - costSum[last];
                            map[4 * stride + cur] = map[last] + wire;
                            prev[4 * stride + cur] = last;
                        }
                    }
                }
                for (int b = std::min(fromy, toy); b <= std::max(fromy, toy); ++b) {
                    if (b != fromy) {
                        viaSweep(map, prev, midScratch, fromx, b);
                    }
                }
                for (int b = std::min(fromx, tox); b <= std::max(fromx, tox); ++b) {
                    if (b != fromx) {
                        viaSweep(map, prev, midScratch, b, fromy);
                    }
                }
                for (int layer = firstRoutingLayer; layer <= lastRoutingLayer; ++layer) {
                    if ((layer & 1) ^ DIRECTION) {
                        for (int b = std::min(fromx, tox); b <= std::max(fromx, tox); ++b) {
                            int last = layer * N * N + b * N + fromy;
                            int cur = layer * N * N + b * N + toy;
                            int64_t wire =
                                last < cur ? costSum[last] - costSum[cur] : costSum[cur] - costSum[last];
                            if (map[4 * stride + last] + wire < map[4 * stride + cur]) {
                                map[4 * stride + cur] = map[4 * stride + last] + wire;
                                prev[4 * stride + cur] = prev[4 * stride + last];
                            }
                        }
                    } else {
                        for (int b = std::min(fromy, toy); b <= std::max(fromy, toy); ++b) {
                            int last = layer * N * N + b * N + fromx;
                            int cur = layer * N * N + b * N + tox;
                            int64_t wire =
                                last < cur ? costSum[last] - costSum[cur] : costSum[cur] - costSum[last];
                            if (map[4 * stride + last] + wire < map[4 * stride + cur]) {
                                map[4 * stride + cur] = map[4 * stride + last] + wire;
                                prev[4 * stride + cur] = prev[4 * stride + last];
                            }
                        }
                    }
                }
                for (int b = std::min(fromy, toy); b <= std::max(fromy, toy); ++b) {
                    if (b != toy) {
                        viaSweep(map, prev, midScratch, tox, b);
                    }
                }
                for (int b = std::min(fromx, tox); b <= std::max(fromx, tox); ++b) {
                    if (b != tox) {
                        viaSweep(map, prev, midScratch, b, toy);
                    }
                }
                for (int layer = firstRoutingLayer; layer <= lastRoutingLayer; ++layer) {
                    if ((layer & 1) ^ DIRECTION) {
                        int cur = layer * N * N + tox * N + toy;
                        for (int b = std::min(fromy, toy); b <= std::max(fromy, toy); ++b) {
                            if (b == toy) {
                                continue;
                            }
                            int last = layer * N * N + tox * N + b;
                            int64_t wire =
                                last < cur ? costSum[last] - costSum[cur] : costSum[cur] - costSum[last];
                            if (map[4 * stride + last] + wire < map[cur + j * stride]) {
                                map[cur + j * stride] = map[4 * stride + last] + wire;
                                prev[cur + j * stride] = prev[4 * stride + last];
                                prev2[10 * stride + cur + j * stride] = midScratch[last];
                            }
                        }
                    } else {
                        int cur = layer * N * N + toy * N + tox;
                        for (int b = std::min(fromx, tox); b <= std::max(fromx, tox); ++b) {
                            if (b == tox) {
                                continue;
                            }
                            int last = layer * N * N + toy * N + b;
                            int64_t wire =
                                last < cur ? costSum[last] - costSum[cur] : costSum[cur] - costSum[last];
                            if (map[4 * stride + last] + wire < map[cur + j * stride]) {
                                map[cur + j * stride] = map[4 * stride + last] + wire;
                                prev[cur + j * stride] = prev[4 * stride + last];
                                prev2[10 * stride + cur + j * stride] = midScratch[last];
                            }
                        }
                    }
                }
            } else if (fromx == tox) {
                for (int layer = firstRoutingLayer; layer <= lastRoutingLayer; ++layer) {
                    if ((layer & 1) ^ DIRECTION) {
                        int last = layer * N * N + fromx * N + fromy;
                        int cur = layer * N * N + tox * N + toy;
                        int64_t wire = last < cur ? costSum[last] - costSum[cur] : costSum[cur] - costSum[last];
                        if (map[cur + j * stride] > map[last] + wire) {
                            map[cur + j * stride] = map[last] + wire;
                            prev[cur + j * stride] = last;
                        }
                    } else {
                        map[layer * N * N + toy * N + tox + j * stride] = PR_INF;
                    }
                }
            } else if (fromy == toy) {
                for (int layer = firstRoutingLayer; layer <= lastRoutingLayer; ++layer) {
                    if (!((layer & 1) ^ DIRECTION)) {
                        int last = layer * N * N + fromy * N + fromx;
                        int cur = layer * N * N + toy * N + tox;
                        int64_t wire = last < cur ? costSum[last] - costSum[cur] : costSum[cur] - costSum[last];
                        if (map[cur + j * stride] > map[last] + wire) {
                            map[cur + j * stride] = map[last] + wire;
                            prev[cur + j * stride] = last;
                        }
                    } else {
                        map[layer * N * N + tox * N + toy + j * stride] = PR_INF;
                    }
                }
            }
        }

        for (int lower = firstRoutingLayer; lower <= lastRoutingLayer; ++lower) {
            int64_t viaSum = 0;
            int64_t minCost = PR_INF;
            int bestpos[3] = {0, 0, 0};
            int64_t costs[3] = {
                points[t + 3] == -1 ? 0 : PR_INF,
                points[t + 4] == -1 ? 0 : PR_INF,
                points[t + 5] == -1 ? 0 : PR_INF,
            };
            for (int upper = lower; upper <= lastRoutingLayer; ++upper) {
                int idx = upper * N * N + (((upper & 1) ^ DIRECTION) ? tox * N + toy : toy * N + tox);
                for (int j = 0; j < 3; ++j) {
                    if (points[t + 3 + j] != -1) {
                        const int64_t childCost = map[idx + (j + 1) * stride];
                        if (isFinitePatternCost(childCost) && childCost < costs[j]) {
                            costs[j] = childCost;
                            bestpos[j] = upper;
                        }
                    }
                }
                const int64_t candidateCost = costs[0] + costs[1] + costs[2];
                if (isFinitePatternCost(candidateCost) && candidateCost < minCost) {
                    minCost = candidateCost;
                }
                if (isFinitePatternCost(minCost) &&
                    (minlayer == -1 || (lower <= minlayer && maxlayer <= upper))) {
                    for (int layer = lower; layer <= upper; ++layer) {
                        int index = layer * N * N + (((layer & 1) ^ DIRECTION) ? tox * N + toy : toy * N + tox);
                        const int64_t candidate = minCost + viaSum;
                        if (isFinitePatternCost(candidate) && map[index] > candidate) {
                            map[index] = candidate;
                            prev[index] =
                                (((lower * LAYER + upper) * LAYER + bestpos[2]) * LAYER + bestpos[1]) * LAYER +
                                bestpos[0];
                        }
                    }
                }
                if (viaCostMap[idx] >= INF) {
                    viaSum = PR_INF;
                } else if (isFinitePatternCost(viaSum + viaCostMap[idx])) {
                    viaSum += viaCostMap[idx];
                } else {
                    viaSum = PR_INF;
                }
            }
        }
    }

    int64_t minTotalDist = PR_INF;
    int rootX = points[0] / N;
    int rootY = points[0] % N;
    for (int layer = firstRoutingLayer; layer <= lastRoutingLayer; ++layer) {
        int idx = layer * N * N + (((layer & 1) ^ DIRECTION) ? rootX * N + rootY : rootY * N + rootX);
        if (map[idx] < minTotalDist) {
            minTotalDist = map[idx];
            map[5 * stride + points[0]] = layer;
        }
    }
    if (!isFinitePatternCost(minTotalDist)) {
        candidate.rawRoute[0] = -1;
        return;
    }

    for (int t = 0; t < 6 * nodeCnt; t += 6) {
        int x = points[t] / N;
        int y = points[t] % N;
        int layer = static_cast<int>(map[5 * stride + points[t]]);
        int idx = layer * N * N + (((layer & 1) ^ DIRECTION) ? x * N + y : y * N + x);
        int best = prev[idx];
        int minlayer = best / LAYER / LAYER / LAYER / LAYER;
        int maxlayer = best / LAYER / LAYER / LAYER % LAYER;
        for (int viaLayer = minlayer; viaLayer < maxlayer; ++viaLayer) {
            int id = viaLayer * N * N + (((viaLayer & 1) ^ DIRECTION) ? x * N + y : y * N + x);
            appendVia(candidate, id);
        }
        int bestpos = best;
        for (int j = 0; j < 3; ++j) {
            if (points[t + 3 + j] == -1) {
                break;
            }
            int child = points[points[t + 3 + j]];
            int entrylayer = bestpos % LAYER;
            int entry = entrylayer * N * N + (((entrylayer & 1) ^ DIRECTION) ? x * N + y : y * N + x);
            bestpos /= LAYER;
            int last = prev[(j + 1) * stride + entry];
            int lastlayer = last / N / N;
            int lastx = last / N % N;
            int lasty = last % N;
            if (!((lastlayer & 1) ^ DIRECTION)) {
                std::swap(lastx, lasty);
            }
            if (lastx != x && lasty != y) {
                int mid = prev2[10 * stride + (j + 1) * stride + entry];
                int midlayer = mid / N / N;
                int midx = mid / N % N;
                int midy = mid % N;
                if (!((midlayer & 1) ^ DIRECTION)) {
                    std::swap(midx, midy);
                }
                int p1 = ((entrylayer & 1) ^ DIRECTION) ? entry - y + midy : entry - x + midx;
                int p3 = ((midlayer & 1) ^ DIRECTION) ? mid - midy + lasty : mid - midx + lastx;
                int p4 = ((lastlayer & 1) ^ DIRECTION) ? last - lasty + midy : last - lastx + midx;
                appendWire(candidate, entry, p1);
                if (p1 != mid) {
                    for (int viaLayer = std::min(entrylayer, midlayer); viaLayer < std::max(entrylayer, midlayer);
                         ++viaLayer) {
                        int id = viaLayer * N * N + (((viaLayer & 1) ^ DIRECTION) ? midx * N + midy : midy * N + midx);
                        appendVia(candidate, id);
                    }
                }
                appendWire(candidate, mid, p3);
                if (p3 != p4) {
                    int tempx = p3 / N % N;
                    int tempy = p3 % N;
                    if (!(((p3 / N / N) & 1) ^ DIRECTION)) {
                        std::swap(tempx, tempy);
                    }
                    for (int viaLayer = std::min(midlayer, lastlayer); viaLayer < std::max(midlayer, lastlayer);
                         ++viaLayer) {
                        int id = viaLayer * N * N + (((viaLayer & 1) ^ DIRECTION) ? tempx * N + tempy : tempy * N + tempx);
                        appendVia(candidate, id);
                    }
                }
                appendWire(candidate, p4, last);
            } else {
                appendWire(candidate, last, entry);
            }
            map[5 * stride + child] = lastlayer;
        }
    }
    if (candidate.rawRoute.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("backend=" + diagnosticBackend +
                                  " operation=route raw route exceeds INT_MAX entries");
    }
    candidate.rawRoute[0] = static_cast<int>(candidate.rawRoute.size());
    candidate.status = CpuRouteStatus::Routed;
}

CpuRouteBatchSchedule CPURouter::buildRoutingOrder(const std::vector<GrNet>& nets,
                                                   int conflictScanLimit,
                                                   int batchSizeLimit,
                                                   bool packConflictTail) const {
    std::vector<CpuRouteBox> boxes;
    boxes.reserve(nets.size());
    for (int netId = 0; netId < static_cast<int>(nets.size()); ++netId) {
        if (!nets[netId].noroute) {
            const auto& net = nets[netId];
            if (net.lowerx < 0 || net.lowery < 0 || net.upperx >= X || net.uppery >= Y ||
                net.lowerx > net.upperx || net.lowery > net.uppery) {
                throw std::runtime_error("backend=" + diagnosticBackend +
                                         " operation=batch net bbox outside route grid");
            }
            boxes.push_back({netId, net.lowerx, net.upperx, net.lowery, net.uppery});
        }
    }
    return buildCpuRouteBatchSchedule(
        std::move(boxes), X, Y, conflictScanLimit, batchSizeLimit, packConflictTail);
}

void CPURouter::setNetRouteFromRaw(GrNet& net, const std::vector<int>& rawRoute) const {
    if (rawRoute.empty() || rawRoute[0] < 0) {
        net.setWires({});
        net.setVias({});
        return;
    }
    std::vector<int> netWires;
    std::vector<int> netVias;
    int end = std::min(static_cast<int>(rawRoute.size()), rawRoute[0]);
    for (int i = 1; i + 1 < end; i += 2) {
        if (rawRoute[i + 1] > 0) {
            netWires.push_back(rawRoute[i]);
            netWires.push_back(rawRoute[i + 1]);
        } else if (rawRoute[i + 1] == -1) {
            netVias.push_back(rawRoute[i]);
        }
    }
    net.setWires(netWires);
    net.setVias(netVias);
}

void CPURouter::commitCandidate(GrNet& net, const CpuRouteCandidate& candidate) {
    try {
        routeState.commit(candidate);
    } catch (const std::exception& error) {
        throw std::runtime_error("backend=" + diagnosticBackend +
                                 " operation=commit: " + error.what());
    }
    setNetRouteFromRaw(net, candidate.rawRoute);
}

bool CPURouter::batchHasResourceConflict(
    const std::vector<GrNet>& nets,
    const std::vector<CpuRouteCandidate>& candidates,
    std::size_t begin,
    std::size_t end) const {
    if (begin > end || end > candidates.size()) {
        throw std::out_of_range("CPU pattern-routing candidate range is invalid");
    }
    std::unordered_set<uint64_t> resources;
    for (std::size_t candidateId = begin; candidateId < end; ++candidateId) {
        const auto& candidate = candidates[candidateId];
        if (candidate.netId < 0 || candidate.netId >= static_cast<int>(nets.size())) {
            return true;
        }
        const auto& net = nets[candidate.netId];
        auto inNetBox = [&](int encoded) {
            const auto [layer, x, y] = decodeIndex(encoded);
            return layer >= 0 && layer < LAYER && x >= net.lowerx && x <= net.upperx &&
                   y >= net.lowery && y <= net.uppery;
        };
        for (const auto& [start, span] : candidate.wireDeltas) {
            if (start < 0 || span <= 0 ||
                start > static_cast<int>(routeState.wires().size()) - span ||
                !inNetBox(start) || !inNetBox(start + span)) {
                return true;
            }
            for (int offset = 0; offset < span; ++offset) {
                const uint64_t key = static_cast<uint64_t>(start + offset) << 1;
                if (!resources.insert(key).second) {
                    return true;
                }
            }
        }
        for (int encoded : candidate.viaDeltas) {
            if (encoded < 0 || encoded >= static_cast<int>(routeState.vias().size()) ||
                !inNetBox(encoded)) {
                return true;
            }
            const uint64_t key = (static_cast<uint64_t>(encoded) << 1) | uint64_t{1};
            if (!resources.insert(key).second) {
                return true;
            }
        }
    }
    return false;
}

void CPURouter::markOverflowNets(const std::vector<GrNet>& nets) {
    std::fill(overflowWire.begin(), overflowWire.end(), 0);
    std::fill(overflowVia.begin(), overflowVia.end(), 0);
    for (int idx = 0; idx < LAYER * N * N; ++idx) {
        if (idx % N + 1 < N) {
            overflowWire[idx] = (static_cast<float>(routeState.wires()[idx]) + fixed[idx] +
                                 twoCellsViaUsage(idx) > capacity[idx]);
        }
    }
    for (int idx = 0; idx < (LAYER - 1) * N * N; ++idx) {
        int layer = idx / N / N + 1;
        int y = idx % (N * N) / N;
        int x = idx % N;
        int upper = layer * N * N + x * N + y;
        overflowVia[idx] = (inCellUsedArea(idx) > capacity[idx] || inCellUsedArea(upper) > capacity[upper]);
    }
    numOvflNets = 0;
    for (int netId = 0; netId < static_cast<int>(nets.size()); ++netId) {
        const auto& rawRoute = routeState.rawRoutes()[netId];
        if (rawRoute.empty() || rawRoute[0] < 0) {
            ++numOvflNets;
            continue;
        }
        bool overflow = false;
        int end = std::min(static_cast<int>(rawRoute.size()), rawRoute[0]);
        for (int i = 1; i + 1 < end && !overflow; i += 2) {
            if (rawRoute[i + 1] == -1) {
                overflow = overflowVia[rawRoute[i]] != 0;
            } else {
                for (int j = 0; j < rawRoute[i + 1]; ++j) {
                    if (overflowWire[rawRoute[i] + j]) {
                        overflow = true;
                        break;
                    }
                }
            }
        }
        if (overflow) {
            ++numOvflNets;
        }
    }
}

void CPURouter::route(std::vector<GrNet>& nets) {
    const auto totalStart = SteadyClock::now();
    runStats = CpuRouteRunStats{};
    runStats.terminalStatus = "running";
    routeState.resetRoutes(nets.size());
    std::vector<int> pinNum(nets.size(), 0);
    std::vector<int> routeOffsets(nets.size() + 1, 0);
    for (int netId = 0; netId < static_cast<int>(nets.size()); ++netId) {
        if (nets[netId].getPins().size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("backend=" + diagnosticBackend +
                                      " operation=prepare pin count exceeds INT_MAX");
        }
        pinNum[netId] = static_cast<int>(nets[netId].getPins().size());
        const int64_t nextOffset = static_cast<int64_t>(routeOffsets[netId]) +
                                   static_cast<int64_t>(pinNum[netId]) * MAX_ROUTE_LEN_PER_PIN;
        if (nextOffset > std::numeric_limits<int>::max()) {
            throw std::overflow_error("backend=" + diagnosticBackend +
                                      " operation=prepare route offset exceeds INT_MAX");
        }
        routeOffsets[netId + 1] = static_cast<int>(nextOffset);
    }

    const auto scheduleStart = SteadyClock::now();
    CpuRouteBatchSchedule schedule =
        buildRoutingOrder(nets, 0, runStats.batchSizeLimit, false);
    runStats.batchScheduleSeconds = elapsedSeconds(scheduleStart);
    runStats.batchCount = static_cast<int>(schedule.batchSizes.size());
    runStats.candidateBatchCount = runStats.batchCount;
    runStats.candidateBatchSizeLimit = runStats.batchSizeLimit;
    for (int batchSize : schedule.batchSizes) {
        runStats.maxBatchSize = std::max(runStats.maxBatchSize, batchSize);
    }
    CpuRouteScratch scratch;
    int start = 0;
    for (int batchSize : schedule.batchSizes) {
        const auto costStart = SteadyClock::now();
        calculateCostMaps();
        calculateCostSum();
        runStats.costMapSeconds += elapsedSeconds(costStart);
        int gbPinOffset = batchSize;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
            int netId = schedule.netOrder[start + batchIdx];
            const std::size_t pinCount = static_cast<std::size_t>(std::max(1, pinNum[netId]));
            std::vector<int> points(checkedBufferSize(pinCount, 48, 2), 0);
            std::vector<int> gbpoints(checkedBufferSize(pinCount, 4), -1);
            double prepareTime = 0.0;
            const auto prepareStart = SteadyClock::now();
            int len = prepare(prepareTime,
                              nets[netId],
                              points.data(),
                              routeOffsets[netId],
                              gbpoints.data(),
                              gbPinOffset,
                              points.size(),
                              gbpoints.size(),
                              X,
                              Y,
                              N,
                              LAYER,
                              DIRECTION,
                              &runStats.fluteWaitSeconds,
                              &runStats.fluteWallSeconds);
            points.resize(len);
            runStats.prepareSeconds += elapsedSeconds(prepareStart);

            CpuRouteCandidate candidate;
            candidate.batchOrdinal = batchIdx;
            candidate.netId = netId;
            const auto candidateStart = SteadyClock::now();
            routePreparedNet(points, scratch, candidate, false);
            runStats.candidateSeconds += elapsedSeconds(candidateStart);
            runStats.scratchBytesPerWorker = std::max(runStats.scratchBytesPerWorker, scratch.bytes());

            const auto commitStart = SteadyClock::now();
            commitCandidate(nets[netId], candidate);
            runStats.commitSeconds += elapsedSeconds(commitStart);
        }
        start += batchSize;
    }

    for (int netId = 0; netId < static_cast<int>(nets.size()); ++netId) {
        if (nets[netId].noroute || nets[netId].getPins().size() < 2) {
            routeState.setInactiveRoute(netId);
            nets[netId].setWires({});
            nets[netId].setVias({});
        }
    }
    const auto overflowStart = SteadyClock::now();
    markOverflowNets(nets);
    runStats.overflowMarkSeconds = elapsedSeconds(overflowStart);
    runStats.routeHash64 = hashRawRoutes(routeState.rawRoutes());
    runStats.wireMapHash64 = hashIntVector(routeState.wires());
    runStats.viaMapHash64 = hashIntVector(routeState.vias());
    runStats.totalSeconds = elapsedSeconds(totalStart);
    runStats.terminalStatus = "completed";
}

void CPURouter::routeParallel(std::vector<GrNet>& nets, int requestedWorkers) {
    const auto totalStart = SteadyClock::now();
    runStats = CpuRouteRunStats{};
    runStats.requestedWorkers = requestedWorkers;
    runStats.workerClampReason = "none";
    runStats.terminalStatus = "running";
    routeState.resetRoutes(nets.size());

    std::vector<int> pinNum(nets.size(), 0);
    for (int netId = 0; netId < static_cast<int>(nets.size()); ++netId) {
        if (nets[netId].getPins().size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("backend=" + diagnosticBackend +
                                      " operation=prepare pin count exceeds INT_MAX");
        }
        pinNum[netId] = static_cast<int>(nets[netId].getPins().size());
    }

    constexpr int PARALLEL_CONFLICT_SCAN_LIMIT = 100;
    constexpr int PARALLEL_BATCH_SIZE_LIMIT = 4096;
    constexpr int PARALLEL_CANDIDATE_BATCH_SIZE_LIMIT = 64;
    runStats.batchConflictScanLimit = PARALLEL_CONFLICT_SCAN_LIMIT;
    runStats.batchSizeLimit = PARALLEL_BATCH_SIZE_LIMIT;
    runStats.candidateBatchSizeLimit = PARALLEL_CANDIDATE_BATCH_SIZE_LIMIT;
    const auto scheduleStart = SteadyClock::now();
    CpuRouteBatchSchedule schedule = buildRoutingOrder(
        nets, PARALLEL_CONFLICT_SCAN_LIMIT, PARALLEL_BATCH_SIZE_LIMIT, true);
    runStats.batchScheduleSeconds = elapsedSeconds(scheduleStart);
    runStats.batchCount = static_cast<int>(schedule.batchSizes.size());
    runStats.packedConflictBatchCount = static_cast<int>(std::count(
        schedule.packedConflictBatches.begin(), schedule.packedConflictBatches.end(), uint8_t{1}));
    for (int batchSize : schedule.batchSizes) {
        runStats.maxBatchSize = std::max(runStats.maxBatchSize, batchSize);
    }

    const std::size_t stride = static_cast<std::size_t>(LAYER) * static_cast<std::size_t>(N) *
                               static_cast<std::size_t>(N);
    if (stride == 0 || stride > std::numeric_limits<std::size_t>::max() /
                                    (6 * sizeof(int64_t) + 15 * sizeof(int))) {
        throw std::overflow_error("backend=" + diagnosticBackend +
                                  " operation=setup scratch size overflow");
    }
    runStats.scratchBytesPerWorker = stride * (6 * sizeof(int64_t) + 15 * sizeof(int));
    const CpuWorkerLimit workerLimit =
        chooseCpuRouteWorkers(requestedWorkers, runStats.maxBatchSize, runStats.scratchBytesPerWorker);
    runStats.effectiveWorkers = workerLimit.effectiveWorkers;
    runStats.memoryBudgetBytes = workerLimit.memoryBudgetBytes;
    runStats.workerClampReason = workerLimit.clampReason;
    logger.info(
        "CPU PR MT scheduler: requested_workers=%d effective_workers=%d batches=%d max_batch=%d "
        "batch_limit=%d conflict_scan_limit=%d scratch_bytes_per_worker=%zu clamp=%s",
        runStats.requestedWorkers,
        runStats.effectiveWorkers,
        runStats.batchCount,
        runStats.maxBatchSize,
        runStats.batchSizeLimit,
        runStats.batchConflictScanLimit,
        runStats.scratchBytesPerWorker,
        runStats.workerClampReason.c_str());

    CpuRouteWorkerPool workerPool(runStats.effectiveWorkers);
    std::vector<CpuRouteScratch> scratches(static_cast<std::size_t>(runStats.effectiveWorkers));
    int start = 0;
    int batchId = 0;
    for (int batchSize : schedule.batchSizes) {
        const int conflictFreeNetCount =
            schedule.conflictFreeNetCounts[static_cast<std::size_t>(batchId)];
        const std::size_t parallelBegin =
            static_cast<std::size_t>(batchSize - conflictFreeNetCount);
        const auto costStart = SteadyClock::now();
        calculateCostMapsParallel(workerPool, runStats.effectiveWorkers);
        calculateCostSumParallel(workerPool);
        runStats.costMapSeconds += elapsedSeconds(costStart);

        std::vector<std::vector<int>> prepared(static_cast<std::size_t>(batchSize));
        std::vector<CpuRouteCandidate> candidates(static_cast<std::size_t>(batchSize));
        std::vector<double> fluteWaitTimes(static_cast<std::size_t>(batchSize), 0.0);
        std::vector<double> fluteWallTimes(static_cast<std::size_t>(batchSize), 0.0);
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
            candidates[static_cast<std::size_t>(batchIdx)].batchOrdinal = batchIdx;
            candidates[static_cast<std::size_t>(batchIdx)].netId =
                schedule.netOrder[start + batchIdx];
        }
        const auto prepareStart = SteadyClock::now();
        workerPool.run(static_cast<std::size_t>(batchSize), [&](int, std::size_t batchIdx) {
            const int netId = candidates[batchIdx].netId;
            auto& points = prepared[batchIdx];
            const std::size_t pinCount = static_cast<std::size_t>(std::max(1, pinNum[netId]));
            points.assign(checkedBufferSize(pinCount, 48, 2), 0);
            std::vector<int> gbpoints(checkedBufferSize(pinCount, 4), -1);
            double prepareTime = 0.0;
            int unusedGbPinOffset = 0;
            try {
                const int len = prepare(prepareTime,
                                        nets[netId],
                                        points.data(),
                                        0,
                                        gbpoints.data(),
                                        unusedGbPinOffset,
                                        points.size(),
                                        gbpoints.size(),
                                        X,
                                        Y,
                                        N,
                                        LAYER,
                                        DIRECTION,
                                        &fluteWaitTimes[batchIdx],
                                        &fluteWallTimes[batchIdx]);
                points.resize(static_cast<std::size_t>(len));
            } catch (const std::exception& error) {
                throw std::runtime_error(
                    "backend=" + diagnosticBackend + " operation=parallel_prepare net_id=" +
                    std::to_string(netId) + ": " + error.what());
            }
        });
        for (std::size_t batchIdx = 0; batchIdx < static_cast<std::size_t>(batchSize); ++batchIdx) {
            runStats.fluteWaitSeconds += fluteWaitTimes[batchIdx];
            runStats.fluteWallSeconds += fluteWallTimes[batchIdx];
        }
        runStats.prepareSeconds += elapsedSeconds(prepareStart);

        if (parallelBegin > 0) {
            ++runStats.serialFallbackBatchCount;
            const auto fallbackStart = SteadyClock::now();
            for (std::size_t batchIdx = 0; batchIdx < parallelBegin; ++batchIdx) {
                routePreparedNet(
                    prepared[batchIdx], scratches.front(), candidates[batchIdx], true);
            }
            runStats.serialFallbackSeconds += elapsedSeconds(fallbackStart);
            const auto commitStart = SteadyClock::now();
            for (std::size_t batchIdx = 0; batchIdx < parallelBegin; ++batchIdx) {
                const auto& candidate = candidates[batchIdx];
                commitCandidate(nets[candidate.netId], candidate);
            }
            runStats.commitSeconds += elapsedSeconds(commitStart);
        }

        for (std::size_t candidateBegin = parallelBegin;
             candidateBegin < static_cast<std::size_t>(batchSize);
             candidateBegin += PARALLEL_CANDIDATE_BATCH_SIZE_LIMIT) {
            const std::size_t candidateEnd = std::min(
                static_cast<std::size_t>(batchSize),
                candidateBegin + static_cast<std::size_t>(PARALLEL_CANDIDATE_BATCH_SIZE_LIMIT));
            const auto candidateStart = SteadyClock::now();
            workerPool.run(
                candidateEnd - candidateBegin, [&](int workerId, std::size_t localCandidateId) {
                    const std::size_t batchIdx = candidateBegin + localCandidateId;
                    try {
                        routePreparedNet(prepared[batchIdx],
                                         scratches[static_cast<std::size_t>(workerId)],
                                         candidates[batchIdx],
                                         true);
                    } catch (const std::exception& error) {
                        throw std::runtime_error(
                            "backend=" + diagnosticBackend +
                            " operation=parallel_route net_id=" +
                            std::to_string(candidates[batchIdx].netId) + ": " + error.what());
                    }
                });
            runStats.candidateSeconds += elapsedSeconds(candidateStart);
            ++runStats.candidateBatchCount;
            if (runStats.effectiveWorkers > 1 && candidateEnd - candidateBegin > 1) {
                ++runStats.parallelBatchCount;
            }

            if (batchHasResourceConflict(nets, candidates, candidateBegin, candidateEnd)) {
                ++runStats.conflictBatchCount;
                ++runStats.serialFallbackBatchCount;
                const auto fallbackStart = SteadyClock::now();
                for (std::size_t batchIdx = candidateBegin; batchIdx < candidateEnd; ++batchIdx) {
                    auto& candidate = candidates[batchIdx];
                    routePreparedNet(prepared[batchIdx],
                                     scratches.front(),
                                     candidate,
                                     true);
                }
                runStats.serialFallbackSeconds += elapsedSeconds(fallbackStart);
            }
            const auto commitStart = SteadyClock::now();
            for (std::size_t batchIdx = candidateBegin; batchIdx < candidateEnd; ++batchIdx) {
                const auto& candidate = candidates[batchIdx];
                commitCandidate(nets[candidate.netId], candidate);
            }
            runStats.commitSeconds += elapsedSeconds(commitStart);
        }
        start += batchSize;
        ++batchId;
        if (batchId % 100 == 0 || batchId == runStats.batchCount) {
            logger.info("CPU PR MT progress: batch=%d/%d routed_nets=%d",
                        batchId,
                        runStats.batchCount,
                        start);
        }
    }

    for (int netId = 0; netId < static_cast<int>(nets.size()); ++netId) {
        if (nets[netId].noroute || nets[netId].getPins().size() < 2) {
            routeState.setInactiveRoute(netId);
            nets[netId].setWires({});
            nets[netId].setVias({});
        }
    }
    const auto overflowStart = SteadyClock::now();
    markOverflowNets(nets);
    runStats.overflowMarkSeconds = elapsedSeconds(overflowStart);
    runStats.routeHash64 = hashRawRoutes(routeState.rawRoutes());
    runStats.wireMapHash64 = hashIntVector(routeState.wires());
    runStats.viaMapHash64 = hashIntVector(routeState.vias());
    runStats.totalSeconds = elapsedSeconds(totalStart);
    runStats.terminalStatus = "completed";
}

torch::Tensor CPURouter::buildGridTensor(const std::vector<float>& values) const {
    auto map = torch::zeros({LAYER, X, Y}, torch::dtype(torch::kFloat32));
    auto accessor = map.accessor<float, 3>();
    for (int layer = 0; layer < LAYER; ++layer) {
        for (int x = 0; x < X; ++x) {
            for (int y = 0; y < Y; ++y) {
                int encoded = encodedIndex(layer, x, y);
                if (encoded % N + 1 < N) {
                    accessor[layer][x][y] = values[encoded];
                }
            }
        }
    }
    return map;
}

torch::Tensor CPURouter::buildGridTensor(const std::vector<int>& values) const {
    auto map = torch::zeros({LAYER, X, Y}, torch::dtype(torch::kFloat32));
    auto accessor = map.accessor<float, 3>();
    for (int layer = 0; layer < LAYER; ++layer) {
        for (int x = 0; x < X; ++x) {
            for (int y = 0; y < Y; ++y) {
                int encoded = encodedIndex(layer, x, y);
                if (encoded % N + 1 < N) {
                    accessor[layer][x][y] = static_cast<float>(values[encoded]);
                }
            }
        }
    }
    return map;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CPURouter::getDemandMap() {
    auto dmdMap = torch::zeros({LAYER, X, Y}, torch::dtype(torch::kFloat32));
    auto wireMap = torch::zeros({LAYER, X, Y}, torch::dtype(torch::kFloat32));
    auto viaMap = torch::zeros({LAYER, X, Y}, torch::dtype(torch::kFloat32));
    auto dmd = dmdMap.accessor<float, 3>();
    auto wire = wireMap.accessor<float, 3>();
    auto via = viaMap.accessor<float, 3>();
    for (int layer = 0; layer < LAYER; ++layer) {
        for (int x = 0; x < X; ++x) {
            for (int y = 0; y < Y; ++y) {
                int encoded = encodedIndex(layer, x, y);
                if (encoded % N + 1 >= N) {
                    continue;
                }
                float wireDemand = static_cast<float>(routeState.wires()[encoded]) + fixed[encoded];
                float viaDemand = twoCellsViaUsage(encoded);
                wire[layer][x][y] = wireDemand;
                via[layer][x][y] = viaDemand;
                dmd[layer][x][y] = wireDemand + viaDemand;
            }
        }
    }
    return {dmdMap, wireMap, viaMap};
}

torch::Tensor CPURouter::getCapacityMap() { return buildGridTensor(capacity); }

torch::Tensor CPURouter::getRawWireDemandMap() { return buildGridTensor(routeState.wires()); }

torch::Tensor CPURouter::getFixUsageMap() { return buildGridTensor(fixUsage); }

torch::Tensor CPURouter::getMovUsageMap() { return buildGridTensor(movUsage); }

void CPURouter::unsupported(const std::string& operation) const {
    throw std::runtime_error("backend=" + diagnosticBackend + " operation=" + operation + " is unsupported");
}

}  // namespace gr
