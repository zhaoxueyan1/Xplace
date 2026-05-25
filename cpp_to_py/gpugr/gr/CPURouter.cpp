#include "CPURouter.h"

#include "gpugr/gr/PatternRoute.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gr {

namespace {

constexpr int INF = 10000000;
constexpr int PR_INT_INF = 1000000000;
constexpr int MAX_COST = 10000000;
constexpr int64_t PR_INF = (1LL << 60);
constexpr int MAX_ROUTE_LEN_PER_PIN = 130;

inline int boundedExp2(float x) {
    int shift = static_cast<int>(x);
    if (shift < 0) {
        return 0;
    }
    return 1 << std::min(30, shift);
}

}  // namespace

void CPURouter::initialize(const gr::GRDatabase& grdb) {
    LAYER = grdb.nLayers;
    N = grdb.nMaxGrid;
    X = grdb.xSize;
    Y = grdb.ySize;
    DIRECTION = grdb.m1direction;
    const int gridGraphSize = LAYER * N * N;
    wires.assign(gridGraphSize, 0);
    vias.assign(gridGraphSize, 0);
    cost.assign(gridGraphSize, INF);
    viaCostMap.assign(gridGraphSize, 0);
    costSum.assign(gridGraphSize, 0);
    overflowWire.assign(gridGraphSize, 0);
    overflowVia.assign(gridGraphSize, 0);
    rawRoutes.clear();
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
        throw std::runtime_error("backend=cpu_pr operation=setMap missing required GR database map data");
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
        throw std::runtime_error("backend=cpu_pr operation=route generated segment outside legal grid");
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
        usage += static_cast<float>(vias[encoded]);
    }
    if (layer >= 0) {
        usage += static_cast<float>(vias[layer * N * N + x * N + y]);
    }
    return usage;
}

float CPURouter::twoCellsViaUsage(int encoded) const {
    return std::sqrt(0.5f * (inCellViaUsage(encoded) + inCellViaUsage(encoded + 1))) * 1.5f;
}

float CPURouter::inCellUsedArea(int encoded) const {
    float used = 0.0f;
    if (encoded % N > 0) {
        used += fixed[encoded - 1] + static_cast<float>(wires[encoded - 1]);
    }
    if (encoded % N + 1 < N) {
        used += fixed[encoded] + static_cast<float>(wires[encoded]);
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
    if (capacity[encoded] < 0.01f) {
        return INF;
    }
    int expectedLen = static_cast<int>(
        (fixed[encoded] * fixedLength[encoded] + static_cast<float>(wires[encoded]) * wireDist[encoded]) /
        capacity[encoded]);
    float remain = capacity[encoded] - (fixed[encoded] + static_cast<float>(wires[encoded]) + 1.0f +
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
    int layer = encoded / N / N + 1;
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

void CPURouter::viaSweep(std::vector<int64_t>& map,
                         std::vector<int>& prev,
                         std::vector<int>& midScratch,
                         int x,
                         int y) const {
    const int stride = LAYER * N * N;
    for (int layer = 1; layer < LAYER; ++layer) {
        int cur = layer * N * N + (((layer & 1) ^ DIRECTION) ? x * N + y : y * N + x);
        midScratch[cur] = cur;
    }
    for (int layer = 2; layer < LAYER; ++layer) {
        int cur = layer * N * N + (((layer & 1) ^ DIRECTION) ? x * N + y : y * N + x);
        int last = (layer - 1) * N * N + (((layer & 1) ^ DIRECTION) ? y * N + x : x * N + y);
        if (map[4 * stride + cur] > map[4 * stride + last] + viaCostMap[last]) {
            map[4 * stride + cur] = map[4 * stride + last] + viaCostMap[last];
            prev[4 * stride + cur] = prev[4 * stride + last];
            midScratch[cur] = last;
        }
    }
    for (int layer = LAYER - 2; layer >= 1; --layer) {
        int cur = layer * N * N + (((layer & 1) ^ DIRECTION) ? x * N + y : y * N + x);
        int last = (layer + 1) * N * N + (((layer & 1) ^ DIRECTION) ? y * N + x : x * N + y);
        if (map[4 * stride + cur] > map[4 * stride + last] + viaCostMap[cur]) {
            map[4 * stride + cur] = map[4 * stride + last] + viaCostMap[cur];
            prev[4 * stride + cur] = prev[4 * stride + last];
            midScratch[cur] = last;
        }
    }
}

void CPURouter::appendWire(std::vector<int>& rawRoute, int from, int to) {
    if (from == to) {
        return;
    }
    int lo = std::min(from, to);
    int hi = std::max(from, to);
    rawRoute.push_back(lo);
    rawRoute.push_back(hi - lo);
    for (int idx = lo; idx < hi; ++idx) {
        wires[idx] += 1;
    }
}

void CPURouter::appendVia(std::vector<int>& rawRoute, int encoded) {
    rawRoute.push_back(encoded);
    rawRoute.push_back(-1);
    vias[encoded] += 1;
}

void CPURouter::routePreparedNet(const std::vector<int>& pointsBuffer, std::vector<int>& rawRoute) {
    rawRoute.clear();
    rawRoute.push_back(1);
    if (pointsBuffer.size() < 2) {
        return;
    }

    const int stride = LAYER * N * N;
    int nodeCnt = pointsBuffer[1];
    const int* points = pointsBuffer.data() + 2;
    std::vector<int64_t> map(6 * stride, PR_INF);
    std::vector<int> prev(14 * stride, 0);
    std::vector<int>& prev2 = prev;
    std::vector<int> midScratch(stride, 0);

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
                    for (int layer = 1; layer < LAYER; ++layer) {
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
                    for (int layer = 1; layer < LAYER; ++layer) {
                        if ((layer & 1) ^ DIRECTION) {
                            map[4 * stride + layer * N * N + tox * N + b] = PR_INF;
                            map[4 * stride + layer * N * N + fromx * N + b] = PR_INF;
                        } else {
                            map[4 * stride + layer * N * N + b * N + tox] = PR_INF;
                            map[4 * stride + layer * N * N + b * N + fromx] = PR_INF;
                        }
                    }
                }
                for (int layer = 1; layer < LAYER; ++layer) {
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
                for (int layer = 1; layer < LAYER; ++layer) {
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
                for (int layer = 1; layer < LAYER; ++layer) {
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
                for (int layer = 1; layer < LAYER; ++layer) {
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
                for (int layer = 1; layer < LAYER; ++layer) {
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

        for (int lower = 0; lower < LAYER; ++lower) {
            int viaSum = 0;
            int minCost = PR_INT_INF;
            int bestpos[3] = {0, 0, 0};
            int costs[3] = {
                points[t + 3] == -1 ? 0 : PR_INT_INF,
                points[t + 4] == -1 ? 0 : PR_INT_INF,
                points[t + 5] == -1 ? 0 : PR_INT_INF,
            };
            for (int upper = lower; upper < LAYER; ++upper) {
                int idx = upper * N * N + (((upper & 1) ^ DIRECTION) ? tox * N + toy : toy * N + tox);
                for (int j = 0; j < 3; ++j) {
                    if (points[t + 3 + j] != -1) {
                        if (map[idx + (j + 1) * stride] < costs[j]) {
                            costs[j] = static_cast<int>(map[idx + (j + 1) * stride]);
                            bestpos[j] = upper;
                        }
                    }
                }
                if (1LL * costs[0] + costs[1] + costs[2] < minCost) {
                    minCost = costs[0] + costs[1] + costs[2];
                }
                if (minlayer == -1 || (lower <= minlayer && maxlayer <= upper)) {
                    for (int layer = lower; layer <= upper; ++layer) {
                        int index = layer * N * N + (((layer & 1) ^ DIRECTION) ? tox * N + toy : toy * N + tox);
                        if (map[index] > minCost + viaSum) {
                            map[index] = minCost + viaSum;
                            prev[index] =
                                (((lower * LAYER + upper) * LAYER + bestpos[2]) * LAYER + bestpos[1]) * LAYER +
                                bestpos[0];
                        }
                    }
                }
                viaSum += viaCostMap[idx];
            }
        }
    }

    int64_t minTotalDist = PR_INF;
    int rootX = points[0] / N;
    int rootY = points[0] % N;
    for (int layer = 0; layer < LAYER; ++layer) {
        int idx = layer * N * N + (((layer & 1) ^ DIRECTION) ? rootX * N + rootY : rootY * N + rootX);
        if (map[idx] < minTotalDist) {
            minTotalDist = map[idx];
            map[5 * stride + points[0]] = layer;
        }
    }
    if (minTotalDist == PR_INF || minTotalDist < 0) {
        rawRoute[0] = -1;
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
            appendVia(rawRoute, id);
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
                appendWire(rawRoute, entry, p1);
                if (p1 != mid) {
                    for (int viaLayer = std::min(entrylayer, midlayer); viaLayer < std::max(entrylayer, midlayer);
                         ++viaLayer) {
                        int id = viaLayer * N * N + (((viaLayer & 1) ^ DIRECTION) ? midx * N + midy : midy * N + midx);
                        appendVia(rawRoute, id);
                    }
                }
                appendWire(rawRoute, mid, p3);
                if (p3 != p4) {
                    int tempx = p3 / N % N;
                    int tempy = p3 % N;
                    if (!(((p3 / N / N) & 1) ^ DIRECTION)) {
                        std::swap(tempx, tempy);
                    }
                    for (int viaLayer = std::min(midlayer, lastlayer); viaLayer < std::max(midlayer, lastlayer);
                         ++viaLayer) {
                        int id = viaLayer * N * N + (((viaLayer & 1) ^ DIRECTION) ? tempx * N + tempy : tempy * N + tempx);
                        appendVia(rawRoute, id);
                    }
                }
                appendWire(rawRoute, p4, last);
            } else {
                appendWire(rawRoute, last, entry);
            }
            map[5 * stride + child] = lastlayer;
        }
    }
    rawRoute[0] = static_cast<int>(rawRoute.size());
}

std::vector<int> CPURouter::buildRoutingOrder(const std::vector<GrNet>& nets, std::vector<int>& batchSizes) const {
    std::vector<int> sorted;
    sorted.reserve(nets.size());
    for (int netId = 0; netId < static_cast<int>(nets.size()); ++netId) {
        if (!nets[netId].noroute) {
            sorted.push_back(netId);
        }
    }
    std::sort(sorted.begin(), sorted.end(), [&](int lhs, int rhs) {
        int areaL = (nets[lhs].upperx - nets[lhs].lowerx + 1) * (nets[lhs].uppery - nets[lhs].lowery + 1);
        int areaR = (nets[rhs].upperx - nets[rhs].lowerx + 1) * (nets[rhs].uppery - nets[rhs].lowery + 1);
        if (areaL == areaR) {
            return lhs > rhs;
        }
        return areaL > areaR;
    });

    std::vector<short> vis(2000 * 2000, 0);
    auto hasConflict = [&](int netId) {
        const auto& net = nets[netId];
        for (int x = std::max(0, net.lowerx); x <= net.upperx; ++x) {
            for (int y = std::max(0, net.lowery); y <= net.uppery; ++y) {
                if (x < 2000 && y < 2000 && vis[x * 2000 + y]) {
                    return true;
                }
            }
        }
        return false;
    };
    auto setNet = [&](int netId, short value) {
        const auto& net = nets[netId];
        for (int x = std::max(0, net.lowerx); x <= net.upperx; ++x) {
            for (int y = std::max(0, net.lowery); y <= net.uppery; ++y) {
                if (x < 2000 && y < 2000) {
                    vis[x * 2000 + y] = value;
                }
            }
        }
    };

    std::vector<int> order;
    order.reserve(sorted.size());
    batchSizes.clear();
    int lastUnroute = 0;
    while (order.size() < sorted.size()) {
        int start = static_cast<int>(order.size());
        for (size_t i = static_cast<size_t>(lastUnroute); i < sorted.size(); ++i) {
            if (sorted[i] == -1) {
                continue;
            }
            if (!hasConflict(sorted[i])) {
                order.push_back(sorted[i]);
                setNet(sorted[i], 1);
                sorted[i] = -1;
            }
            if (static_cast<int>(order.size()) - start == 250) {
                break;
            }
        }
        while (lastUnroute < static_cast<int>(sorted.size()) && sorted[lastUnroute] == -1) {
            ++lastUnroute;
        }
        for (int i = start; i < static_cast<int>(order.size()); ++i) {
            setNet(order[i], 0);
        }
        batchSizes.push_back(static_cast<int>(order.size()) - start);
    }
    std::reverse(batchSizes.begin(), batchSizes.end());
    std::reverse(order.begin(), order.end());
    return order;
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

void CPURouter::markOverflowNets(const std::vector<GrNet>& nets) {
    std::fill(overflowWire.begin(), overflowWire.end(), 0);
    std::fill(overflowVia.begin(), overflowVia.end(), 0);
    for (int idx = 0; idx < LAYER * N * N; ++idx) {
        if (idx % N + 1 < N) {
            overflowWire[idx] = (static_cast<float>(wires[idx]) + fixed[idx] + twoCellsViaUsage(idx) > capacity[idx]);
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
        const auto& rawRoute = rawRoutes[netId];
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
    rawRoutes.clear();
    rawRoutes.resize(nets.size());
    std::vector<int> pinNum(nets.size(), 0);
    std::vector<int> routeOffsets(nets.size() + 1, 0);
    for (int netId = 0; netId < static_cast<int>(nets.size()); ++netId) {
        pinNum[netId] = static_cast<int>(nets[netId].getPins().size());
        routeOffsets[netId + 1] = routeOffsets[netId] + pinNum[netId] * MAX_ROUTE_LEN_PER_PIN;
    }

    std::vector<int> batchSizes;
    std::vector<int> order = buildRoutingOrder(nets, batchSizes);
    int start = 0;
    for (int batchSize : batchSizes) {
        calculateCostMaps();
        calculateCostSum();
        int gbPinOffset = batchSize;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
            int netId = order[start + batchIdx];
            std::vector<int> points(2 + std::max(1, pinNum[netId] * 4) * 12, 0);
            std::vector<int> gbpoints(std::max(1, pinNum[netId] * 4), -1);
            double prepareTime = 0.0;
            int len = prepare(prepareTime,
                              nets[netId],
                              points.data(),
                              routeOffsets[netId],
                              gbpoints.data(),
                              gbPinOffset,
                              X,
                              Y,
                              N,
                              LAYER,
                              DIRECTION);
            points.resize(len);
            routePreparedNet(points, rawRoutes[netId]);
            setNetRouteFromRaw(nets[netId], rawRoutes[netId]);
        }
        start += batchSize;
    }

    for (int netId = 0; netId < static_cast<int>(nets.size()); ++netId) {
        if (nets[netId].noroute || nets[netId].getPins().size() < 2) {
            rawRoutes[netId] = {0};
            nets[netId].setWires({});
            nets[netId].setVias({});
        }
    }
    markOverflowNets(nets);
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
                float wireDemand = static_cast<float>(wires[encoded]) + fixed[encoded];
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

torch::Tensor CPURouter::getRawWireDemandMap() { return buildGridTensor(wires); }

torch::Tensor CPURouter::getFixUsageMap() { return buildGridTensor(fixUsage); }

torch::Tensor CPURouter::getMovUsageMap() { return buildGridTensor(movUsage); }

void CPURouter::unsupported(const std::string& operation) const {
    throw std::runtime_error("backend=cpu_pr operation=" + operation + " is unsupported");
}

}  // namespace gr
