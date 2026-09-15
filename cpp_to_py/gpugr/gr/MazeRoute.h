#pragma once
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <vector>

namespace gr {

extern const int MAX_LAYER_NUM, MAX_PIN_NUM;

class GPUMazeRouter {
public:
    void startGPU(int device_id, int layer, int x, int y, int routing_layer_begin = 0, int routing_layer_end = -1);
    void endGPU();
    void ensureBatchStorage(int batchSize, int maxPinsPerNet);
    void query();
    void run(int DIRECTION, int iterleft);

    void getResults(double &t,
                    int64_t *costSum,
                    int *pins,
                    int *dist,
                    int *fgprev,
                    int *wireCost,
                    int *viaCost,
                    int *wires,
                    int *vias,
                    int *routes,
                    int N,
                    int SCALE,
                    int DIRECTION,
                    int routesOffset,
                    int netId,
                    int *batchNetIds,
                    int64_t *routeLengths,
                    const int64_t *routeOffsets);

    const static int MAX_TOT_PIN_NUM = 1000000, MAX_NUM_NET = 100;

    int MAX_TURN_NUM = 10;
    int LAYER, X, Y, NX, NY; // 0: x-1,x,x+1...  1: y-1,y,y+1...
    int routingLayerBegin = 0, routingLayerEnd = -1;
    int *costMap = nullptr, *viaMap = nullptr, *markMap = nullptr, *cudaRoutedPin = nullptr;

    int *cost = nullptr, *via = nullptr, *costL = nullptr, *costR = nullptr,
        *cudaMap = nullptr, *reset = nullptr;
    int *cudaPrev = nullptr;
    const int MAX_PIN_SIZE_PER_NET = 500000, MAX_PIN_NUM = 10000;

    int firstTime = 0; // varible to determine whether the first run of TF
    int batchCapacity = 0;
    int pinCapacity = 0;
    int pinBatchCapacity = 0;
};
extern int counter1, counter2, counter3, counter4;
extern double time1;

}  // namespace gr
