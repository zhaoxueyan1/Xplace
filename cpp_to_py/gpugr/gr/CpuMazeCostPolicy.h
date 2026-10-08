#pragma once

#include <algorithm>
#include <cmath>

namespace gr {

// Match RouteForce's CUDA RRR schedule. iteration zero is pattern routing;
// iteration 1..rerouteIterations consumes the remaining congestion budget.
// CPU map construction applies the discount to undiscounted layer costs.
struct CpuMazeCostPolicy {
    double logisticSlope;
    double viaMultiplier;
    double shortViolationDiscount;

    static CpuMazeCostPolicy forIteration(int iteration, int rerouteIterations) {
        return {std::ldexp(1.0, std::min(iteration, 30)),
                iteration == 0 ? 1.0 : std::max(100.0 / std::pow(5.0, iteration - 1), 4.0),
                iteration == 0 ? 0.1 : 0.1 + 0.9 * iteration / rerouteIterations};
    }
};

}  // namespace gr
