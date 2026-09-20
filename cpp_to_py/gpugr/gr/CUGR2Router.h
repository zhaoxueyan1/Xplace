#pragma once

#include "gpugr/db/GRDatabase.h"
#include "gpugr/gr/CPURouter.h"

#include <tuple>

namespace gr {

class CUGR2Router {
public:
    void initialize(const GRDatabase& grdb);
    void route(GRDatabase& grdb);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> getDemandMap();
    torch::Tensor getCapacityMap();
    torch::Tensor getRawWireDemandMap();
    torch::Tensor getFixUsageMap();
    torch::Tensor getMovUsageMap();
    int getNumOvflNets() const { return numOvflNets_; }
    std::vector<std::vector<int>> getRawRoutes() const { return accumulator_.getRawRoutes(); }

private:
    void installFluteProvider();

    CPURouter accumulator_;
    int numOvflNets_ = 0;
    std::string flutePowvFile_;
    std::string flutePostFile_;
};

}  // namespace gr
