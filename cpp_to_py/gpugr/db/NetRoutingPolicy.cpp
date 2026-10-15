#include "NetRoutingPolicy.h"
#include "GRSetting.h"
#include "common/db/Geometry.h"
#include "common/db/Net.h"
#include "common/db/Pin.h"
#include <algorithm>

namespace gr {
int netFanout(const db::Net& net) {
    return std::count_if(net.pins.begin(), net.pins.end(), [](const db::Pin* pin) {
        return pin->type->direction() != 'o';
    });
}

NetExclusion routingExclusion(const db::Net& net) {
    if (net._type == 'c' || grSetting.clockNetNames.count(net.name)) return NetExclusion::Clock;
    if (netFanout(net) > MAX_GR_FANOUT) return NetExclusion::HighFanout;
    return NetExclusion::None;
}
}  // namespace gr
