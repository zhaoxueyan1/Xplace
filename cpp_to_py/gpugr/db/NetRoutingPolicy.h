#pragma once

namespace db { class Net; }

namespace gr {
enum class NetExclusion { None = 0, Clock = 1, HighFanout = 2 };
constexpr int MAX_GR_FANOUT = 100;

// Electrical loads, not access groups or total pins. The DEF parser reverses
// chip port directions, so an INPUT port is a driver in this domain.
int netFanout(const db::Net& net);
NetExclusion routingExclusion(const db::Net& net);
}  // namespace gr
