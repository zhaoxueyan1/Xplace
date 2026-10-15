#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

namespace gr {

// An indexed min heap holds at most one entry per grid node. Decrease-key
// preserves the former (A* priority, distance, encoded node) tie order.
class CpuMazeFrontier {
public:
    void reset(std::size_t nodeCount);
    void pushOrDecrease(int node, int64_t priority, int64_t distance);
    int pop();
    bool empty() const { return heap_.empty(); }
    std::size_t bytes() const;

private:
    using Entry = std::tuple<int64_t, int64_t, int>;
    void exchange(std::size_t lhs, std::size_t rhs);
    std::vector<Entry> heap_;
    std::vector<int> positions_;
};

}  // namespace gr
