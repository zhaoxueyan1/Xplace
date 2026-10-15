#include "CpuMazeFrontier.h"

#include <algorithm>

namespace gr {

void CpuMazeFrontier::reset(std::size_t nodeCount) {
    for (const auto& entry : heap_) positions_[std::get<2>(entry)] = -1;
    heap_.clear();
    if (positions_.size() != nodeCount) positions_.assign(nodeCount, -1);
    heap_.reserve(nodeCount);
}

void CpuMazeFrontier::exchange(std::size_t lhs, std::size_t rhs) {
    std::swap(heap_[lhs], heap_[rhs]);
    positions_[std::get<2>(heap_[lhs])] = static_cast<int>(lhs);
    positions_[std::get<2>(heap_[rhs])] = static_cast<int>(rhs);
}

void CpuMazeFrontier::pushOrDecrease(int node, int64_t priority, int64_t distance) {
    const Entry entry{priority, distance, node};
    int position = positions_[node];
    if (position < 0) {
        position = static_cast<int>(heap_.size());
        positions_[node] = position;
        heap_.push_back(entry);
    } else {
        heap_[position] = entry;
    }
    while (position > 0) {
        const int parent = (position - 1) / 2;
        if (!(heap_[position] < heap_[parent])) break;
        exchange(position, parent);
        position = parent;
    }
}

int CpuMazeFrontier::pop() {
    const int node = std::get<2>(heap_.front());
    exchange(0, heap_.size() - 1);
    heap_.pop_back();
    positions_[node] = -1;
    std::size_t position = 0;
    while (position * 2 + 1 < heap_.size()) {
        std::size_t child = position * 2 + 1;
        if (child + 1 < heap_.size() && heap_[child + 1] < heap_[child]) ++child;
        if (!(heap_[child] < heap_[position])) break;
        exchange(position, child);
        position = child;
    }
    return node;
}

std::size_t CpuMazeFrontier::bytes() const {
    return heap_.capacity() * sizeof(Entry) + positions_.capacity() * sizeof(int);
}

}  // namespace gr
