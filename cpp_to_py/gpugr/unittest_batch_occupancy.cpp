#include "gpugr/gr/BatchOccupancy.h"

#include <cassert>
#include <algorithm>
#include <vector>

class ReferenceOccupancy {
public:
    void reset(int x_size, int y_size) {
        xSize = x_size;
        ySize = y_size;
        bits.assign(xSize * ySize, 0);
    }

    bool noConflict(int xl, int xr, int yl, int yr) const {
        if (!normalizeBox(xl, xr, yl, yr)) {
            return true;
        }
        for (int x = xl; x <= xr; ++x) {
            for (int y = yl; y <= yr; ++y) {
                if (bits[x * ySize + y]) {
                    return false;
                }
            }
        }
        return true;
    }

    void insert(int xl, int xr, int yl, int yr) {
        set(xl, xr, yl, yr, 1);
    }

    void remove(int xl, int xr, int yl, int yr) {
        set(xl, xr, yl, yr, 0);
    }

private:
    int xSize = 0;
    int ySize = 0;
    std::vector<int> bits;

    bool normalizeBox(int& xl, int& xr, int& yl, int& yr) const {
        if (xSize <= 0 || ySize <= 0) {
            return false;
        }
        xl = std::max(0, xl);
        yl = std::max(0, yl);
        xr = std::min(xr, xSize - 1);
        yr = std::min(yr, ySize - 1);
        return xl <= xr && yl <= yr;
    }

    void set(int xl, int xr, int yl, int yr, int value) {
        if (!normalizeBox(xl, xr, yl, yr)) {
            return;
        }
        for (int x = xl; x <= xr; ++x) {
            for (int y = yl; y <= yr; ++y) {
                bits[x * ySize + y] = value;
            }
        }
    }
};

void compareWithReference() {
    gr::BatchOccupancy occupancy;
    ReferenceOccupancy reference;
    occupancy.reset(6, 130);
    reference.reset(6, 130);

    struct Box {
        int xl;
        int xr;
        int yl;
        int yr;
    };
    const std::vector<Box> boxes = {
        {1, 3, 63, 65},
        {0, 0, 0, 129},
        {-4, 1, -2, 2},
        {4, 9, 127, 140},
        {2, 2, 64, 64},
        {3, 4, 65, 66},
    };

    for (const Box& box : boxes) {
        assert(occupancy.noConflict(box.xl, box.xr, box.yl, box.yr) ==
               reference.noConflict(box.xl, box.xr, box.yl, box.yr));
        if (reference.noConflict(box.xl, box.xr, box.yl, box.yr)) {
            occupancy.insert(box.xl, box.xr, box.yl, box.yr);
            reference.insert(box.xl, box.xr, box.yl, box.yr);
        }
    }

    for (const Box& box : boxes) {
        assert(occupancy.noConflict(box.xl, box.xr, box.yl, box.yr) ==
               reference.noConflict(box.xl, box.xr, box.yl, box.yr));
    }

    for (const Box& box : boxes) {
        occupancy.remove(box.xl, box.xr, box.yl, box.yr);
        reference.remove(box.xl, box.xr, box.yl, box.yr);
        assert(occupancy.noConflict(box.xl, box.xr, box.yl, box.yr) ==
               reference.noConflict(box.xl, box.xr, box.yl, box.yr));
    }
}

int main() {
    gr::BatchOccupancy occupancy;
    occupancy.reset(8, 130);

    assert(occupancy.noConflict(1, 3, 63, 65));
    occupancy.insert(1, 3, 63, 65);

    assert(!occupancy.noConflict(2, 2, 64, 64));
    assert(!occupancy.noConflict(3, 4, 65, 66));
    assert(occupancy.noConflict(4, 4, 63, 65));
    assert(occupancy.noConflict(1, 3, 66, 66));

    occupancy.remove(1, 3, 63, 65);
    assert(occupancy.noConflict(1, 3, 63, 65));

    occupancy.insert(0, 0, 0, 129);
    assert(!occupancy.noConflict(0, 0, 0, 0));
    assert(!occupancy.noConflict(0, 0, 129, 129));
    assert(occupancy.noConflict(1, 1, 0, 129));

    compareWithReference();

    return 0;
}
