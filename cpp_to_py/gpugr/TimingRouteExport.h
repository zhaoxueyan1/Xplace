#pragma once
#include <pybind11/pybind11.h>

namespace gr {
class RouteForce;
}

namespace Xplace {
pybind11::dict exportTimingRoutes(gr::RouteForce& routeforce);
}
