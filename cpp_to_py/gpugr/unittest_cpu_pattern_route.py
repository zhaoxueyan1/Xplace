#!/usr/bin/python3

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

XPLACE_ROOT = Path(__file__).resolve().parents[2]
BUILD_ROOT = XPLACE_ROOT / "build_cpu_pr"
RUNTIME_ROOT = os.environ.get("XPLACE_CPU_RUNTIME_ROOT")

if RUNTIME_ROOT:
    RUNTIME_ROOT = Path(RUNTIME_ROOT).resolve()
    EXTENSION_ROOT = RUNTIME_ROOT / "cpp_to_py/cpybin"
    GPUGR_ROOT = EXTENSION_ROOT
    IO_PARSER_ROOT = EXTENSION_ROOT
    COMMON_LIB = EXTENSION_ROOT / "libxplace_common.so"
    FLUTE_LIB = EXTENSION_ROOT / "libflute.so"
    FLUTE_LUT_ROOT = RUNTIME_ROOT / "thirdparty/flute"
    LIBRARY_DIRS = [EXTENSION_ROOT]
else:
    GPUGR_ROOT = BUILD_ROOT / "cpp_to_py/gpugr"
    IO_PARSER_ROOT = BUILD_ROOT / "cpp_to_py/io_parser"
    COMMON_LIB = BUILD_ROOT / "cpp_to_py/common/libxplace_common.so"
    FLUTE_LIB = BUILD_ROOT / "thirdparty/flute/libflute.so"
    FLUTE_LUT_ROOT = XPLACE_ROOT / "thirdparty/flute"
    LIBRARY_DIRS = [
        BUILD_ROOT / "cpp_to_py/common",
        BUILD_ROOT / "thirdparty/flute",
    ]


class TestCPUPatternRoute(unittest.TestCase):
    def _run_routeable_net(self, backend):
        gpugr_candidates = sorted(GPUGR_ROOT.glob("gpugr*.so"))
        io_parser_candidates = sorted(IO_PARSER_ROOT.glob("io_parser*.so"))
        if (
            not gpugr_candidates
            or not io_parser_candidates
            or not COMMON_LIB.exists()
            or not FLUTE_LIB.exists()
        ):
            if RUNTIME_ROOT:
                self.fail(f"CPU GPUGR runtime artifacts are incomplete under {RUNTIME_ROOT}")
            self.skipTest("CPU GPUGR runtime artifacts are not built")

        with tempfile.TemporaryDirectory(prefix="xplace_cpu_pattern_route_") as tmp:
            tmp_path = Path(tmp)
            lef_path = tmp_path / "cpu_pr.lef"
            def_path = tmp_path / "cpu_pr.def"
            lef_path.write_text(
                """VERSION 5.8 ;
BUSBITCHARS "[]" ;
DIVIDERCHAR "/" ;
UNITS
  DATABASE MICRONS 1000 ;
END UNITS
LAYER M1
  TYPE ROUTING ;
  DIRECTION HORIZONTAL ;
  PITCH 1 ;
  WIDTH 0.2 ;
  SPACING 0.2 ;
END M1
LAYER VIA1
  TYPE CUT ;
  SPACING 0.2 ;
END VIA1
LAYER M2
  TYPE ROUTING ;
  DIRECTION VERTICAL ;
  PITCH 1 ;
  WIDTH 0.2 ;
  SPACING 0.2 ;
END M2
VIA VIA12 DEFAULT
  LAYER M1 ;
    RECT -0.1 -0.1 0.1 0.1 ;
  LAYER VIA1 ;
    RECT -0.1 -0.1 0.1 0.1 ;
  LAYER M2 ;
    RECT -0.1 -0.1 0.1 0.1 ;
END VIA12
SITE core
  CLASS CORE ;
  SIZE 1 BY 1 ;
END core
MACRO BUF
  CLASS CORE ;
  ORIGIN 0 0 ;
  SIZE 1 BY 1 ;
  SYMMETRY X Y ;
  SITE core ;
  PIN A
    DIRECTION INPUT ;
    USE SIGNAL ;
    PORT
      LAYER M1 ;
      RECT 0.1 0.4 0.3 0.6 ;
    END
  END A
  PIN Y
    DIRECTION OUTPUT ;
    USE SIGNAL ;
    PORT
      LAYER M1 ;
      RECT 0.7 0.4 0.9 0.6 ;
    END
  END Y
END BUF
END LIBRARY
"""
            )
            def_path.write_text(
                """VERSION 5.8 ;
DIVIDERCHAR "/" ;
BUSBITCHARS "[]" ;
DESIGN CPU_PR_SMOKE ;
UNITS DISTANCE MICRONS 1000 ;
DIEAREA ( 0 0 ) ( 10000 10000 ) ;
ROW ROW_0 core 0 0 N DO 10 BY 1 STEP 1000 0 ;
TRACKS Y 500 DO 10 STEP 1000 LAYER M1 ;
TRACKS X 500 DO 10 STEP 1000 LAYER M2 ;
COMPONENTS 2 ;
- U0 BUF + PLACED ( 1000 1000 ) N ;
- U1 BUF + PLACED ( 8000 8000 ) N ;
END COMPONENTS
PINS 0 ;
END PINS
SPECIALNETS 0 ;
END SPECIALNETS
NETS 1 ;
- N1 ( U0 Y ) ( U1 A ) ;
END NETS
END DESIGN
"""
            )

            script = f"""
import importlib.util
import json
import resource
from pathlib import Path

import torch

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

io_parser = load_module("io_parser", Path({str(io_parser_candidates[0])!r}))
gpugr = load_module("gpugr", Path({str(gpugr_candidates[0])!r}))
assert gpugr.cuda_enabled() is False
for guarded_backend in ("cpu_pr", "cugr2"):
    try:
        gpugr.load_gr_params({{"backend": guarded_backend, "rrrIters": 1}})
    except RuntimeError as error:
        assert f"backend={{guarded_backend}}" in str(error), error
    else:
        raise AssertionError(f"{{guarded_backend}} accepted unsupported rrr_iters")
assert io_parser.load_params({{
    "lef": {str(lef_path)!r},
    "def": {str(def_path)!r},
    "lite_mode": True,
    "random_place": False,
    "num_threads": 1,
}})
rawdb = io_parser.create_database()
rawdb.load()
rawdb.setup()
gpdb = io_parser.create_gpdatabase(rawdb)
gpdb.setup()
gpugr.read_flute(
    {str(FLUTE_LUT_ROOT / "POWV9.dat")!r},
    {str(FLUTE_LUT_ROOT / "POST9.dat")!r},
)
assert gpugr.load_gr_params({{
    "backend": {backend!r},
    "rrrIters": 0,
    "route_xSize": 8,
    "route_ySize": 8,
    "bottom_routing_layer": "M1",
    "top_routing_layer": "M2",
}})
grdb = gpugr.create_grdatabase(rawdb, gpdb)
summary = grdb.grnet_summary()
assert summary["routeable_net_count"] == 1, summary
routeforce = gpugr.create_routeforce(grdb)
routeforce.run_ggr()
maps = routeforce.dmd_map()
cap_map = routeforce.cap_map()
raw_wire_map = routeforce.raw_wire_dmd_map()
assert len(maps) == 3
assert tuple(maps[0].shape) == (2, 8, 8), maps[0].shape
assert tuple(cap_map.shape) == (2, 8, 8), cap_map.shape
assert tuple(raw_wire_map.shape) == (2, 8, 8), raw_wire_map.shape
print("CPU_PATTERN_ROUTE=" + json.dumps({{
    "backend": gpugr.resolved_backend(),
    "routes": len(routeforce.route_entries()),
    "dmd_shape": list(maps[0].shape),
}}))
"""
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = ":".join(
                [*(str(path) for path in LIBRARY_DIRS), env.get("LD_LIBRARY_PATH", "")]
            )
            result = subprocess.run(
                [sys.executable, "-c", script],
                text=True,
                capture_output=True,
                timeout=30,
                env=env,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            marker = next(
                line for line in result.stdout.splitlines() if line.startswith("CPU_PATTERN_ROUTE=")
            )
            self.assertEqual(
                json.loads(marker.removeprefix("CPU_PATTERN_ROUTE=")),
                {"backend": backend, "routes": 1, "dmd_shape": [2, 8, 8]},
            )

    def test_routeable_net_uses_cpu_backend(self):
        self._run_routeable_net("cpu_pr")

    def test_routeable_net_uses_cugr2_backend(self):
        self._run_routeable_net("cugr2")


if __name__ == "__main__":
    unittest.main()
