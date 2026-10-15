#!/usr/bin/python3

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


XPLACE_ROOT = Path(__file__).resolve().parents[2]
BUILD_ROOT = XPLACE_ROOT / "build_cpu_pr"


class TestPortlessIOPinGRNet(unittest.TestCase):
    def test_portless_io_pin_net_is_not_routed(self):
        gpugr_candidates = sorted((BUILD_ROOT / "cpp_to_py" / "gpugr").glob("gpugr*.so"))
        io_parser_candidates = sorted((BUILD_ROOT / "cpp_to_py" / "io_parser").glob("io_parser*.so"))
        common_lib = BUILD_ROOT / "cpp_to_py" / "common" / "libxplace_common.so"
        flute_lib = BUILD_ROOT / "thirdparty" / "flute" / "libflute.so"
        if not gpugr_candidates or not io_parser_candidates or not common_lib.exists() or not flute_lib.exists():
            self.skipTest("build_cpu_pr GPUGR artifacts are not built")

        with tempfile.TemporaryDirectory(prefix="xplace_portless_io_grnet_") as tmp:
            tmp_path = Path(tmp)
            lef_path = tmp_path / "portless_io.lef"
            def_path = tmp_path / "portless_io.def"
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
  PITCH 100 ;
  WIDTH 40 ;
  SPACING 40 ;
END M1
SITE core
  CLASS CORE ;
  SIZE 1 BY 1 ;
END core
MACRO CORE_DUMMY
  CLASS CORE ;
  ORIGIN 0 0 ;
  SIZE 1 BY 1 ;
  SYMMETRY X Y ;
  SITE core ;
END CORE_DUMMY
END LIBRARY
"""
            )
            def_path.write_text(
                """VERSION 5.8 ;
DIVIDERCHAR "/" ;
BUSBITCHARS "[]" ;
DESIGN PORTLESS_IO_DESIGN ;
UNITS DISTANCE MICRONS 1000 ;
DIEAREA ( 0 0 ) ( 1000 1000 ) ;
ROW ROW_0 core 0 0 N DO 1 BY 1 STEP 1000 0 ;
COMPONENTS 0 ;
END COMPONENTS
PINS 1 ;
- PAD + NET N1 + DIRECTION INPUT + USE SIGNAL ;
END PINS
SPECIALNETS 0 ;
END SPECIALNETS
NETS 1 ;
- N1 ( PIN PAD ) ;
END NETS
END DESIGN
"""
            )

            script = f"""
import importlib.util
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
params = {{
    "lef": {str(lef_path)!r},
    "def": {str(def_path)!r},
    "lite_mode": False,
    "random_place": False,
    "num_threads": 1,
}}
assert io_parser.load_params(params)
rawdb = io_parser.create_database()
rawdb.load()
rawdb.setup()
gpdb = io_parser.create_gpdatabase(rawdb)
gpdb.setup()
assert gpugr.load_gr_params({{
    "backend": "cpu_pr",
    "rrrIters": 0,
    "route_xSize": 8,
    "route_ySize": 8,
}})
grdb = gpugr.create_grdatabase(rawdb, gpdb)
summary = grdb.grnet_summary()
assert summary["num_gr_nets"] == 1, summary
assert summary["routeable_net_count"] == 0, summary
assert summary["noroute_net_count"] == 1, summary
assert summary["pin_group_count"] == 0, summary
print("portless_io_grnet_skipped")
"""
            env = os.environ.copy()
            library_dirs = [
                BUILD_ROOT / "cpp_to_py" / "common",
                BUILD_ROOT / "thirdparty" / "flute",
            ]
            env["LD_LIBRARY_PATH"] = ":".join(
                [*(str(path) for path in library_dirs), env.get("LD_LIBRARY_PATH", "")]
            )
            try:
                result = subprocess.run(
                    [sys.executable, "-c", script],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                    env=env,
                )
            except subprocess.TimeoutExpired as exc:
                self.fail(f"portless I/O pin GR setup timed out: {exc}")

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout.count("Skip GR pin without access points"), 1, result.stdout)
            self.assertIn("portless_io_grnet_skipped", result.stdout)


if __name__ == "__main__":
    unittest.main()
