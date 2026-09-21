#!/usr/bin/python3

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cpu_route_toy_fixture import TOY_DEF, TOY_LEF

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
    def _run_routeable_net(self, backend, threads=1, expected_workers=1):
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
            lef_path.write_text(TOY_LEF)
            def_path.write_text(TOY_DEF)

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
for guarded_backend in ("cpu_pr", "cpu_pr_mt"):
    try:
        gpugr.load_gr_params({{"backend": guarded_backend, "rrrIters": 1}})
    except RuntimeError as error:
        assert f"backend={{guarded_backend}}" in str(error), error
    else:
        raise AssertionError(f"{{guarded_backend}} accepted unsupported rrr_iters")
try:
    gpugr.load_gr_params({{"backend": "cpu_pr_mt", "rrrIters": 0, "threads": 0}})
except RuntimeError as error:
    assert "threads must be positive" in str(error), error
else:
    raise AssertionError("cpu_pr_mt accepted a non-positive thread count")
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
    "threads": {threads},
    "route_xSize": 8,
    "route_ySize": 8,
    "bottom_routing_layer": "M1",
    "top_routing_layer": "M2",
}})
grdb = gpugr.create_grdatabase(rawdb, gpdb)
summary = grdb.grnet_summary()
assert summary["routeable_net_count"] == 2, summary
routeforce = gpugr.create_routeforce(grdb)
routeforce.run_ggr()
resolved_backend = gpugr.resolved_backend()
run_stats = routeforce.run_stats()
if {backend!r} in ("cpu_pr", "cpu_pr_mt"):
    assert run_stats["terminal_status"] == "completed", run_stats
    assert run_stats["requested_workers"] == {threads}, run_stats
    assert run_stats["effective_workers"] == {expected_workers}, run_stats
    expected_batch_limit = 4096 if {backend!r} == "cpu_pr_mt" else 250
    expected_scan_limit = 100 if {backend!r} == "cpu_pr_mt" else 0
    assert run_stats["batch_size_limit"] == expected_batch_limit, run_stats
    assert run_stats["batch_conflict_scan_limit"] == expected_scan_limit, run_stats
if {expected_workers} > 1:
    assert run_stats["parallel_batch_count"] > 0, run_stats
maps = routeforce.dmd_map()
cap_map = routeforce.cap_map()
raw_wire_map = routeforce.raw_wire_dmd_map()
assert len(maps) == 3
assert tuple(maps[0].shape) == (2, 8, 8), maps[0].shape
assert tuple(cap_map.shape) == (2, 8, 8), cap_map.shape
assert tuple(raw_wire_map.shape) == (2, 8, 8), raw_wire_map.shape
if {backend!r} == "cpu_pr_mt":
    parallel_routes = routeforce.raw_routes()
    parallel_maps = tuple(route_map.clone() for route_map in maps)
    assert gpugr.load_gr_params({{
        "backend": "cpu_pr",
        "rrrIters": 0,
        "threads": 1,
        "route_xSize": 8,
        "route_ySize": 8,
        "bottom_routing_layer": "M1",
        "top_routing_layer": "M2",
    }})
    reference_grdb = gpugr.create_grdatabase(rawdb, gpdb)
    reference = gpugr.create_routeforce(reference_grdb)
    reference.run_ggr()
    assert reference.raw_routes() == parallel_routes
    reference_stats = reference.run_stats()
    for hash_field in ("route_hash64", "wire_map_hash64", "via_map_hash64"):
        assert reference_stats[hash_field] == run_stats[hash_field], (
            hash_field,
            reference_stats,
            run_stats,
        )
    for reference_map, parallel_map in zip(reference.dmd_map(), parallel_maps):
        assert torch.equal(reference_map, parallel_map)
print("CPU_PATTERN_ROUTE=" + json.dumps({{
    "backend": resolved_backend,
    "routes": len(routeforce.route_entries()),
    "dmd_shape": list(maps[0].shape),
    "effective_workers": run_stats["effective_workers"],
}}))
"""
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = ":".join(
                [*(str(path) for path in LIBRARY_DIRS), env.get("LD_LIBRARY_PATH", "")]
            )
            env["XPLACE_FLUTE_LUT_ROOT"] = str(FLUTE_LUT_ROOT)
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
                {
                    "backend": backend,
                    "routes": 2,
                    "dmd_shape": [2, 8, 8],
                    "effective_workers": expected_workers,
                },
            )

    def test_routeable_net_uses_cpu_backend(self):
        self._run_routeable_net("cpu_pr")

    def test_routeable_net_uses_cpu_parallel_backend_plumbing(self):
        self._run_routeable_net("cpu_pr_mt")

    def test_routeable_nets_use_two_cpu_parallel_workers(self):
        self._run_routeable_net("cpu_pr_mt", threads=2, expected_workers=2)


if __name__ == "__main__":
    unittest.main()
