#!/usr/bin/env python3
"""Qualify CPU backend dispatch inside an ENABLE_CUDA=1 gpugr build.

Loads a CUDA-enabled gpugr extension by explicit path, routes the shared toy
fixture with cpu_pr and cpu_pr_mt(1/2/8), and proves:

1. the build matches the expected CUDA capability (``--expect-cuda on|off``);
2. every CPU backend run reaches ``terminal_status == "completed"``;
3. raw routes and native hashes match the serial cpu_pr oracle exactly;
4. CUDA is never initialized by the CPU dispatch (``torch.cuda.is_initialized()``
   stays False).

Run with a CUDA-enabled torch interpreter, with the build's native library
dirs on LD_LIBRARY_PATH:

    LD_LIBRARY_PATH=<build>/cpp_to_py/common:<build>/thirdparty/flute \
    XPLACE_FLUTE_LUT_ROOT=<xplace>/thirdparty/flute \
    <cu128-python> qualify_cuda_build_cpu_dispatch.py \
        --gpugr-so <build>/cpp_to_py/gpugr/gpugr<suffix>.so \
        --io-parser-so <build>/cpp_to_py/io_parser/io_parser<suffix>.so
"""

import argparse
import importlib.util
import json
import os
import resource
import tempfile
from pathlib import Path

from cpu_route_toy_fixture import TOY_DEF, TOY_LEF


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def route_toy(gpugr, io_parser, rawdb, gpdb, backend, threads):
    assert gpugr.load_gr_params(
        {
            "backend": backend,
            "rrrIters": 0,
            "threads": threads,
            "route_xSize": 8,
            "route_ySize": 8,
            "bottom_routing_layer": "M1",
            "top_routing_layer": "M2",
        }
    )
    grdb = gpugr.create_grdatabase(rawdb, gpdb)
    routeforce = gpugr.create_routeforce(grdb)
    routeforce.run_ggr()
    run_stats = routeforce.run_stats()
    assert run_stats["terminal_status"] == "completed", run_stats
    assert run_stats["requested_workers"] == threads, run_stats
    return {
        "raw_routes": routeforce.raw_routes(),
        "route_hash64": run_stats["route_hash64"],
        "wire_map_hash64": run_stats["wire_map_hash64"],
        "via_map_hash64": run_stats["via_map_hash64"],
        "effective_workers": run_stats["effective_workers"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpugr-so", required=True, type=Path)
    parser.add_argument("--io-parser-so", required=True, type=Path)
    parser.add_argument("--expect-cuda", choices=("on", "off"), default="on")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    import torch

    io_parser = load_module("io_parser", args.io_parser_so)
    gpugr = load_module("gpugr", args.gpugr_so)

    expected_cuda = args.expect_cuda == "on"
    assert gpugr.cuda_enabled() is expected_cuda, (
        f"expected cuda_enabled() == {expected_cuda} for {args.gpugr_so}"
    )

    with tempfile.TemporaryDirectory(prefix="gpugr_cuda_build_cpu_dispatch_") as tmp:
        tmp_path = Path(tmp)
        lef_path = tmp_path / "toy.lef"
        def_path = tmp_path / "toy.def"
        lef_path.write_text(TOY_LEF)
        def_path.write_text(TOY_DEF)

        assert io_parser.load_params(
            {
                "lef": str(lef_path),
                "def": str(def_path),
                "lite_mode": True,
                "random_place": False,
                "num_threads": 1,
            }
        )
        rawdb = io_parser.create_database()
        rawdb.load()
        rawdb.setup()
        gpdb = io_parser.create_gpdatabase(rawdb)
        gpdb.setup()

        lut_root = Path(os.environ["XPLACE_FLUTE_LUT_ROOT"])
        gpugr.read_flute(str(lut_root / "POWV9.dat"), str(lut_root / "POST9.dat"))

        reference = route_toy(gpugr, io_parser, rawdb, gpdb, "cpu_pr", 1)
        runs = {"cpu_pr_t1": reference}
        for threads in (1, 2, 8):
            runs[f"cpu_pr_mt_t{threads}"] = route_toy(
                gpugr, io_parser, rawdb, gpdb, "cpu_pr_mt", threads
            )

    for name, run in runs.items():
        for field in ("raw_routes", "route_hash64", "wire_map_hash64", "via_map_hash64"):
            assert run[field] == reference[field], (name, field, run, reference)

    evidence = {
        "gpugr_extension": str(args.gpugr_so),
        "cuda_enabled": gpugr.cuda_enabled(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_initialized_by_cpu_dispatch": torch.cuda.is_initialized(),
        "runs": {
            name: {key: value for key, value in run.items() if key != "raw_routes"}
            for name, run in runs.items()
        },
        "raw_route_parity": True,
    }
    assert evidence["cuda_initialized_by_cpu_dispatch"] is False, evidence

    text = json.dumps(evidence, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
