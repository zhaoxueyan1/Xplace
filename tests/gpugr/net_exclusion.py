"""Exercise the real CPU/CUDA router, including zero-access excluded nets.

Run with --build <native build> --backend cpu_pr_mt|cuda --output <directory>.
CUDA --rrr 0 also checks the public route-gradient interface.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

XPLACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(XPLACE / "cpp_to_py/gpugr"))
from cpu_route_toy_fixture import TOY_DEF, TOY_LEF

ROUTING_LEF = TOY_LEF.replace("      LAYER M1 ;", "      LAYER M2 ;").replace(
    "SITE core\n",
    """LAYER VIA2
  TYPE CUT ;
  SPACING 0.2 ;
END VIA2
LAYER M3
  TYPE ROUTING ;
  DIRECTION HORIZONTAL ;
  PITCH 1 ;
  WIDTH 0.2 ;
  SPACING 0.2 ;
END M3
VIA VIA23 DEFAULT
  LAYER M2 ;
    RECT -0.1 -0.1 0.1 0.1 ;
  LAYER VIA2 ;
    RECT -0.1 -0.1 0.1 0.1 ;
  LAYER M3 ;
    RECT -0.1 -0.1 0.1 0.1 ;
END VIA23
SITE core
""",
).replace("RECT 0.1 0.4 0.3 0.6", "RECT 0 0 1 1").replace(
    "RECT 0.7 0.4 0.9 0.6", "RECT 0 0 1 1"
)


def extension(name, build):
    path = next((build / "cpp_to_py" / name).glob(f"{name}*.so"))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def design(variant):
    nets = [
        ("clk_data", 1, "SIGNAL"),
        ("limit100", 100, "SIGNAL"),
        ("not_named_clk", 1, "CLOCK"),
        ("sta_clock", 1, "SIGNAL"),
        ("limit101", 101, "SIGNAL"),
    ]
    components, records = [], []
    for name, fanout, use in nets:
        for pin in range(fanout + 1):
            x, y = (1000, 1000) if pin == 0 else (8000, 5000)
            components.append(f"- {name}_{pin} BUF + PLACED ( {x} {y} ) N ;")
        excluded = use == "CLOCK" or name == "sta_clock" or fanout > 100
        if (variant == "baseline" and excluded) or (variant == "excluded" and not excluded):
            continue
        pins = " ".join(f"( {name}_{pin} {'Y' if pin == 0 else 'A'} )" for pin in range(fanout + 1))
        records.append(f"- {name} {pins} + USE {use} ;")
    return (
        TOY_DEF.split("COMPONENTS 4 ;")[0].replace("LAYER M1 ;", "LAYER M1 M3 ;")
        + f"COMPONENTS {len(components)} ;\n" + "\n".join(components)
        + "\nEND COMPONENTS\nPINS 0 ;\nEND PINS\nSPECIALNETS 0 ;\nEND SPECIALNETS\n"
        + f"NETS {len(records)} ;\n" + "\n".join(records) + "\nEND NETS\nEND DESIGN\n"
    )


def route_gradient(router, gpdb):
    wire, via = router.dmd_map()[1:]
    shape = wire.shape[-2:]
    options = {"dtype": torch.float32, "device": "cuda"}
    plane = torch.ones(shape, **options)
    weights = torch.ones(max(shape) + 2, **options)
    node2pin = gpdb.node2pin_info_tensor()
    gradient = router.route_grad(
        plane, wire.sum(0), via.sum(0), plane, weights, weights,
        torch.ones((2, *shape), **options),
        node2pin[1].cuda(), node2pin[2].cuda(),
        1.0, 1.0, 1.0, len(gpdb.node_id2node_name()),
    )
    torch.cuda.synchronize()
    assert bool(torch.isfinite(gradient).all())
    names = gpdb.node_id2node_name()
    excluded = [
        index for index, name in enumerate(names)
        if name.startswith(("not_named_clk_", "sta_clock_", "limit101_"))
    ]
    torch.testing.assert_close(gradient[excluded], torch.zeros_like(gradient[excluded]))
    return gradient.cpu().numpy()


def route(io, gr, args, variant):
    directory = args.output / variant
    directory.mkdir(parents=True)
    lef, deffile = directory / "toy.lef", directory / "toy.def"
    lef.write_text(ROUTING_LEF)
    deffile.write_text(design(variant))
    assert io.load_params({"lef": str(lef), "def": str(deffile), "lite_mode": True,
                           "random_place": False, "num_threads": 2})
    rawdb = io.create_database()
    rawdb.load()
    rawdb.setup()
    gpdb = io.create_gpdatabase(rawdb)
    gpdb.setup()
    assert gr.load_gr_params({"backend": args.backend, "rrrIters": args.rrr,
                             "threads": 2, "route_xSize": 8, "route_ySize": 8,
                             "bottom_routing_layer": "M2", "top_routing_layer": "M3",
                             "clock_net_names": ["sta_clock"]})
    database = gr.create_grdatabase(rawdb, gpdb)
    summary = dict(database.grnet_summary())
    router = gr.create_routeforce(database)
    router.run_ggr()
    pack = router.timing_route_pack()
    demand = tuple(t.cpu().numpy().copy() for t in router.dmd_map())
    demand += (router.raw_wire_dmd_map().cpu().numpy().copy(),)
    raw = dict(zip(pack["net_names"], router.raw_routes(), strict=True))
    report = {"summary": summary, "net_names": pack["net_names"],
              "status": pack["net_status"].tolist(), "fanout": pack["net_fanout"].tolist(),
              "reason": pack["net_exclusion_reason"].tolist(),
              "clock_count": pack["excluded_clock_net_count"],
              "high_fanout_count": pack["excluded_high_fanout_net_count"]}
    gradient = route_gradient(router, gpdb) if args.backend == "cuda" and args.rrr == 0 else None
    report["route_gradient_checked"] = gradient is not None
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return pack, demand, raw, gradient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu_pr_mt", "cuda"], required=True)
    parser.add_argument("--rrr", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rrr is None:
        args.rrr = 3 if args.backend == "cuda" else 0
    torch.set_num_threads(2)
    io, gr = extension("io_parser", args.build), extension("gpugr", args.build)
    gr.read_flute(str(XPLACE / "thirdparty/flute/POWV9.dat"), str(XPLACE / "thirdparty/flute/POST9.dat"))
    baseline, base_demand, base_raw, base_gradient = route(io, gr, args, "baseline")
    pack, demand, raw, gradient = route(io, gr, args, "mixed")
    assert pack["net_names"] == ["clk_data", "limit100", "not_named_clk", "sta_clock", "limit101"]
    np.testing.assert_array_equal(pack["net_status"], [0, 0, 4, 4, 4])
    np.testing.assert_array_equal(pack["net_exclusion_reason"], [0, 0, 1, 1, 2])
    np.testing.assert_array_equal(pack["net_fanout"], [1, 100, 1, 1, 101])
    assert pack["excluded_clock_net_count"] == 2
    assert pack["excluded_high_fanout_net_count"] == 1
    for name in base_raw:
        assert raw[name] == base_raw[name], (name, raw[name], base_raw[name])
    for actual, expected in zip(demand, base_demand, strict=True):
        np.testing.assert_array_equal(actual, expected)
    if gradient is not None:
        assert np.any(base_gradient)
        np.testing.assert_array_equal(gradient, base_gradient)
    for net in (2, 3, 4):
        assert pack["net_vertex_start"][net] == pack["net_vertex_start"][net + 1]
        assert pack["net_edge_start"][net] == pack["net_edge_start"][net + 1]
        np.testing.assert_array_equal(pack["pin_vertex"][pack["pin_net"] == net], -1)
        assert raw[pack["net_names"][net]] == [0]
    empty, empty_demand, _, empty_gradient = route(io, gr, args, "excluded")
    np.testing.assert_array_equal(empty["net_status"], [4, 4, 4])
    assert len(empty["vertex_net"]) == len(empty["edge_from"]) == 0
    # dmd_map includes cell obstacles; raw wires and vias are route-only.
    for plane in empty_demand[2:]:
        assert not np.any(plane), plane
    if empty_gradient is not None:
        np.testing.assert_array_equal(empty_gradient, np.zeros_like(empty_gradient))
    suffix = ", zero excluded force and unchanged signal gradients" if gradient is not None else ""
    print(f"PASS {args.backend}: clock types and STA identities, fanout 100/101, unchanged signal demand, all excluded{suffix}")


if __name__ == "__main__":
    main()
