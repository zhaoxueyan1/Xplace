# GGR: Superfast Full-Scale GPU-Accelerated Global Routing
GGR is a superfast full-scale GPU-accelerated global router developed by the research team supervised by Prof. Evangeline F. Y. Young and Prof. Martin D.F. Wong at The Chinese University of Hong Kong (CUHK). It includes an efficient and high-quality Z-shape pattern routing and a GPU-accelerated maze router GAMER.

More details are in the following paper:

Shiju Lin, Jinwei Liu, Evangeline F.Y. Young and Martin D.F. Wong. "[GAMER: GPU-Accelerated Maze Routing](https://ieeexplore.ieee.org/document/9799536)". In IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, vol. 42, no. 2, pp. 583-593, Feb. 2023. 

Shiju Lin and Martin D. F. Wong. "[Superfast Full-Scale GPU-Accelerated Global Routing](https://doi.org/10.1145/3508352.3549474)". In Proceedings of the 41st IEEE/ACM International Conference on Computer-Aided Design (ICCAD '22). Association for Computing Machinery, New York, NY, USA, Article 51, 1–8. 

**GGR is integrated in Xplace now!**

## Notes and Limitations
- We default use `N=0` (without CUDA graph optimization). When `N >= 1`, an unknown error would occur before terminating the Python process. To enable CUDA graph optimization, please set `use_tf = true` in `cpp_to_py/gpugr/gr/MazeRoute.cu` and re-compile the project. 
- GGR is **deterministic** when `N=0` but not test in `N >= 1`.
- We currently only support LEF/DEF format.
- The runtime of this version is a little bit slower than the version described in the GGR paper because we use a stronger but slower parser and make the algorithm deterministic yet robust while sacrificing the runtime.

## Parameters
Please refer to `cpp_to_py/gpugr/PyBindCppMain.cpp`.

- All backends exclude DEF `USE CLOCK` nets and nets with electrical fanout
  greater than 100 before constructing pin access groups. Fanout counts
  non-output pins in the parser's electrical direction domain; fanout 100
  remains eligible. Excluded nets contribute no routed wire/via demand.
- `clock_net_names`: Explicit STA clock identities, in addition to DEF clock
  types. Names are matched exactly; names containing `clk` are not inferred
  to be clocks. This setting resets on every `load_gr_params` call.
- `timing_route_pack()` preserves excluded net/pin IDs with `net_status=4`,
  empty geometry and `pin_vertex=-1`. `net_exclusion_reason` is 0 for ordinary
  nets, 1 for clocks, and 2 for high fanout (clock takes precedence). It also
  reports `net_fanout`, both exclusion counts, and `max_gr_fanout`.
- CUDA `route_grad` gives excluded or unconnected placement pins zero force.
  An entirely excluded design returns zero node gradients without launching
  an empty global-pin kernel. `tests/gpugr/net_exclusion.py --backend cuda
  --rrr 0` checks this interface as well as unchanged ordinary-net gradients.
- `device_id`: GPU id.
- `route_xSize` / `route_ySize`: Given GR GridGraph size. If the given size is `0`, use the `GRIDGRAPH` definition in DEF file. If the given size is `0` and there is no definition in DEF, we set it as `512`.
- `rrrIters`: The number of rip-up and re-route iterations (maze route). If `rrrIters = 0`, perform the pattern route only.
- `backend`: `cuda` selects the CUDA router, `cpu_pr` selects the serial diagnostic
  CPU pattern router, and `cpu_pr_mt` selects the deterministic parallel CPU
  pattern router. All CPU pattern backends require `rrrIters = 0`; they reject
  positive RRR values instead of silently changing the requested effort. `auto`
  does not select `cpu_pr_mt`.
- `threads`: Requested route workers for `cpu_pr_mt`. The native `run_stats()`
  result reports both requested and effective workers; the latter is clamped by
  CPU affinity, maximum batch size, and the memory budget (host available
  memory limited by the process cgroup). The same value may be used independently by the parser, so callers
  should record parser threads separately.
- `cpu_pr_mt` uses a deterministic CPU scheduler with a 4096-net batch cap and
  ends a batch after 100 consecutive bbox conflicts. These values are reported
  as `batch_size_limit` and `batch_conflict_scan_limit`; `cpu_pr` retains its
  original 250-net, full-scan scheduler and remains the serial oracle. When the
  conflict limit is reached, the remaining cost-epoch slots are packed
  deterministically. Each cost epoch uses one cost-map snapshot. Its packed,
  bbox-conflicting prefix runs serially once; the scheduler-proven
  conflict-free suffix is split into at most 64-net parallel candidate batches.
  Candidate generation uses worker-local state, and a candidate batch with an
  unexpected actual wire/via conflict is rerun serially before deterministic
  commit. This bounds cost-map refreshes and avoids speculative duplicate work
  on conflict-heavy designs without concurrent shared-state writes.
- `cpu_pr_maze` is opt-in. It initializes with `cpu_pr_mt`; `rrrIters=0`
  preserves exactly the same routes and usage. Positive RRR selects active nets
  crossing overflowing resources, orders them by decreasing bbox area then net
  ID, and searches fixed logical batches of at most 32 nets. Batch membership
  does not change with worker count. Each batch reads current immutable cost
  maps; commits refresh only dependent resources. A pending candidate is
  re-searched when an integer cost on its route actually changed. Congested
  finite costs, including the 10,000,000 cap, remain traversable; explicit
  obstacles, physical grid bounds, zero wire capacity and layer limits govern
  legality. Pin-access via ladders use the ordinary via-step legality check.
- Maze searches use an indexed A* heap with at most one entry per grid node.
  Worker selection includes live maps/routes, saved old routes, checkpoints,
  fallback scratch, net inputs and bounded pending candidates. The worker pool
  and scratch are reused across batches. `peak_memory_bytes` is a conservative
  reservation for router-owned storage, **not measured process RSS**; this is
  identified by `memory_accounting`. Record process peak RSS independently
  when benchmarking. `maze_batch_size_limit` / `maze_max_batch_size` describe
  the RRR batches; the existing batch fields describe initial PR.
- Best-state selection is lexicographic on overflow excess, overflow resource
  count, wire grid-edge count and via count. It does not optimize STA directly.
  `rrr_iterations` and accepted/initial route QoR distinguish active overflow
  nets, failed routes and unrouted eligible nets. Restoring a checkpoint rebuilds
  usage and costs from one canonical route table.
- `csrn_scale`: The size of coarsen grid in maze routing.
- `route_guide`: The file name of output route guide.
- `bottom_routing_layer` / `top_routing_layer`: Optional inclusive routing-layer
  names resolved against the LEF `TYPE ROUTING` stack. Empty values use the
  first/last routing layer. Planar edges and vias outside this interval are
  unavailable to both pattern and maze routing.

## Testing

The CPU router unit tests are plain C++17 programs with no external test
framework. Build each one from the repository root with `g++` against the
`cpp_to_py` include root and the matching implementation file:

```bash
# Batch occupancy invariant checks (header-only implementation).
g++ -std=c++17 -O2 -Icpp_to_py \
    -o /tmp/unittest_batch_occupancy \
    cpp_to_py/gpugr/unittest_batch_occupancy.cpp

# Batch scheduler packing/conflict-limit behavior.
g++ -std=c++17 -O2 -Icpp_to_py \
    -o /tmp/unittest_cpu_route_batch_scheduler \
    cpp_to_py/gpugr/unittest_cpu_route_batch_scheduler.cpp \
    cpp_to_py/gpugr/gr/CpuRouteBatchScheduler.cpp

# Scratch sizing, resetRegion validation, and checked arithmetic.
g++ -std=c++17 -O2 -Icpp_to_py \
    -o /tmp/unittest_cpu_route_types \
    cpp_to_py/gpugr/unittest_cpu_route_types.cpp \
    cpp_to_py/gpugr/gr/CpuRouteTypes.cpp

# Worker pool lifecycle and error propagation (requires pthread).
g++ -std=c++17 -O2 -pthread -Icpp_to_py \
    -o /tmp/unittest_cpu_route_worker_pool \
    cpp_to_py/gpugr/unittest_cpu_route_worker_pool.cpp \
    cpp_to_py/gpugr/gr/CpuRouteWorkerPool.cpp \
    cpp_to_py/gpugr/gr/CpuRouteResourceLimits.cpp
```

Each binary returns a zero exit status on success. The worker pool test also
has a ThreadSanitizer variant for the parallel dispatch paths:

```bash
g++ -std=c++17 -O1 -g -fsanitize=thread -pthread -Icpp_to_py \
    -o /tmp/unittest_cpu_route_worker_pool_tsan \
    cpp_to_py/gpugr/unittest_cpu_route_worker_pool.cpp \
    cpp_to_py/gpugr/gr/CpuRouteWorkerPool.cpp \
    cpp_to_py/gpugr/gr/CpuRouteResourceLimits.cpp
```

## Citation
If you find **GGR** useful in your research, please consider to cite:
```bibtex
@inproceedings{lin2022ggr,
author = {Lin, Shiju and Wong, Martin D. F.},
booktitle = {Proceedings of the 41st IEEE/ACM International Conference on Computer-Aided Design},
title = {Superfast Full-Scale GPU-Accelerated Global Routing},
year = {2022},
}

@article{lin2023gamer,
  author={Lin, Shiju and Liu, Jinwei and Young, Evangeline F. Y. and Wong, Martin D. F.},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems}, 
  title={GAMER: GPU-Accelerated Maze Routing}, 
  year={2023},
```


## Example
```python
# main_test_gr.py
import torch
import time

from utils import IOParser
from cpp_to_py import gpugr
from src import Flute
from src.core.route_force import calc_gr_wl_via, estimate_num_shorts

num_threads = 20
gpu_id = 0
Flute.register(num_threads)
torch.cuda.synchronize("cuda:{}".format(gpu_id))

# 1) Benchmark Setting
root = "your_path"
design_name = "ispd19_test9"
params = {
    "benchmark": "iccad2019",
    "lef": "%s/%s/%s.input.lef" % (root, design_name, design_name),
    "def": "%s/%s/%s.input.def" % (root, design_name, design_name),
    "design_name": design_name,
}
route_guide_file = "test.guide"

# 2) LEF/DEF Parser
print("--- Start GR ---")
start_gr_time = time.time()
parser = IOParser()
rawdb, gpdb = parser.read(
    params, verbose_log=True, lite_mode=True, random_place=False, num_threads=num_threads
)

# 3) Construct Global Routing Database and Run GGR
gpugr.load_gr_params(
    {
        "device_id": gpu_id,
        "route_xSize": 0,
        "route_ySize": 0,
        "rrrIters": 1,
        "route_guide": route_guide_file,
        "bottom_routing_layer": "MET2",
        "top_routing_layer": "MET5",
    }
)
grdb = gpugr.create_grdatabase(rawdb, gpdb)
print("GPUGR layer range:", grdb.routing_layer_range(), grdb.routing_layer_names())
routeforce = gpugr.create_routeforce(grdb)
routeforce.run_ggr()
end_gr_time = time.time()
print("--- End GR ---")

# 4) Report Global Routing Statistics
skip_m1_route = True
m1direction = gpdb.m1direction()  # 0 for H, 1 for V, metal1's layer idx is 0
hId = 1 if m1direction else 0
vId = 0 if m1direction else 1
if skip_m1_route:
    hId = hId + 2 if hId == 0 else hId
    vId = vId + 2 if vId == 0 else vId

dmd_map, wire_dmd_map, via_dmd_map = routeforce.dmd_map()
cap_map: torch.Tensor = routeforce.cap_map()

cg_mapH = dmd_map[hId::2].sum(dim=0) / cap_map[hId::2].sum(dim=0)
cg_mapV = dmd_map[vId::2].sum(dim=0) / cap_map[vId::2].sum(dim=0)
cg_mapHV = torch.stack((cg_mapH, cg_mapV))
cg_mapHV = torch.where(cg_mapHV > 1, cg_mapHV - 1, 0)

numOvflNets = routeforce.num_ovfl_nets()
gr_wirelength, gr_numVias = calc_gr_wl_via(grdb, routeforce)
gr_numShorts = estimate_num_shorts(routeforce, gpdb, cap_map, wire_dmd_map, via_dmd_map)

gr_time = end_gr_time - start_gr_time

print(
    "#OvflNets: %d, GR WL: %d, GR #Vias: %d, #EstShorts: %d | GR Time: %.4f"
    % (numOvflNets, gr_wirelength, gr_numVias, gr_numShorts, gr_time)
)
```

## Contact

[Shiju Lin](https://appsrv.cse.cuhk.edu.hk/~sjlin/) (sjlin@cse.cuhk.edu.hk) and [Lixin Liu](https://liulixinkerry.github.io/) (lxliu@cse.cuhk.edu.hk)


## License

GGR is an open source project licensed under a BSD 3-Clause License that can be found in the [LICENSE](../../LICENSE) file.
