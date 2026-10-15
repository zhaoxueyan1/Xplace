import torch
import importlib

_EXTENSION_NAMES = [
    "dct_cuda",
    "flute_cpp",
    "hpwl_cuda",
    "io_parser",
    "density_map_cuda",
    "draw_placement",
    "wa_wirelength_hpwl_cuda",
    "gpugr",
    "gpudp",
    "routedp",
    "gputimer",
    "wirelength_timing_cuda"
]

__all__ = []
_extension_import_errors = {}

for _name in _EXTENSION_NAMES:
    try:
        globals()[_name] = importlib.import_module(f"{__name__}.cpybin.{_name}")
        __all__.append(_name)
    except ImportError as exc:
        _extension_import_errors[_name] = exc


def __getattr__(name):
    if name in _extension_import_errors:
        raise ImportError(f"cpp_to_py extension '{name}' is unavailable") from _extension_import_errors[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


del _name
