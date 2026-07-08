import importlib.util
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch


def load_visualization_module():
    module_path = os.path.join(
        os.path.dirname(__file__),
        "utils",
        "visualization.py",
    )
    spec = importlib.util.spec_from_file_location("xplace_visualization", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InitialDensityPlotTest(unittest.TestCase):
    def test_save_initial_density_map_writes_png_and_npy(self):
        visualization = load_visualization_module()
        density_map = torch.tensor([[0.0, 0.5], [1.0, 1.5]], dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                result_dir=tmpdir,
                exp_id="plot-test",
                eval_dir="eval",
                design_name="unit",
            )
            paths = visualization.save_initial_density_map(
                density_map,
                args,
                tag="unit_test",
            )

            self.assertTrue(os.path.exists(paths["png"]))
            self.assertTrue(os.path.exists(paths["npy"]))


if __name__ == "__main__":
    unittest.main()
