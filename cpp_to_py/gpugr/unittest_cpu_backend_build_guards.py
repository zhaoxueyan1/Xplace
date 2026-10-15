#!/usr/bin/python3

import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


XPLACE_ROOT = Path(__file__).resolve().parents[2]


class TestGGRCPUBackendBuildGuards(unittest.TestCase):
    def test_top_level_cmake_has_explicit_cuda_option(self):
        text = (XPLACE_ROOT / "CMakeLists.txt").read_text()
        self.assertRegex(text, r"option\s*\(\s*XPLACE_ENABLE_CUDA")
        self.assertNotIn('Xplace only supports Torch-CUDA mode', text)
        self.assertRegex(text, r"project\s*\(\s*xplace\s+LANGUAGES\s+C\s+CXX\s*\)")
        self.assertRegex(text, r"enable_language\s*\(\s*CUDA\s*\)")

    def test_cpp_to_py_cuda_only_subdirs_are_guarded(self):
        text = (XPLACE_ROOT / "cpp_to_py" / "CMakeLists.txt").read_text()
        for subdir in (
            "dct_cuda",
            "density_map_cuda",
            "gpudp",
            "hpwl_cuda",
            "wa_wirelength_hpwl_cuda",
            "wirelength_timing_cuda",
        ):
            self.assertRegex(
                text,
                rf"if\s*\(\s*TORCH_ENABLE_CUDA\s*\)[\s\S]*add_subdirectory\s*\(\s*{re.escape(subdir)}\s*\)",
                msg=f"{subdir} must be guarded by TORCH_ENABLE_CUDA",
            )

    def test_gpugr_cmake_excludes_cuda_sources_when_disabled(self):
        text = (XPLACE_ROOT / "cpp_to_py" / "gpugr" / "CMakeLists.txt").read_text()
        self.assertRegex(text, r"if\s*\(\s*TORCH_ENABLE_CUDA\s*\)[\s\S]*SRC_FILES_GGR_CUDA")
        self.assertRegex(text, r"ENABLE_CUDA=\$\{TORCH_ENABLE_CUDA\}")
        self.assertRegex(text, r"target_compile_definitions\s*\(\s*ggr[\s\S]*ENABLE_CUDA=\$\{TORCH_ENABLE_CUDA\}")

    def test_cpu_build_import_reports_cuda_disabled_when_present(self):
        cpu_build_dir = XPLACE_ROOT / "build_cpu_pr" / "cpp_to_py" / "gpugr"
        candidates = sorted(cpu_build_dir.glob("gpugr*.so"))
        if not candidates:
            self.skipTest("build_cpu_pr gpugr extension is not built")
        script = f"""
import importlib.util
import pathlib
so = pathlib.Path({str(candidates[0])!r})
spec = importlib.util.spec_from_file_location('gpugr', so)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
assert mod.cuda_enabled() is False
try:
    mod.load_gr_params({{'backend': 'bad'}})
except Exception as exc:
    assert 'Unsupported gpugr backend' in str(exc)
else:
    raise AssertionError('invalid backend did not fail')
print('cpu_build_import_ok')
"""
        result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True, check=True)
        self.assertIn("cpu_build_import_ok", result.stdout)

    def test_cpu_package_import_does_not_require_cuda_extensions(self):
        cpu_build_root = XPLACE_ROOT / "build_cpu_pr" / "cpp_to_py"
        gpugr_candidates = sorted((cpu_build_root / "gpugr").glob("gpugr*.so"))
        io_parser_candidates = sorted((cpu_build_root / "io_parser").glob("io_parser*.so"))
        common_lib = cpu_build_root / "common" / "libxplace_common.so"
        flute_lib = XPLACE_ROOT / "build_cpu_pr" / "thirdparty" / "flute" / "libflute.so"
        if not gpugr_candidates or not io_parser_candidates or not common_lib.exists() or not flute_lib.exists():
            self.skipTest("build_cpu_pr package import dependencies are not built")

        with tempfile.TemporaryDirectory(prefix="xplace_cpu_pkg_") as tmp:
            tmp_path = Path(tmp)
            package_dir = tmp_path / "cpp_to_py"
            cpybin_dir = package_dir / "cpybin"
            cpybin_dir.mkdir(parents=True)
            shutil.copy2(XPLACE_ROOT / "cpp_to_py" / "__init__.py", package_dir / "__init__.py")
            for path in (gpugr_candidates[0], io_parser_candidates[0], common_lib, flute_lib):
                shutil.copy2(path, cpybin_dir / path.name)

            script = """
from cpp_to_py import gpugr
assert gpugr.cuda_enabled() is False
print('cpu_package_import_ok')
"""
            result = subprocess.run(
                [sys.executable, "-c", script],
                text=True,
                capture_output=True,
                env={"PYTHONPATH": str(tmp_path)},
                check=True,
            )
            self.assertIn("cpu_package_import_ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
