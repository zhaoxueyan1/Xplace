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


class TestPortlessLefPin(unittest.TestCase):
    def test_portless_pin_loads_without_geometry(self):
        candidates = sorted((BUILD_ROOT / "cpp_to_py" / "io_parser").glob("io_parser*.so"))
        if not candidates:
            self.skipTest("build_cpu_pr io_parser extension is not built")

        with tempfile.TemporaryDirectory(prefix="xplace_portless_pin_") as tmp:
            tmp_path = Path(tmp)
            lef_path = tmp_path / "portless.lef"
            def_path = tmp_path / "portless.def"
            lef_path.write_text(
                """VERSION 5.8 ;
BUSBITCHARS \"[]\" ;
DIVIDERCHAR \"/\" ;
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
MACRO PORTLESS
  CLASS CORE ;
  ORIGIN 0 0 ;
  SIZE 1 BY 1 ;
  SYMMETRY X Y ;
  SITE core ;
  PIN A
    DIRECTION INPUT ;
    USE SIGNAL ;
  END A
END PORTLESS
END LIBRARY
"""
            )
            def_path.write_text(
                """VERSION 5.8 ;
DIVIDERCHAR \"/\" ;
BUSBITCHARS \"[]\" ;
DESIGN PORTLESS_DESIGN ;
UNITS DISTANCE MICRONS 1000 ;
DIEAREA ( 0 0 ) ( 1000 1000 ) ;
COMPONENTS 0 ;
END COMPONENTS
PINS 0 ;
END PINS
SPECIALNETS 0 ;
END SPECIALNETS
NETS 0 ;
END NETS
END DESIGN
"""
            )

            script = f"""
import importlib.util
from pathlib import Path

so = Path({str(candidates[0])!r})
spec = importlib.util.spec_from_file_location('io_parser', so)
io_parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(io_parser)
assert io_parser.load_params({{'lef': {str(lef_path)!r}, 'def': {str(def_path)!r}, 'lite_mode': True, 'random_place': False}})
database = io_parser.create_database()
database.load()
print('portless_pin_loaded')
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
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    env=env,
                )
            except subprocess.TimeoutExpired as exc:
                self.fail(f"portless LEF PIN parsing timed out: {exc}")

            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("portless_pin_loaded", result.stdout)


if __name__ == "__main__":
    unittest.main()
