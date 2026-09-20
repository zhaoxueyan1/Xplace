import sys
import tempfile
import unittest
from pathlib import Path


XPLACE_ROOT = Path(__file__).resolve().parents[2]
if str(XPLACE_ROOT) not in sys.path:
    sys.path.insert(0, str(XPLACE_ROOT))


class CugrNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from cpp_to_py.cpybin import cugr
        except Exception as exc:  # pragma: no cover - build guard
            raise unittest.SkipTest(f"CUGR extension is unavailable: {exc}")
        cls.cugr = cugr
        toy_root = XPLACE_ROOT / "thirdparty/cu-gr/toys/iccad2019c/ispd18_sample"
        cls.lef = toy_root / "ispd18_sample.input.lef"
        cls.def_file = toy_root / "ispd18_sample.input.def"
        cls.lut_root = XPLACE_ROOT / "thirdparty/cu-gr/src/flute"
        if not all(path.is_file() for path in (cls.lef, cls.def_file)):
            raise unittest.SkipTest("bundled CUGR toy input is unavailable")

    def setUp(self):
        self.cugr.reset()

    def tearDown(self):
        self.cugr.reset()

    def _options(self, **overrides):
        options = {
            "lefs": [str(self.lef)],
            "def": str(self.def_file),
            "powv_file": str(self.lut_root / "POWV9.dat"),
            "post_file": str(self.lut_root / "POST9.dat"),
            "route_num_bins_x": 16,
            "route_num_bins_y": 16,
            "threads": 1,
            "bottom_routing_layer": "Metal2",
            "top_routing_layer": "Metal5",
            "design_identity": "native-unit-toy-v1",
        }
        options.update(overrides)
        return options

    def test_rrr_zero_and_one_translate_to_total_passes(self):
        initial = self.cugr.run(self._options(rrr_iters=0))
        rerouted = self.cugr.run(self._options(rrr_iters=1))
        self.assertEqual(initial["total_passes"], 1)
        self.assertEqual(rerouted["total_passes"], 2)
        self.assertGreaterEqual(initial["short_vio_area"], 0.0)
        self.assertGreaterEqual(rerouted["short_vio_area"], 0.0)
        self.assertGreater(initial["peak_rss_mb"], 0.0)
        self.assertGreaterEqual(rerouted["peak_rss_mb"], initial["peak_rss_mb"])
        self.assertEqual(initial["session_mode"], "full_reparse_reset")
        self.assertEqual(rerouted["session_mode"], "coordinate_refresh")
        self.assertTrue(all(net["routed"] for net in initial["nets"]))

    def test_unknown_option_and_missing_def_fail_before_native_parse(self):
        with self.assertRaisesRegex(ValueError, "unknown option"):
            self.cugr.run(self._options(unknown_option=True))

        options = self._options()
        options["def"] = str(self.def_file.with_name("missing.def"))
        with self.assertRaisesRegex(ValueError, "DEF file does not exist"):
            self.cugr.run(options)

    def test_same_design_identity_reuses_session_and_changed_identity_reparses(self):
        first = self.cugr.run(self._options(rrr_iters=0))
        second = self.cugr.run(self._options(rrr_iters=0))
        changed = self.cugr.run(self._options(rrr_iters=0, design_identity="native-unit-toy-v2"))
        self.assertFalse(first["session_cache_hit"])
        self.assertTrue(second["session_cache_hit"])
        self.assertFalse(changed["session_cache_hit"])
        self.assertEqual(changed["session_mode"], "full_reparse_reset")

    def test_coordinate_refresh_changes_and_restores_placement_state(self):
        original_text = self.def_file.read_text(encoding="utf-8")
        original_line = "- inst2015 NAND3X2 + PLACED ( 88000 78660 ) N ;"
        moved_line = "- inst2015 NAND3X2 + PLACED ( 90000 78660 ) N ;"
        self.assertIn(original_line, original_text)

        with tempfile.TemporaryDirectory(prefix="cugr_refresh_") as directory:
            moved_def = Path(directory) / "moved.def"
            moved_def.write_text(original_text.replace(original_line, moved_line, 1), encoding="utf-8")

            first = self.cugr.run(self._options(rrr_iters=0))
            moved = self.cugr.run(self._options(rrr_iters=0, **{"def": str(moved_def)}))
            restored = self.cugr.run(self._options(rrr_iters=0))

        self.assertFalse(first["session_cache_hit"])
        self.assertTrue(moved["session_cache_hit"])
        self.assertTrue(restored["session_cache_hit"])
        self.assertEqual(moved["session_mode"], "coordinate_refresh")
        self.assertEqual(restored["session_mode"], "coordinate_refresh")
        self.assertGreater(moved["total_sec"], 0.0)
        self.assertGreater(moved["refresh_sec"], 0.0)
        self.assertNotEqual(first["movable_usage"], moved["movable_usage"])
        self.assertEqual(first["movable_usage"], restored["movable_usage"])

    def test_coordinate_refresh_is_crash_free_for_supported_thread_counts(self):
        original_text = self.def_file.read_text(encoding="utf-8")
        original_line = "- inst2015 NAND3X2 + PLACED ( 88000 78660 ) N ;"
        moved_line = "- inst2015 NAND3X2 + PLACED ( 90000 78660 ) N ;"
        self.assertIn(original_line, original_text)

        with tempfile.TemporaryDirectory(prefix="cugr_thread_refresh_") as directory:
            moved_def = Path(directory) / "moved.def"
            moved_def.write_text(original_text.replace(original_line, moved_line, 1), encoding="utf-8")

            for threads in (1, 2, 8):
                self.cugr.reset()
                options = self._options(threads=threads)
                first = self.cugr.run(options)
                moved = self.cugr.run({**options, "def": str(moved_def)})
                restored = self.cugr.run(options)

                self.assertEqual(first["session_mode"], "full_reparse_reset")
                self.assertEqual(moved["session_mode"], "coordinate_refresh")
                self.assertEqual(restored["session_mode"], "coordinate_refresh")
                self.assertEqual(first["movable_usage"], restored["movable_usage"])

    def test_failed_refresh_invalidates_session_and_next_valid_run_reparses(self):
        original_text = self.def_file.read_text(encoding="utf-8")
        invalid_text = original_text.replace(
            "- inst2015 NAND3X2 + PLACED ( 88000 78660 ) N ;",
            "- missing_cell NAND3X2 + PLACED ( 88000 78660 ) N ;",
            1,
        )
        self.assertNotEqual(original_text, invalid_text)

        with tempfile.TemporaryDirectory(prefix="cugr_failed_refresh_") as directory:
            invalid_def = Path(directory) / "invalid.def"
            invalid_def.write_text(invalid_text, encoding="utf-8")

            self.cugr.run(self._options(rrr_iters=0))
            with self.assertRaisesRegex(RuntimeError, "refresh phase failed"):
                self.cugr.run(self._options(rrr_iters=0, **{"def": str(invalid_def)}))

            recovered = self.cugr.run(self._options(rrr_iters=0))

        self.assertFalse(recovered["session_cache_hit"])
        self.assertEqual(recovered["session_mode"], "full_reparse_reset")

    def test_fixed_usage_is_stable_when_only_movable_geometry_changes(self):
        original_text = self.def_file.read_text(encoding="utf-8")
        fixed_line = "- inst2015 NAND3X2 + PLACED ( 88000 78660 ) N ;"
        moved_line = "- inst2591 NAND4X2 + PLACED ( 100000 71820 ) N ;"
        fixed_instance_line = "- inst2015 NAND3X2 + FIXED ( 88000 78660 ) N ;"
        moved_instance_line = "- inst2591 NAND4X2 + PLACED ( 104000 71820 ) N ;"
        self.assertIn(fixed_line, original_text)
        self.assertIn(moved_line, original_text)

        fixed_text = original_text.replace(fixed_line, fixed_instance_line, 1)
        moved_text = fixed_text.replace(moved_line, moved_instance_line, 1)
        with tempfile.TemporaryDirectory(prefix="cugr_fixed_refresh_") as directory:
            directory = Path(directory)
            fixed_def = directory / "fixed.def"
            moved_def = directory / "moved.def"
            fixed_def.write_text(fixed_text, encoding="utf-8")
            moved_def.write_text(moved_text, encoding="utf-8")

            first = self.cugr.run(self._options(**{"def": str(fixed_def)}))
            moved = self.cugr.run(self._options(**{"def": str(moved_def)}))

        self.assertGreater(sum(first["fixed_usage"]), 0.0)
        self.assertEqual(first["fixed_usage"], moved["fixed_usage"])
        self.assertNotEqual(first["movable_usage"], moved["movable_usage"])


if __name__ == "__main__":
    unittest.main()
