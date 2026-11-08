"""Regression tests for the acceptance driver's failure propagation."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "verify_packages", Path(__file__).resolve().parents[1] / "verify-packages.py"
)
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


class AcceptanceDriverTests(unittest.TestCase):
    def test_failure_continues_to_other_workspaces_and_survives_in_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "results.json"
            outcomes = [SimpleNamespace(returncode=code) for code in [0, 101, 0, 0, 0]]
            with patch("sys.argv", ["verify-packages.py", "--check", "--report", str(report)]), \
                    patch.object(DRIVER.subprocess, "run", side_effect=outcomes) as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(DRIVER.main(), 1)
            rows = json.loads(report.read_text())
            self.assertEqual(run.call_count, 5)
            self.assertEqual([row["exit_code"] for row in rows], [0, 101, 0, 0, 0])
            self.assertEqual(rows[1]["workspace"], "ruv-swarm")
            self.assertTrue(all("--locked" in row["command"] for row in rows))

    def test_missing_cargo_is_failure_not_a_passing_empty_check(self):
        with patch("sys.argv", ["verify-packages.py", "--check", "--workspace", "."]), \
                patch.object(DRIVER.subprocess, "run", side_effect=FileNotFoundError("cargo")), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(DRIVER.main(), 1)

    def test_security_regression_target_runs_even_if_library_tests_fail(self):
        with patch("sys.argv", ["verify-packages.py", "--test", "--workspace", "cuda-wasm"]), \
                patch.object(DRIVER.subprocess, "run", side_effect=[
                    SimpleNamespace(returncode=101), SimpleNamespace(returncode=0),
                ]) as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(DRIVER.main(), 1)
        self.assertEqual(run.call_count, 2)
        self.assertIn("safety_regressions", run.call_args_list[1].args[0])

    def test_success_returns_zero(self):
        with patch("sys.argv", ["verify-packages.py", "--check", "--workspace", "."]), \
                patch.object(DRIVER.subprocess, "run", return_value=SimpleNamespace(returncode=0)), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(DRIVER.main(), 0)


if __name__ == "__main__":
    unittest.main()
