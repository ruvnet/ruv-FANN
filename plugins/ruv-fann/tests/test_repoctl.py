#!/usr/bin/env python3
"""Real-process tests for the plugin runner; no model, network or build needed."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


sys.dont_write_bytecode = True
PLUGIN = Path(__file__).resolve().parents[1]
RUNNER = PLUGIN / "scripts" / "repoctl.py"
SPEC = importlib.util.spec_from_file_location("repoctl", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fixture(root: Path) -> None:
    write(root / "Cargo.toml", '[package]\nname="ruv-fann"\nversion="0.2.2"\n')
    for directory in MODULE.WORKSPACES[1:]:
        write(root / directory / "Cargo.toml",
              f'[package]\nname="{directory}"\nversion="0.2.0"\n')
    for path in ("package.json", "ruv-swarm/npm/package.json", "cuda-wasm/package.json"):
        write(root / path, json.dumps({"name": path.split("/")[0], "version": "2.0.0"}))
    write(root / "README.md", "Fixture checkout\n")
    write(root / ".mcp.json", '{"mcpServers":{"custom":{"command":"keep-me"}}}\n')
    write(root / ".claude/settings.json", '{"permissions":{"deny":["Bash(curl:*)"]}}\n')
    write(root / "scripts/verify-packages.py", 'import json,sys\nprint(json.dumps(sys.argv[1:]))\n')


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="ruv-plugin-tests-")
        self.root = Path(self.temp.name) / "checkout with spaces"
        fixture(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def invoke(self, *args: str, repo: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, "-B", str(RUNNER), *args, "--repo", str(repo or self.root)],
                              text=True, capture_output=True, timeout=10, check=False)

    def test_doctor_inventory_and_no_child_execution(self) -> None:
        marker = self.root / "unexpected-execution"
        write(self.root / "scripts/verify-packages.py", f'open({str(marker)!r},"w").close()\n')
        output = self.invoke("doctor")
        data = json.loads(output.stdout)
        self.assertEqual(output.returncode, 0, output.stderr)
        self.assertTrue(data["read_only"])
        self.assertEqual(len(data["packages"]), 8)
        self.assertEqual(data["packages"][0]["version"], "0.2.2")
        self.assertFalse(marker.exists())

    def test_plan_exact_argv_and_offline_environment(self) -> None:
        output = self.invoke("plan", "--profile", "forecasting")
        data = json.loads(output.stdout)
        self.assertEqual(output.returncode, 0)
        self.assertEqual(data["commands"][0]["argv"][-3:], ["--check", "--workspace", "neuro-divergent"])
        self.assertEqual(data["commands"][1]["argv"][-3:], ["--test", "--workspace", "neuro-divergent"])
        self.assertEqual(data["environment"]["CARGO_NET_OFFLINE"], "true")
        self.assertEqual(data["commands"][0]["cwd"], str(self.root))

    def test_all_profiles_are_explicit_and_bounded(self) -> None:
        for profile in MODULE.PROFILES:
            with self.subTest(profile=profile):
                data = json.loads(self.invoke("plan", "--profile", profile).stdout)
                self.assertLessEqual(len(data["commands"]), 2)
                self.assertTrue(all("verify-packages.py" in command["argv"][2]
                                    for command in data["commands"]))

    def test_verify_invokes_real_script_and_preserves_user_config(self) -> None:
        protected = [self.root / ".mcp.json", self.root / ".claude/settings.json"]
        before = [path.read_bytes() for path in protected]
        data = json.loads(self.invoke("verify", "--profile", "docs").stdout)
        self.assertEqual(data["status"], "passed")
        self.assertEqual(json.loads(data["results"][0]["output"]), ["--docs"])
        self.assertEqual([path.read_bytes() for path in protected], before)
        self.assertFalse(list(self.root.rglob("__pycache__")))

    def test_code_profile_runs_both_checks_in_order(self) -> None:
        data = json.loads(self.invoke("verify", "--profile", "cuda").stdout)
        self.assertEqual(data["status"], "passed")
        self.assertEqual([json.loads(item["output"])[0] for item in data["results"]],
                         ["--check", "--test"])

    def test_nonzero_is_failure_and_stops_remaining_checks(self) -> None:
        write(self.root / "scripts/verify-packages.py", 'import sys\nprint("check failed")\nsys.exit(7)\n')
        output = self.invoke("verify", "--profile", "core")
        data = json.loads(output.stdout)
        self.assertEqual(output.returncode, 1)
        self.assertEqual(len(data["results"]), 1)
        self.assertEqual(data["results"][0]["reason"], "nonzero_exit")
        self.assertEqual(data["results"][0]["exit_code"], 7)

    def test_deadline_kills_process_group(self) -> None:
        write(self.root / "scripts/verify-packages.py",
              'import subprocess,sys,time\n'
              'child=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"])\n'
              'print(child.pid,flush=True)\ntime.sleep(30)\n')
        started = time.monotonic()
        output = self.invoke("verify", "--timeout", "0.2")
        data = json.loads(output.stdout)
        self.assertEqual(output.returncode, 1)
        self.assertLess(time.monotonic() - started, 3)
        result = data["results"][0]
        self.assertEqual(result["reason"], "timeout")
        child_pid = int(result["output"].strip())
        status = Path(f"/proc/{child_pid}/status")
        if status.exists():
            self.assertIn("State:\tZ", status.read_text())

    def test_output_budget_fails_closed(self) -> None:
        write(self.root / "scripts/verify-packages.py", 'import sys\nsys.stdout.write("x"*1000000)\n')
        output = self.invoke("verify")
        data = json.loads(output.stdout)
        self.assertEqual(output.returncode, 1)
        self.assertEqual(data["results"][0]["reason"], "output_limit")
        self.assertTrue(data["results"][0]["output_truncated"])
        self.assertEqual(len(data["results"][0]["output"]), MODULE.OUTPUT_LIMIT)

    def test_success_does_not_leave_background_child(self) -> None:
        write(self.root / "scripts/verify-packages.py",
              'import subprocess,sys\n'
              'child=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"],'
              'stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n'
              'print(child.pid,flush=True)\n')
        output = self.invoke("verify")
        data = json.loads(output.stdout)
        self.assertEqual(output.returncode, 0)
        child_pid = int(data["results"][0]["output"].strip())
        time.sleep(0.05)
        status = Path(f"/proc/{child_pid}/status")
        if status.exists():
            self.assertIn("State:\tZ", status.read_text())

    def test_metadata_report_budget_fails_closed(self) -> None:
        write(self.root / "package.json", json.dumps({"name": "x" * 300_000}))
        output = self.invoke("doctor")
        self.assertEqual(output.returncode, 2)
        self.assertLess(len(output.stdout), 1000)
        self.assertIn("report exceeds", json.loads(output.stdout)["error"])

    def test_literal_shell_characters_in_repo_path(self) -> None:
        special = Path(self.temp.name) / "repo $(touch INJECTED); 'quoted'"
        fixture(special)
        output = self.invoke("verify", repo=special)
        self.assertEqual(output.returncode, 0, output.stdout + output.stderr)
        self.assertFalse((special / "INJECTED").exists())
        self.assertFalse((Path(self.temp.name) / "INJECTED").exists())

    def test_invalid_repository_name_rejected_before_execution(self) -> None:
        write(self.root / "Cargo.toml", '[package]\nname="wrong"\nversion="1.0.0"\n')
        output = self.invoke("verify")
        self.assertEqual(output.returncode, 2)
        self.assertEqual(json.loads(output.stdout)["status"], "invalid_input")

    def test_missing_repository_file(self) -> None:
        (self.root / "README.md").unlink()
        self.assertEqual(self.invoke("doctor").returncode, 2)

    def test_escaping_script_symlink_rejected(self) -> None:
        target = Path(self.temp.name) / "outside.py"
        write(target, 'raise Exception("must not execute")\n')
        script = self.root / "scripts/verify-packages.py"
        script.unlink()
        script.symlink_to(target)
        self.assertEqual(self.invoke("verify").returncode, 2)

    def test_internal_script_symlink_also_rejected(self) -> None:
        script = self.root / "scripts/verify-packages.py"
        script.rename(script.with_name("real.py"))
        script.symlink_to("real.py")
        self.assertEqual(self.invoke("plan").returncode, 2)

    def test_parent_directory_symlink_rejected(self) -> None:
        (self.root / "scripts").rename(self.root / "real-scripts")
        (self.root / "scripts").symlink_to("real-scripts", target_is_directory=True)
        self.assertEqual(self.invoke("doctor").returncode, 2)

    def test_workspace_manifest_escape_rejected(self) -> None:
        write(self.root / "ruv-swarm/Cargo.toml", '[workspace]\nmembers=["../../outside"]\n')
        self.assertEqual(self.invoke("doctor").returncode, 2)

    def test_malformed_toml_rejected(self) -> None:
        write(self.root / "Cargo.toml", "not valid TOML here")
        self.assertEqual(self.invoke("doctor").returncode, 2)

    def test_wrong_toml_table_type_rejected_as_json(self) -> None:
        for manifest, value in (("Cargo.toml", 'package="ruv-fann"\n'),
                                ("ruv-swarm/Cargo.toml", 'workspace="wrong"\n')):
            with self.subTest(manifest=manifest):
                original = (self.root / manifest).read_bytes()
                write(self.root / manifest, value)
                output = self.invoke("doctor")
                self.assertEqual(output.returncode, 2)
                self.assertEqual(json.loads(output.stdout)["status"], "invalid_input")
                self.assertNotIn("Traceback", output.stderr)
                (self.root / manifest).write_bytes(original)

    def test_malformed_json_rejected(self) -> None:
        write(self.root / "package.json", "{")
        self.assertEqual(self.invoke("doctor").returncode, 2)

    def test_oversized_manifest_rejected(self) -> None:
        write(self.root / "Cargo.toml", "#" * (MODULE.FILE_LIMIT + 1))
        self.assertEqual(self.invoke("doctor").returncode, 2)

    def test_invalid_profile_rejected(self) -> None:
        output = self.invoke("verify", "--profile", "publish")
        self.assertEqual(output.returncode, 2)
        self.assertEqual(json.loads(output.stdout)["status"], "invalid_input")

    def test_invalid_deadlines_rejected(self) -> None:
        for timeout in ("0", "nan", "inf", "1801", "-1"):
            with self.subTest(timeout=timeout):
                self.assertEqual(self.invoke("verify", "--timeout", timeout).returncode, 2)

    def test_child_cargo_environment_is_offline(self) -> None:
        write(self.root / "scripts/verify-packages.py",
              'import json,os\nprint(json.dumps({k:os.environ[k] for k in '
              '["CARGO_NET_OFFLINE","GIT_OPTIONAL_LOCKS","PYTHONDONTWRITEBYTECODE"]}))\n')
        data = json.loads(self.invoke("verify").stdout)
        self.assertEqual(json.loads(data["results"][0]["output"]), MODULE.OFFLINE_ENV)

    def test_doctor_and_plan_do_not_change_checkout(self) -> None:
        def snapshot() -> dict[str, bytes]:
            return {str(path.relative_to(self.root)): path.read_bytes()
                    for path in self.root.rglob("*") if path.is_file()}
        before = snapshot()
        self.assertEqual(self.invoke("doctor").returncode, 0)
        self.assertEqual(self.invoke("plan").returncode, 0)
        self.assertEqual(snapshot(), before)

    def test_workspace_inherited_version(self) -> None:
        write(self.root / "ruv-swarm/Cargo.toml",
              '[workspace]\nmembers=["core"]\n[workspace.package]\nversion="1.0.8"\n')
        write(self.root / "ruv-swarm/core/Cargo.toml",
              '[package]\nname="ruv-swarm-core"\nversion.workspace=true\n')
        data = json.loads(self.invoke("doctor").stdout)
        package = next(item for item in data["packages"] if item["name"] == "ruv-swarm-core")
        self.assertEqual(package["version"], "1.0.8")


class CheckoutIntegrationTests(unittest.TestCase):
    """Runs the real checked-out repository's cheap integration gates, if present."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = PLUGIN.parents[1]
        if not (cls.repo / "Cargo.toml").is_file():
            raise unittest.SkipTest("plugin-only distribution; repository integration is separate")

    def test_actual_checkout_inventory(self) -> None:
        result = subprocess.run([sys.executable, "-B", str(RUNNER), "doctor", "--repo", str(self.repo)],
                                text=True, capture_output=True, check=False, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        cargo = [item for item in json.loads(result.stdout)["packages"] if item["kind"] == "cargo"]
        self.assertEqual(len(cargo), 26)
        self.assertEqual(len({item["manifest"] for item in cargo}), 26)

    def test_actual_checkout_readme_svg_gate(self) -> None:
        result = subprocess.run([sys.executable, "-B", str(RUNNER), "verify", "--profile", "docs",
                                 "--repo", str(self.repo)], text=True, capture_output=True,
                                check=False, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("README local links and SVG source checks passed", result.stdout)

    def test_actual_checkout_git_inventory_gate(self) -> None:
        if not (self.repo / ".git").exists():
            self.skipTest("git checkout is required by verify-packages inventory")
        result = subprocess.run([sys.executable, "-B", str(RUNNER), "verify", "--profile", "inventory",
                                 "--repo", str(self.repo)], text=True, capture_output=True,
                                check=False, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        data = json.loads(result.stdout)
        self.assertTrue(any(item["name"] == "ruv-fann"
                            for item in json.loads(data["results"][0]["output"])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
