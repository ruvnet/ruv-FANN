"""Exercise release control flow with a stub; never contact a registry."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class ReleaseScripts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        source = Path(__file__).resolve().parents[1]
        for name in ("publish.sh", "publish-simple.sh"):
            shutil.copyfile(source / name, self.root / name)
        self.log = self.root / "cargo-calls"
        cargo = self.root / "cargo"
        cargo.write_text(
            '#!/usr/bin/env bash\n'
            'printf "%s\\n" "$*" >> "$RELEASE_TEST_LOG"\n'
            'if [[ "$*" == *"${RELEASE_TEST_FAIL:-NO_MATCH}"* ]]; then exit 17; fi\n'
        )
        cargo.chmod(0o755)
        self.env = dict(os.environ, PATH=f"{self.root}:{os.environ['PATH']}",
                        RELEASE_TEST_LOG=str(self.log))
        # Former scripts executed .env as shell code, including in dry runs.
        (self.root / ".env").write_text("touch env-was-executed\n")

    def run_script(self, *args, script="publish.sh", failure=None):
        env = dict(self.env)
        if failure:
            env["RELEASE_TEST_FAIL"] = failure
        result = subprocess.run(["bash", str(self.root / script), *args],
                                env=env, text=True, capture_output=True)
        calls = self.log.read_text().splitlines() if self.log.exists() else []
        self.assertFalse((self.root / "env-was-executed").exists())
        self.assertNotIn("login", "\n".join(calls))
        return result, calls

    def test_default_only_validates(self):
        result, calls = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls[:2], ["test --workspace --all-targets --locked",
                                     "check --workspace --exclude opencv-sys --all-features --locked"])
        self.assertEqual(sum(call.startswith("package ") for call in calls), 1)
        self.assertFalse(any(call.startswith("publish ") for call in calls))

    def test_validation_failure_stops_release(self):
        result, calls = self.run_script("--publish", failure="check")
        self.assertEqual(result.returncode, 17)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("successful publication", result.stdout)

    def test_package_failure_stops_before_publish(self):
        result, calls = self.run_script("--publish", failure="package --workspace")
        self.assertEqual(result.returncode, 17)
        self.assertFalse(any(call.startswith("publish ") for call in calls))

    def test_explicit_publish_and_wrapper(self):
        result, calls = self.run_script("--publish", script="publish-simple.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([call for call in calls if call.startswith("publish ")],
                         [f"publish -p {name} --locked" for name in
                          ("opencv-core", "opencv-wasm", "opencv-sdk", "opencv-sys")])
        self.assertIn("check -p opencv-sys --locked", calls)

    def test_invalid_mode_runs_nothing(self):
        result, calls = self.run_script("--anything")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
