"""Boundary and real-process tests for Rune integration; no Cargo/model calls."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("rune_adapter", SCRIPTS / "rune.py")
assert SPEC and SPEC.loader
RUNE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNE)
from rune_support import GateError, Invalid, Runner, clean_env, source_files, tree_digest


def write(path: Path, data: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else data)


def fixture(root: Path) -> None:
    write(root / "Cargo.toml", '[package]\nname="ruv-fann"\nversion="0.2.2"\n[workspace]\n')
    write(root / "Cargo.lock", 'version=4\n')
    for name in ("README.md", "LICENSE", "LICENSE-MIT", "LICENSE-APACHE"):
        write(root / name, "Fixture data\n")
    for directory in ("src", "tests", "examples", "benches"):
        write(root / directory / "fixture.rs", "// Real-process boundary fixture, not an E2E substitute.\n")
    write(root / "tests/autograd/validation.rs", "// Nested training regression helper.\n")
    write(root / "src/lib.rs", "pub fn fixture() {}\n")
    write(root / "config/rune-probe/Cargo.toml",
          '[package]\nname="ruv-fann-rune-probe"\nversion="0.0.0"\npublish=false\n'
          '[dependencies]\nruv-fann={path="core",version="=0.2.2"}\n')
    write(root / "config/rune-probe/Cargo.lock", 'version=4\n')
    write(root / "config/rune-probe/src/main.rs", "fn main() {}\n")
    write(root / ".swarm/memory.db", "private swarm state\n")
    write(root / ".claude/settings.json", '{"permissions":{"deny":["Bash(*)"]}}')
    write(root / ".cargo/config.toml", '[build]\n[target.\'cfg(all())\']\nrustflags=["-A","unused_variables"]\n')


class SnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="rune-boundary-")
        self.home = Path(self.temp.name)
        self.repo = self.home / "checkout with spaces; $(touch BAD)"
        fixture(self.repo)
        self.config = RUNE.configuration()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_snapshot_actual_inputs_excludes_private_state(self) -> None:
        output = self.home / "materialized"
        output.mkdir()
        before = (self.repo / "Cargo.lock").read_bytes()
        result = RUNE.prepare(self.repo, output, self.config)
        names = {item["path"] for item in result["source"]["files"]}
        self.assertIn("core/src/lib.rs", names)
        self.assertIn("core/Cargo.lock", names)
        self.assertIn("core/tests/autograd/validation.rs", names)
        self.assertIn("Cargo.lock", names)
        self.assertFalse(any(".swarm" in name or ".claude" in name for name in names))
        self.assertEqual((output / "workspace/core/src/lib.rs").read_bytes(),
                         (self.repo / "src/lib.rs").read_bytes())
        self.assertEqual((self.repo / "Cargo.lock").read_bytes(), before)

    def test_tree_identity_changes_for_source_or_relative_path(self) -> None:
        files = source_files(self.repo, self.config)
        before = tree_digest(files)
        write(self.repo / "src/lib.rs", "pub fn changed() {}\n")
        self.assertNotEqual(tree_digest(files), before)
        unchanged = tree_digest(files)
        moved = {**files, "src/renamed.rs": files["src/lib.rs"]}
        del moved["src/lib.rs"]
        self.assertNotEqual(tree_digest(moved), unchanged)

    def test_nested_autograd_helper_is_bound_to_the_source_digest(self) -> None:
        files = source_files(self.repo, self.config)
        self.assertIn("tests/autograd/validation.rs", files)
        before = tree_digest(files)
        write(self.repo / "tests/autograd/validation.rs", "// Changed validation helper.\n")
        self.assertNotEqual(tree_digest(files), before)

    def test_digest_uses_rust_pathbuf_component_order(self) -> None:
        # Oracle order was executed with Rust std::path::PathBuf::sort:
        # src/lib.rs, src/training/adam.rs, src/training.rs. A lexical string
        # sort places training.rs first and disagrees with Rune's source hash.
        content = {"src/training.rs": b"flat", "src/training/adam.rs": b"nested", "src/lib.rs": b"root"}
        paths = {}
        for name, data in content.items():
            write(self.repo / name, data)
            paths[name] = self.repo / name
        self.assertEqual(tree_digest(paths), "462d11ce4faa60b3349dd23da6d4474c2c08e2b629d8fe0ed77bc5e36f20307f")

    def test_symlink_source_rejected(self) -> None:
        (self.repo / "src/link.rs").symlink_to(self.home / "private.txt")
        with self.assertRaisesRegex(Invalid, "symlink"):
            source_files(self.repo, self.config)

    def test_backslash_source_filename_is_rejected_before_copy(self) -> None:
        write(self.repo / "src" / "training\\child.rs", "// unsupported literal backslash\n")
        with self.assertRaisesRegex(Invalid, "backslashes"):
            source_files(self.repo, self.config)

    def test_non_utf8_source_filename_is_rejected_before_copy(self) -> None:
        raw = os.fsencode(self.repo / "src") + b"/invalid-\xff.rs"
        descriptor = os.open(raw, os.O_WRONLY | os.O_CREAT, 0o600)
        os.close(descriptor)
        with self.assertRaisesRegex(Invalid, "valid UTF-8"):
            source_files(self.repo, self.config)

    def test_special_source_entry_rejected(self) -> None:
        os.mkfifo(self.repo / "src/fifo.rs")
        with self.assertRaisesRegex(Invalid, "special"):
            source_files(self.repo, self.config)

    def test_streaming_source_file_budget(self) -> None:
        config = deepcopy(self.config)
        config["limits"]["source_files"] = 7
        with self.assertRaisesRegex(Invalid, "file/byte budget"):
            source_files(self.repo, config)

    def test_streaming_discovery_budget(self) -> None:
        config = deepcopy(self.config)
        config["limits"]["source_files"] = 16
        for number in range(70):
            write(self.repo / "src" / f"not-input-{number}.txt", "ignored")
        with self.assertRaisesRegex(Invalid, "discovery"):
            source_files(self.repo, config)

    def test_escaping_source_tree_rejected(self) -> None:
        config = deepcopy(self.config)
        config["source_trees"] = {"../outside": [".rs"]}
        with self.assertRaisesRegex(Invalid, "below"):
            source_files(self.repo, config)

    def test_wrapper_config_rejected(self) -> None:
        write(self.repo / ".cargo/config.toml", '[build]\nrustc-wrapper="/outside/steal"\n')
        with self.assertRaisesRegex(Invalid, "lint-only"):
            source_files(self.repo, self.config)

    def test_linker_flags_rejected(self) -> None:
        write(self.repo / ".cargo/config.toml", '[target.\'cfg(all())\']\nrustflags=["-Clinker=/evil"]\n')
        with self.assertRaises(Invalid):
            source_files(self.repo, self.config)

    def test_outside_manifest_path_and_root_patches_rejected(self) -> None:
        original = (self.repo / "Cargo.toml").read_text()
        for extra in ('\n[lib]\npath="../outside.rs"\n',
                      '\n[patch.crates-io]\nthing={version="1"}\n'):
            write(self.repo / "Cargo.toml", original + extra)
            with self.assertRaises(Invalid):
                RUNE.checkout(str(self.repo))

    def test_probe_cannot_disable_core_defaults_or_add_dependency(self) -> None:
        path = self.repo / "config/rune-probe/Cargo.toml"
        original = path.read_text()
        for changed in (original.replace('version="=0.2.2"', 'default-features=false'),
                        original + 'other={git="https://invalid.example/repo"}\n'):
            path.write_text(changed)
            with self.assertRaises(Invalid):
                RUNE.closure(self.repo, self.config)

    def test_probe_lock_cannot_introduce_new_registry_version(self) -> None:
        write(self.repo / "config/rune-probe/Cargo.lock", 'version=4\n[[package]]\nname="outside"\n'
              'version="1.0.0"\nsource="registry+https://github.com/rust-lang/crates.io-index"\n'
              'checksum="' + "0" * 64 + '"\n')
        with self.assertRaisesRegex(Invalid, "outside the actual core lock"):
            RUNE.closure(self.repo, self.config)

    def test_output_directory_cannot_be_checkout_or_existing(self) -> None:
        with self.assertRaises(Invalid):
            RUNE.new_external(str(self.repo / "output"), self.repo)
        with self.assertRaises(Invalid):
            RUNE.new_external(str(self.home), self.repo)

    def test_cargo_config_ancestry_rejected(self) -> None:
        write(self.home / ".cargo/config.toml", '[build]\nrustc-wrapper="/bad"\n')
        with self.assertRaisesRegex(Invalid, "ambient Cargo config"):
            RUNE.new_external(str(self.home / "external"), self.repo)

    def prepared(self) -> Path:
        output = self.home / "prepared"
        output.mkdir()
        report = RUNE.prepare(self.repo, output, self.config)
        RUNE.write_json(output / "receipt.json", report)
        return output

    def test_matching_prepared_snapshot_can_be_reused(self) -> None:
        output = self.prepared()
        selected, report = RUNE.verification_output(str(output), self.repo, self.config)
        self.assertEqual(selected, output)
        self.assertEqual(report["status"], "prepared")
        self.assertTrue((output / "prepare-receipt.json").is_file())

    def test_reuse_cannot_overwrite_final_receipt(self) -> None:
        output = self.prepared()
        RUNE.write_json(output / "receipt.json", {"status": "passed", "upstream": self.config["upstream"]})
        with self.assertRaisesRegex(Invalid, "overwrite a final receipt"):
            RUNE.verification_output(str(output), self.repo, self.config)
        self.assertEqual(RUNE.load_json(output / "receipt.json")["status"], "passed")

    def test_changed_prepared_source_rejected(self) -> None:
        output = self.prepared()
        write(output / "workspace/core/src/lib.rs", "pub fn changed(){}\n")
        with self.assertRaisesRegex(Invalid, "no longer matches"):
            RUNE.verification_output(str(output), self.repo, self.config)

    def test_reuse_rejects_special_files(self) -> None:
        output = self.prepared()
        os.mkfifo(output / "workspace/build.rs")
        with self.assertRaisesRegex(Invalid, "special file"):
            RUNE.verification_output(str(output), self.repo, self.config)

    def test_reuse_rejects_unexpected_external_log_directory(self) -> None:
        output = self.prepared()
        (output / "logs").symlink_to(self.home)
        with self.assertRaisesRegex(Invalid, "unexpected entries"):
            RUNE.verification_output(str(output), self.repo, self.config)

    def test_reuse_has_a_directory_discovery_budget(self) -> None:
        output = self.prepared()
        config = deepcopy(self.config)
        config["limits"]["source_files"] = 20
        for index in range(90):
            (output / "workspace" / f"empty-{index}").mkdir()
        with self.assertRaisesRegex(Invalid, "discovery exceeds"):
            RUNE.verification_output(str(output), self.repo, config)


class ProcessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="rune-process-")
        self.output = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def runner(self, timeout: float = 5, limit: int = 262144) -> Runner:
        return Runner(self.output, clean_env(self.output), timeout, limit)

    def test_credentials_and_tool_overrides_omitted(self) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "never-forward", "OPENAI_API_KEY": "secret",
                                     "RUSTC_WRAPPER": "/bad", "RUSTFLAGS": "-C target-cpu=native",
                                     "CARGO_BUILD_TARGET": "outside", "CARGO_PROFILE_RELEASE_DEBUG": "2"}):
            runner = self.runner()
        output = runner.run("environment", [sys.executable, "-c", "import json,os;print(json.dumps(dict(os.environ)))"], self.output)
        data = json.loads(output)
        for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "RUSTC_WRAPPER", "RUSTFLAGS",
                    "CARGO_BUILD_TARGET", "CARGO_PROFILE_RELEASE_DEBUG", "HOME"):
            self.assertNotIn(key, data)
        self.assertEqual(data["CARGO_NET_OFFLINE"], "true")
        self.assertEqual(data["CARGO_BUILD_JOBS"], "1")

    def test_real_process_failure_preserves_exit_and_log(self) -> None:
        runner = self.runner()
        with self.assertRaises(GateError):
            runner.run("failure", [sys.executable, "-c", "print('failed');raise SystemExit(7)"], self.output)
        self.assertEqual(runner.commands[0]["exit_code"], 7)
        self.assertEqual((self.output / runner.commands[0]["log"]).read_text().strip(), "failed")

    def test_expected_failure_needs_both_nonzero_and_diagnostic(self) -> None:
        runner = self.runner()
        with self.assertRaises(GateError):
            runner.run("false-positive", [sys.executable, "-c", "print('expected')"], self.output,
                       expected_error="expected")

    def test_actual_expected_negative_passes(self) -> None:
        runner = self.runner()
        runner.run("rejection", [sys.executable, "-c", "print('stale source');raise SystemExit(1)"],
                   self.output, expected_error="stale source")
        self.assertEqual(runner.commands[0]["status"], "passed")

    def test_timeout_kills_group(self) -> None:
        runner = self.runner(timeout=0.2)
        started = time.monotonic()
        with self.assertRaises(GateError):
            runner.run("timeout", [sys.executable, "-c", "import time;time.sleep(30)"], self.output)
        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual(runner.commands[0]["reason"], "timeout")

    def test_output_is_bounded_even_for_expected_failure(self) -> None:
        runner = self.runner(limit=1024)
        with self.assertRaises(GateError):
            runner.run("output", [sys.executable, "-c", "print('expected'*10000);raise SystemExit(1)"],
                       self.output, expected_error="expected")
        self.assertEqual(runner.commands[0]["reason"], "output_limit")
        self.assertLessEqual((self.output / runner.commands[0]["log"]).stat().st_size, 1024)

    def test_literal_shell_metacharacters_are_arguments(self) -> None:
        runner = self.runner()
        value = "$(touch BAD); 'literal'"
        output = runner.run("arguments", [sys.executable, "-c", "import sys;print(sys.argv[1])", value], self.output)
        self.assertEqual(output.strip(), value)
        self.assertFalse((self.output / "BAD").exists())

    def test_ignored_untracked_build_script_is_not_a_pinned_source(self) -> None:
        source = self.output / "source"
        source.mkdir()
        subprocess.run(["git", "init", "--quiet", str(source)], check=True, capture_output=True)
        write(source / ".git/info/exclude", "crates/rune-toolchain/build.rs\n")
        write(source / "crates/rune-toolchain/build.rs", "fn main() {}\n")
        runner = self.runner()
        git = ["git", "-c", "core.fsmonitor=false", "-C", str(source)]
        ordinary = subprocess.run(git + ["status", "--porcelain"], text=True, capture_output=True, check=True)
        self.assertEqual(ordinary.stdout.strip(), "")
        with self.assertRaisesRegex(Invalid, "untracked build inputs"):
            RUNE.reject_untracked_build_inputs(runner, git, self.output, "test")


class StateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="rune-state-")
        self.root = Path(self.temp.name)
        self.compiler = {"rustc": {"identity": "rustc fixture\nhost: x86_64-unknown-linux-gnu"}}
        self.source = {"rune_tree_sha256": "1" * 64, "probe_lock_sha256": "2" * 64}
        self.plan = {"schema": "rune.plan/1", "source_digest": "1" * 64, "lock_digest": "2" * 64,
                     "toolchain": self.compiler["rustc"]["identity"], "target": "x86_64-unknown-linux-gnu",
                     "policy": {"max_usd": 2.0, "max_tokens": 200000, "max_seconds": 600, "max_memory_mb": 64},
                     "parent": None}
        self.plan_id = hashlib.sha256(json.dumps(self.plan, separators=(",", ":")).encode()).hexdigest()
        self.plan_path = self.root / f".rune/plans/{self.plan_id}.json"
        write(self.plan_path, json.dumps(self.plan))
        write(self.root / ".rune/history", self.plan_id + "\n")
        self.product_name = "ruv-fann-rune-probe"
        self.product = self.root / f"target/rune/release/{self.product_name}"
        write(self.product, b"fixture bytes for identity checks; this file is never executed")
        self.product.chmod(0o755)
        digest = RUNE.sha(self.product)
        self.object = self.root / f".rune/objects/{digest}"
        write(self.object, self.product.read_bytes())
        self.receipt = {"plan_id": self.plan_id, "artifact": self.product_name, "artifact_digest": digest,
                        "offline": True, "model_calls": 0}
        self.receipt_path = self.root / f".rune/receipts/{self.plan_id}.json"
        write(self.receipt_path, json.dumps(self.receipt))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def inspect(self) -> None:
        RUNE.inspect_state(self.root, self.source, self.compiler, self.product_name)

    def test_consistent_identity_metadata(self) -> None:
        self.inspect()

    def test_tampered_plan_identity_rejected(self) -> None:
        self.plan["source_digest"] = "0" * 64
        write(self.plan_path, json.dumps(self.plan))
        with self.assertRaisesRegex(GateError, "plan identity"):
            self.inspect()

    def test_stale_snapshot_digest_rejected(self) -> None:
        self.source["rune_tree_sha256"] = "0" * 64
        with self.assertRaisesRegex(GateError, "actual snapshot"):
            self.inspect()

    def test_tampered_object_rejected(self) -> None:
        write(self.object, b"bad")
        with self.assertRaisesRegex(GateError, "stored object digest"):
            self.inspect()

    def test_tampered_selected_product_rejected(self) -> None:
        write(self.product, b"bad")
        with self.assertRaisesRegex(GateError, "selected product digest"):
            self.inspect()

    def test_receipt_must_name_exact_plan_offline_and_zero_calls(self) -> None:
        for key, value in (("plan_id", "0" * 64), ("offline", False), ("model_calls", 1),
                           ("model_calls", False), ("artifact", "../../wrong")):
            with self.subTest(key=key, value=value):
                write(self.receipt_path, json.dumps({**self.receipt, key: value}))
                with self.assertRaises(GateError):
                    self.inspect()


if __name__ == "__main__":
    unittest.main(verbosity=2)
