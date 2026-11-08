"""Release-driver integrity checks, including a real offline Cargo staging fixture."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("verify_release", Path(__file__).resolve().parents[1] / "verify-release.py")
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


class ReleaseTests(unittest.TestCase):
    def test_toml_roundtrip_preserves_nested_cargo_tables(self):
        value = {"package": {"name": "a", "include": ["src/**", "Cargo.toml"]}, "bin": [{"name": "a", "path": "src/main.rs"}],
                 "target": {"cfg(unix)": {"dependencies": {"x": {"version": "1", "default-features": False}}}}}
        self.assertEqual(DRIVER.tomllib.loads(DRIVER.dump_toml(value)), value)

    def test_registry_lock_changes_fail_but_exact_verified_root_additions_are_recorded(self):
        original = {("a", "1.0.0", "registry+crates", "abc")}
        extra = ("b", "2.0.0", "registry+crates", "def")
        self.assertEqual(DRIVER.check_external_locks(original, original | {extra}, {extra}), [extra])
        for changed in [set(), {("a", "1.0.1", "registry+crates", "abc")}, original | {extra}]:
            with self.assertRaises(ValueError): DRIVER.check_external_locks(original, changed)

    def test_only_root_dependency_paths_are_redirected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "Cargo.toml"
            manifest.write_text('[dependencies]\nroot={package="ruv-fann",version="0.2.2",path="old"}\nx={version="1",features=["a"]}\n')
            DRIVER.redirect_root(manifest, Path(directory) / "immutable", "ruv-fann", "0.2.2")
            table = DRIVER.tomllib.loads(manifest.read_text())["dependencies"]
            self.assertEqual(table["root"]["path"], "immutable")
            self.assertEqual(table["x"], {"version": "1", "features": ["a"]})
            with self.assertRaises(ValueError): DRIVER.redirect_root(manifest, Path(directory), "ruv-fann", "9.0.0")

    def test_snapshot_rebases_internal_symlinks_and_rejects_external_links(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); repo = base / "repo"; repo.mkdir()
            source = repo / "source.rs"; source.write_text("pub fn value() {}")
            link = repo / "linked.rs"; link.symlink_to(source)
            listing = SimpleNamespace(stdout=b"source.rs\0linked.rs\0")
            with patch.object(DRIVER.subprocess, "run", return_value=listing):
                DRIVER.snapshot(repo, base / "copy", base / "output")
            self.assertEqual((base / "copy/linked.rs").resolve(), base / "copy/source.rs")
            external = base / "external.rs"; external.write_text("outside")
            link.unlink(); link.symlink_to(external)
            with patch.object(DRIVER.subprocess, "run", return_value=listing):
                with self.assertRaises(ValueError): DRIVER.snapshot(repo, base / "rejected", base / "output")

    def test_archive_traversal_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "bad.crate"
            for name in ["root-1.0.0/../../escape", "root-1.0.0/..\\..\\escape", "C:/escape", "root-1.0.0/file:stream"]:
                with tarfile.open(archive, "w:gz") as stream:
                    member = tarfile.TarInfo(name); member.size = 1
                    stream.addfile(member, io.BytesIO(b"x"))
                with self.assertRaises(ValueError): DRIVER.archive_info(archive)
            self.assertFalse((Path(directory) / "escape").exists())

    def test_archive_license_payloads_and_runtime_state_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "candidate.crate"
            for licenses, runtime in [(["LICENSE-MIT"], False), (["LICENSE-MIT", "LICENSE-APACHE"], False), (["LICENSE-MIT", "LICENSE-APACHE"], True)]:
                payload = {"Cargo.toml": b'[package]\nname="candidate"\nversion="1.0.0"\nlicense="MIT OR Apache-2.0"\n'}
                payload.update({name: b"fixture license" for name in licenses})
                if runtime: payload["agent.db"] = b"runtime"
                with tarfile.open(archive, "w:gz") as stream:
                    for name, contents in payload.items():
                        member = tarfile.TarInfo("candidate-1.0.0/" + name); member.size = len(contents)
                        stream.addfile(member, io.BytesIO(contents))
                if len(licenses) == 1 or runtime:
                    with self.assertRaises(ValueError): DRIVER.archive_info(archive)
                else: self.assertEqual(DRIVER.archive_info(archive)["license_files"], sorted(licenses))

    def test_publication_order_includes_renamed_and_target_dependencies(self):
        def row(name, manifest): return {"name": name, "version": "1", "sha256": name, "manifest": manifest}
        roots = [row("root", {}), row("leaf", {"target": {"cfg(unix)": {"build-dependencies": {"alias": {"package": "root", "version": "1"}}}}})]
        self.assertEqual([item["name"] for item in DRIVER.publication_order(roots)], ["root", "leaf"])
        with self.assertRaises(ValueError): DRIVER.publication_order([row("a", {"dependencies": {"b": "1"}}), row("b", {"dependencies": {"a": "1"}})])

    def test_command_failure_and_missing_executable_are_never_verified(self):
        for outcome in [SimpleNamespace(returncode=101), FileNotFoundError("cargo")]:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory); (output / "logs").mkdir(); receipt = {"commands": []}
                with patch.object(DRIVER.subprocess, "run", side_effect=outcome if isinstance(outcome, Exception) else None,
                                  return_value=outcome), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(RuntimeError): DRIVER.execute(["cargo", "package"], output, {}, output, receipt, "failed")
                self.assertIn(receipt["commands"][0]["exit_code"], (101, 127))

    def test_real_offline_archive_staging_has_no_live_lock_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); repo = base / "repo"; repo.mkdir()
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            leaf = repo / "neuro-divergent"; leaf.mkdir()
            for folder, name in [(repo, "mission-release-root-fixture"), (leaf, "mission-release-leaf-fixture")]:
                (folder / "src").mkdir(); (folder / "LICENSE").write_text("MIT fixture license\n")
                manifest = '[package]\nname="' + name + '"\nversion="99.0.0"\nedition="2021"\nlicense="MIT"\ninclude=["src/**","Cargo.toml","LICENSE"]\n[workspace]\n'
                if folder == repo:
                    manifest += 'exclude=["neuro-divergent"]\n'; code = 'pub fn value()->usize { 17 }\n'
                else:
                    manifest += 'members=["helper"]\n[dependencies]\nmission-release-root-fixture={version="99.0.0",path=".."}\nmission-release-helper={version="99.0.0",path="helper"}\n'
                    helper = folder / "helper"; (helper / "src").mkdir(parents=True)
                    (helper / "LICENSE").write_text("MIT fixture license")
                    (helper / "Cargo.toml").write_text('[package]\nname="mission-release-helper"\nversion="99.0.0"\nedition="2021"\nlicense="MIT"\n[dependencies]\nmission-release-root-fixture={version="99.0.0",path="../.."}\n')
                    (helper / "src/lib.rs").write_text('pub fn helper()->usize{mission_release_root_fixture::value()}')
                    code = 'pub fn value()->usize { mission_release_root_fixture::value() }\n'
                (folder / "Cargo.toml").write_text(manifest); (folder / "src/lib.rs").write_text(code)
            env = os.environ | {"CARGO_NET_OFFLINE": "true", "CARGO_INCREMENTAL": "0", "CARGO_PROFILE_DEV_DEBUG": "0"}
            for folder in [repo, leaf]:
                subprocess.run(["cargo", "generate-lockfile", "--manifest-path", str(folder / "Cargo.toml")], env=env, check=True, capture_output=True)
            # These source-like tracked/untracked artifacts must not leak into a snapshot.
            (repo / "agent.db").write_text("ephemeral")
            args = SimpleNamespace(repo=repo, output=base / "output", workspace=["neuro-divergent"], reuse_targets=False, target_root=None)
            with patch.dict(os.environ, env), patch.object(DRIVER, "WORKSPACES", {".": 1, "neuro-divergent": 2}), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(DRIVER.verify(args), 0)
            receipt = json.loads((args.output / "receipt.json").read_text())
            self.assertTrue(receipt["verified"])
            self.assertEqual(len(receipt["archives"]), 3)
            self.assertEqual(receipt["live_lock_sha256_before"], receipt["live_lock_sha256_after"])
            self.assertFalse((args.output / "source/agent.db").exists())
            self.assertTrue(all(command["exit_code"] == 0 for command in receipt["commands"]))
            self.assertTrue(all("--no-verify" not in command["command"] for command in receipt["commands"]))
            for archive in receipt["archives"]:
                self.assertEqual(archive["sha256"], DRIVER.digest(args.output / archive["archive"]))
            self.assertEqual((args.output / "receipt.sha256").read_text().split()[0], DRIVER.digest(args.output / "receipt.json"))


if __name__ == "__main__": unittest.main()
