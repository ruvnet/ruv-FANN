#!/usr/bin/env python3
"""Opt-in pinned Rune setup and offline verification of actual ruv-FANN source.

Python3.11+, POSIX, preinstalled Cargo/Rust and explicitly prefetched caches are
required. Nothing here calls a model, promotes a candidate, or edits product code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import tomllib
from typing import Any

from rune_support import (GateError, Invalid, Runner, clean_env, load_json, new_external,
                          plain_file, reject_ancestor_configs, sha, source_files,
                          toolchain, tree_digest, validate_manifest, write_json)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/rune.json"
HEX = re.compile(r"^[a-f0-9]{64}$")
PLAN_KEYS = ("schema", "source_digest", "lock_digest", "toolchain", "target", "policy", "parent")
POLICY_KEYS = ("max_usd", "max_tokens", "max_seconds", "max_memory_mb")


def configuration() -> dict[str, Any]:
    config = load_json(CONFIG_PATH)
    upstream = config.get("upstream", {})
    if config.get("schema") != "ruv-fann.rune-config/1" or \
            upstream.get("repository") != "https://github.com/ruvnet/rune" or \
            upstream.get("package") != "rune-toolchain" or upstream.get("binary") != "rune" or \
            upstream.get("version") != "0.2.0" or \
            not re.fullmatch(r"[a-f0-9]{40}", upstream.get("commit", "")) or \
            not HEX.fullmatch(upstream.get("cargo_lock_sha256", "")):
        raise Invalid("invalid Rune source pin configuration")
    return config


def checkout(value: str) -> Path:
    repo = Path(value).expanduser().resolve(strict=True)
    if not repo.is_dir():
        raise Invalid("--repo must be a directory")
    validate_manifest(repo)
    return repo


def closure(repo: Path, config: dict[str, Any]) -> tuple[dict[str, Path], dict[str, Path]]:
    core = source_files(repo, config)
    template = config["probe_directory"]
    probe = {name: plain_file(repo, f"{template}/{name}")
             for name in ("Cargo.toml", "Cargo.lock", "src/main.rs")}
    manifest = tomllib.loads(probe["Cargo.toml"].read_text())
    package = manifest.get("package", {})
    if package.get("name") != config["product_name"] or package.get("publish") is not False or \
            manifest.get("dependencies", {}).get("ruv-fann", {}).get("path") != "core":
        raise Invalid("probe template must be private and depend on the contained actual core")
    if set(manifest.get("dependencies", {})) != {"ruv-fann"} or \
            set(manifest) - {"package", "dependencies", "workspace", "profile", "bin"} or \
            set(manifest["dependencies"]["ruv-fann"]) - {"path", "version"}:
        raise Invalid("probe must use only its reviewed contained core dependency")
    declared_version = manifest["dependencies"]["ruv-fann"].get("version")
    if declared_version not in (None, "=" + tomllib.loads(core["Cargo.toml"].read_text())["package"]["version"]):
        raise Invalid("probe dependency version must exactly match the selected actual core")
    if package.get("build", False) is not False:
        raise Invalid("probe build scripts are unsupported")
    if manifest.get("workspace", {}) not in ({}, {"members": ["."]}, {"members": []}):
        raise Invalid("probe may not load workspace members outside its actual core dependency")
    for target in manifest.get("bin", []):
        if target.get("name") != config["product_name"] or target.get("path", "src/main.rs") != "src/main.rs":
            raise Invalid("probe has an unexpected executable target")
    def registry_entries(path: Path) -> set[tuple[str, str, str, str]]:
        lock = tomllib.loads(path.read_text())
        entries = set()
        for item in lock.get("package", []):
            source = item.get("source")
            if source is not None:
                if not source.startswith("registry+") or not HEX.fullmatch(item.get("checksum", "")):
                    raise Invalid("only checksum-pinned registry dependencies are supported")
                entries.add((item["name"], item["version"], source, item["checksum"]))
        return entries
    if not registry_entries(probe["Cargo.lock"]).issubset(registry_entries(core["Cargo.lock"])):
        raise Invalid("probe lock introduces registry entries outside the actual core lock")
    return core, probe


def prepare(repo: Path, output: Path, config: dict[str, Any]) -> dict[str, Any]:
    core, probe = closure(repo, config)
    original = tree_digest(core)
    workspace = output / "workspace"
    workspace.mkdir(mode=0o700)
    selected = {**{f"core/{name}": path for name, path in core.items()}, **probe}
    records = []
    for name, path in sorted(selected.items()):
        target = workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        if sha(path) != sha(target):
            raise GateError(f"source changed while copying: {name}")
        records.append({"path": name, "sha256": sha(target), "bytes": target.stat().st_size})
    if tree_digest(source_files(repo, config)) != original:
        raise GateError("live core source changed during snapshot")
    snapshot = {name: workspace / name for name in selected}
    return {"schema": "ruv-fann.rune-evidence/1", "status": "prepared", "upstream": config["upstream"],
            "source": {"repo": str(repo), "workspace": str(workspace), "core_tree_sha256": original,
                       "rune_tree_sha256": tree_digest(snapshot), "core_lock_sha256": sha(core["Cargo.lock"]),
                       "probe_lock_sha256": sha(probe["Cargo.lock"]), "files": records},
            "scope": "Actual unmodified core plus a private executable probe; no other workspace/state is copied.",
            "model_calls": 0}


def reject_untracked_build_inputs(runner: Runner, git: list[str], cwd: Path, phase: str) -> None:
    paths = ["--", "Cargo.toml", "Cargo.lock", "build.rs", "crates"]
    for ignored in (False, True):
        flags = ["ls-files", "--others", "--exclude-standard"] + (["--ignored"] if ignored else [])
        result = runner.run(f"{phase}-untracked-{'ignored' if ignored else 'ordinary'}", git + flags + paths, cwd)
        if result.strip():
            raise Invalid("untracked build inputs (including ignored files) are not part of the pinned Rune source")


def setup(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    source = Path(args.source).expanduser().resolve(strict=True)
    if source.is_relative_to(ROOT):
        raise Invalid("Rune source must be a separate pinned clone outside the checkout")
    reject_ancestor_configs(source)
    output = new_external(args.prefix, ROOT, source)
    report: dict[str, Any] = {"schema": "ruv-fann.rune-tool/1", "status": "failed",
                              "upstream": config["upstream"], "model_calls": 0}
    runner = Runner(output, clean_env(output), args.timeout)
    try:
        git = ["git", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null", "-C", str(source)]
        commit = runner.run("source-commit", git + ["rev-parse", "HEAD"], output).strip()
        status = runner.run("source-status", git + ["status", "--porcelain", "--untracked-files=normal"], output)
        modes = runner.run("source-file-modes", git + ["ls-files", "--stage", "--",
                           "Cargo.toml", "Cargo.lock", "build.rs", "crates"], output)
        if commit != config["upstream"]["commit"] or status.strip() or \
                any(line.startswith("120000 ") for line in modes.splitlines()):
            raise Invalid("Rune clone must have the pinned commit, a clean tree, and no tracked symlinks")
        reject_untracked_build_inputs(runner, git, output, "initial")
        source_inputs = []
        for line in modes.splitlines():
            metadata, name = line.split("\t", 1)
            mode, expected, stage = metadata.split()
            if mode not in ("100644", "100755") or stage != "0" or name.startswith('"'):
                raise Invalid("Rune build inputs must be ordinary, unambiguous tracked files")
            path = plain_file(source, name)
            data = path.read_bytes()
            actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if actual != expected:
                raise Invalid(f"Rune build input bytes differ from pinned Git blob: {name}")
            source_inputs.append({"path": name, "sha256": sha(path), "git_blob": expected})
        if sha(plain_file(source, "Cargo.lock")) != config["upstream"]["cargo_lock_sha256"]:
            raise Invalid("Rune Cargo.lock differs from the pinned source")
        compiler = toolchain(runner, output)
        command = [compiler["cargo"]["path"], "build", "--release", "--locked", "--offline",
                   "--package", "rune-toolchain", "--bin", "rune", "--target-dir", str(output / "target")]
        runner.run("build-pinned-rune", command, source)
        built = plain_file(output, "target/release/rune", 268_435_456)
        (output / "bin").mkdir()
        binary = output / "bin/rune"
        shutil.copyfile(built, binary)
        binary.chmod(0o755)
        version = runner.run("rune-version", [str(binary), "--version"], output).strip()
        if version != "rune " + config["upstream"]["version"]:
            raise GateError("installed executable is not the pinned rune-toolchain version")
        if runner.run("source-final-status", git + ["status", "--porcelain", "--untracked-files=normal"], output).strip() or \
                sha(source / "Cargo.lock") != config["upstream"]["cargo_lock_sha256"]:
            raise GateError("pinned tool sources/lock changed during setup")
        if runner.run("source-final-commit", git + ["rev-parse", "HEAD"], output).strip() != commit or \
                any(sha(plain_file(source, entry["path"])) != entry["sha256"] for entry in source_inputs):
            raise GateError("pinned source commit/bytes changed during setup")
        reject_untracked_build_inputs(runner, git, output, "final")
        report.update(status="passed", binary={"path": "bin/rune", "sha256": sha(binary), "version": version},
                      source={"path": str(source), "commit": commit, "clean": True,
                              "cargo_lock_sha256": sha(source / "Cargo.lock"), "inputs": source_inputs}, toolchain=compiler,
                      build_command=command,
                      trust="Local locked-source build receipt; not a signed upstream binary attestation.")
    except (Invalid, GateError, OSError, ValueError) as error:
        report["error"] = str(error)
        raise
    finally:
        report["commands"] = runner.commands
        write_json(output / "tool-receipt.json", report)
    return {**report, "receipt_path": str(output / "tool-receipt.json")}


def installed_tool(path: str, config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    receipt_path = Path(path).expanduser().absolute()
    if any(item.is_symlink() for item in (receipt_path, *receipt_path.parents)):
        raise Invalid("tool receipt ancestry must not contain symlinks")
    receipt = load_json(plain_file(receipt_path.parent.resolve(), receipt_path.name))
    if receipt.get("schema") != "ruv-fann.rune-tool/1" or receipt.get("status") != "passed" or \
            receipt.get("upstream") != config["upstream"] or receipt.get("model_calls") != 0:
        raise Invalid("tool receipt is not a successful build of the configured source pin")
    source = receipt.get("source", {})
    command = receipt.get("build_command", [])
    if source.get("commit") != config["upstream"]["commit"] or source.get("clean") is not True or \
            source.get("cargo_lock_sha256") != config["upstream"]["cargo_lock_sha256"] or \
            not isinstance(command, list) or len(command) != 11 or \
            command[1:9] != ["build", "--release", "--locked", "--offline", "--package", "rune-toolchain", "--bin", "rune"] or \
            command[9] != "--target-dir":
        raise Invalid("tool receipt source/build fields are inconsistent with the configured pin")
    binary_data = receipt.get("binary", {})
    if binary_data.get("path") != "bin/rune" or not HEX.fullmatch(binary_data.get("sha256", "")):
        raise Invalid("tool receipt has an unexpected executable path or digest")
    binary = plain_file(receipt_path.parent.resolve(), "bin/rune", 268_435_456)
    if sha(binary) != binary_data["sha256"] or not os.access(binary, os.X_OK):
        raise Invalid("installed Rune binary differs from its build receipt")
    return binary, receipt


def inspect_state(workspace: Path, source: dict[str, Any], compiler: dict[str, Any],
                  product_name: str) -> tuple[dict[str, Any], dict[str, Any], Path]:
    history = plain_file(workspace, ".rune/history").read_text().splitlines()
    if len(history) != 1 or not HEX.fullmatch(history[0]):
        raise GateError("expected exactly one frozen Rune plan")
    plan_id = history[0]
    plan = load_json(plain_file(workspace, f".rune/plans/{plan_id}.json"))
    if set(plan) != set(PLAN_KEYS) or set(plan.get("policy", {})) != set(POLICY_KEYS):
        raise GateError("unexpected plan schema/fields")
    canonical = {name: plan[name] for name in PLAN_KEYS}
    canonical["policy"] = {name: plan["policy"][name] for name in POLICY_KEYS}
    identity = hashlib.sha256(json.dumps(canonical, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    if identity != plan_id or plan.get("schema") != "rune.plan/1" or plan.get("parent") is not None:
        raise GateError("plan identity/schema mismatch")
    if plan["source_digest"] != source["rune_tree_sha256"] or \
            plan["lock_digest"] != source["probe_lock_sha256"] or \
            plan["toolchain"] != compiler["rustc"]["identity"]:
        raise GateError("Rune plan does not bind the actual snapshot, lock, and compiler")
    host = next(line[6:] for line in compiler["rustc"]["identity"].splitlines() if line.startswith("host: "))
    if plan["target"] != host:
        raise GateError("Rune plan target differs from the observed compiler host")
    receipt = load_json(plain_file(workspace, f".rune/receipts/{plan_id}.json"))
    if set(receipt) != {"plan_id", "artifact", "artifact_digest", "offline", "model_calls"} or \
            receipt["plan_id"] != plan_id or receipt["artifact"] != product_name or \
            receipt["offline"] is not True or type(receipt["model_calls"]) is not int or \
            receipt["model_calls"] != 0 or not HEX.fullmatch(receipt["artifact_digest"]):
        raise GateError("Rune receipt does not name the expected offline, model-free executable")
    product = plain_file(workspace, f"target/rune/release/{product_name}", 268_435_456)
    stored = plain_file(workspace, f".rune/objects/{receipt['artifact_digest']}", 268_435_456)
    if sha(product) != receipt["artifact_digest"]:
        raise GateError("selected product digest mismatch")
    if sha(stored) != receipt["artifact_digest"]:
        raise GateError("stored object digest mismatch")
    if not os.access(product, os.X_OK):
        raise GateError("selected Rune product is not executable")
    return plan, receipt, product


def verification_output(value: str, repo: Path, config: dict[str, Any]) -> tuple[Path, dict[str, Any] | None]:
    requested = Path(value).expanduser().absolute()
    prepared = None
    if requested.exists():
        if any(item.is_symlink() for item in (requested, *requested.parents)) or requested.resolve().is_relative_to(repo):
            raise Invalid("prepared output must be an ordinary external directory")
        output = requested.resolve()
        reject_ancestor_configs(output)
        entries = set()
        for path in output.iterdir():
            entries.add(path.name)
            if path.name not in {"workspace", "receipt.json"} or len(entries) > 2:
                raise Invalid("prepared output has unexpected entries outside the snapshot")
        if entries != {"workspace", "receipt.json"}:
            raise Invalid("prepared output has unexpected entries outside the snapshot")
        prepared = load_json(output / "receipt.json")
        if prepared.get("status") != "prepared" or prepared.get("upstream") != config["upstream"]:
            raise Invalid("refusing to overwrite a final receipt or arbitrary output directory")
        core, probe = closure(repo, config)
        expected = {**{f"core/{name}": path for name, path in core.items()}, **probe}
        actual = {name: plain_file(output / "workspace", name) for name in expected}
        if tree_digest(expected) != prepared["source"]["rune_tree_sha256"] or \
                tree_digest(actual) != prepared["source"]["rune_tree_sha256"]:
            raise Invalid("prepared snapshot no longer matches the actual selected sources")
        discovered, entries, deadline = set(), 0, time.monotonic() + 10
        for path in (output / "workspace").rglob("*"):
            entries += 1
            if entries > config["limits"]["source_files"] * 4 or time.monotonic() > deadline:
                raise Invalid("prepared snapshot discovery exceeds entry/time budget")
            if path.is_symlink():
                raise Invalid("prepared snapshot contains a symlink")
            if not path.is_file() and not path.is_dir():
                raise Invalid("prepared snapshot contains a special file")
            if path.is_file():
                discovered.add(str(path.relative_to(output / "workspace")))
            if len(discovered) > len(expected):
                raise Invalid("prepared snapshot has unexpected files")
        if discovered != set(expected):
            raise Invalid("prepared snapshot has missing/unexpected files")
        write_json(output / "prepare-receipt.json", prepared)
    else:
        output = new_external(value, repo)
    return output, prepared


def verify(args: argparse.Namespace, repo: Path, config: dict[str, Any]) -> dict[str, Any]:
    binary, tool_receipt = installed_tool(args.tool_receipt, config)
    output, prepared = verification_output(args.output, repo, config)
    report: dict[str, Any] = {"schema": "ruv-fann.rune-evidence/1", "status": "failed",
                              "upstream": config["upstream"], "negative_checks": {}}
    runner = Runner(output, clean_env(output), args.timeout)
    try:
        report.update(prepared or prepare(repo, output, config)); report["status"] = "failed"
        workspace = output / "workspace"
        compiler = toolchain(runner, output)
        report["toolchain"] = compiler
        report["tool"] = {"receipt_sha256": sha(Path(args.tool_receipt)), "binary_sha256": sha(binary),
                          "source_commit": tool_receipt["upstream"]["commit"]}
        rune = [str(binary), "--workspace", str(workspace)]
        runner.run("freeze", rune + ["build", "--freeze"], output)
        runner.run("replay", rune + ["build"], output)
        runner.run("fresh-verify", rune + ["verify"], output)
        runner.run("explain", rune + ["explain"], output)
        plan, receipt, product = inspect_state(workspace, report["source"], compiler, config["product_name"])
        report["rune_plan"], report["rune_receipt"] = plan, receipt
        report["product"] = {"name": product.name, "sha256": sha(product), "stored_object_verified": True}
        probe = runner.run("runtime-probe", [str(product)], output)
        observed = json.loads(probe)
        if not isinstance(observed, dict) or observed.get("status") != "passed":
            raise GateError("digest-bound probe did not report passed")
        report["runtime_probe"] = observed
        command = [compiler["cargo"]["path"], "test", "--release", "--locked", "--offline",
                   "--manifest-path", str(workspace / "core/Cargo.toml"), "--target-dir",
                   str(output / "correctness-target"), "--lib"]
        for name in config["correctness_tests"]:
            command += ["--test", name]
        runner.run("release-source-correctness", command, workspace / "core")
        report["correctness"] = {"status": "passed", "tests": config["correctness_tests"],
                                 "scope": "Release tests of same actual core source; separate from selected product digest."}
        source = workspace / "core/src/lib.rs"
        old = source.read_bytes()
        try:
            source.write_bytes(old + b"\n// Rune adapter stale-source rejection probe.\n")
            runner.run("reject-stale-source", rune + ["build"], output,
                       expected_error="stale plan: source_digest changed")
            report["negative_checks"]["stale_source"] = {"status": "passed", "checked_by": "rune build"}
        finally:
            source.write_bytes(old)
        plan_path = workspace / f".rune/plans/{receipt['plan_id']}.json"
        old = plan_path.read_bytes()
        try:
            bad = dict(plan); bad["source_digest"] = "0" * 64
            write_json(plan_path, bad)
            runner.run("reject-plan-tamper", rune + ["build"], output, expected_error="plan id mismatch")
            report["negative_checks"]["plan_tamper"] = {"status": "passed", "checked_by": "rune build"}
        finally:
            plan_path.write_bytes(old)
        stored = workspace / f".rune/objects/{receipt['artifact_digest']}"
        old = stored.read_bytes()
        try:
            stored.write_bytes(b"tampered Rune object")
            try:
                inspect_state(workspace, report["source"], compiler, config["product_name"])
            except GateError as error:
                if str(error) != "stored object digest mismatch":
                    raise
                report["negative_checks"]["object_tamper"] = {"status": "passed", "checked_by": "adapter SHA-256"}
            else:
                raise GateError("object tampering was accepted")
        finally:
            stored.write_bytes(old)
        inspect_state(workspace, report["source"], compiler, config["product_name"])
        if tree_digest(source_files(repo, config)) != report["source"]["core_tree_sha256"]:
            raise GateError("live core inputs changed during verification")
        actual = {entry["path"]: plain_file(workspace, entry["path"]) for entry in report["source"]["files"]}
        if tree_digest(actual) != report["source"]["rune_tree_sha256"]:
            raise GateError("snapshot inputs changed during verification")
        if sha(binary) != report["tool"]["binary_sha256"]:
            raise GateError("Rune binary changed during verification")
        if any(sha(Path(value["path"])) != value["sha256"] for value in compiler.values()):
            raise GateError("recorded Cargo/rustc binaries changed during verification")
        report.update(status="passed", fresh_verify=True, model_calls=0, offline=True,
                      environment=runner.env, unchanged_core_lock=True,
                      limits={"timeout_seconds": args.timeout, "output_bytes_per_command": runner.limit},
                      exclusions=[".swarm", ".claude", ".claude-flow", "other workspaces", "credentials"],
                      scope="One exact executable reproduces and runs; release-source tests are additional evidence. No speedup or whole-output-graph claim.")
    except (Invalid, GateError, OSError, ValueError, KeyError, TypeError) as error:
        report["error"] = str(error)
        raise GateError(str(error)) from error
    finally:
        report["commands"] = runner.commands
        write_json(output / "receipt.json", report)
    return {"status": report["status"], "receipt_path": str(output / "receipt.json"),
            "product": report["product"], "negative_checks": report["negative_checks"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    for name in ("plan", "prepare", "setup", "verify"):
        item = sub.add_parser(name)
        item.add_argument("--timeout", type=float, default=900)
        if name != "setup":
            item.add_argument("--repo", default=".")
        if name in ("prepare", "verify"):
            item.add_argument("--output", required=True)
        if name == "setup":
            item.add_argument("--source", required=True)
            item.add_argument("--prefix", required=True)
        if name == "verify":
            item.add_argument("--tool-receipt", required=True)
    args = parser.parse_args()
    try:
        if not math.isfinite(args.timeout) or not 0.1 <= args.timeout <= 3600:
            raise Invalid("--timeout must be finite and between 0.1 and 3600 seconds")
        config = configuration()
        if args.operation == "setup":
            result = setup(args, config)
        else:
            repo = checkout(args.repo)
            if args.operation == "plan":
                core, probe = closure(repo, config)
                result = {"status": "planned", "upstream": config["upstream"], "model_calls": 0,
                          "core_files": len(core), "probe_files": len(probe),
                          "core_tree_sha256": tree_digest(core),
                          "steps": ["private snapshot", "freeze", "replay", "fresh verify", "explain",
                                    "digest-bound runtime probe", "release-source tests", "three rejection checks"],
                          "network": "offline; caches must be prefetched explicitly", "source_mutation": False}
            elif args.operation == "prepare":
                output = new_external(args.output, repo)
                result = prepare(repo, output, config)
                write_json(output / "receipt.json", result)
            else:
                result = verify(args, repo, config)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (Invalid, OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "invalid_input", "error": str(error)}))
        return 2
    except GateError as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
