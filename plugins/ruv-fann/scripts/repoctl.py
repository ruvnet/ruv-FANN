#!/usr/bin/env python3
"""Inspect or run explicit, bounded, offline ruv-FANN repository checks.

Requires Python 3.11+. This is a command runner, not a sandbox: execute checks
only in a checkout you trust. Doctor and plan never execute repository code.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from typing import Any


WORKSPACES = (".", "ruv-swarm", "neuro-divergent", "cuda-wasm", "opencv-rust")
PROFILES = {"inventory": None, "docs": None, "core": ".", "swarm": "ruv-swarm",
            "forecasting": "neuro-divergent", "cuda": "cuda-wasm", "opencv": "opencv-rust"}
OUTPUT_LIMIT = 262_144
FILE_LIMIT = 1_048_576
OFFLINE_ENV = {"CARGO_NET_OFFLINE": "true", "GIT_OPTIONAL_LOCKS": "0",
               "PYTHONDONTWRITEBYTECODE": "1"}


class InputError(ValueError):
    """An invalid or unsupported repository/option was supplied."""


class JsonParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print(json.dumps({"status": "invalid_input", "error": message}))
        self.exit(2)


def table(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InputError(f"expected TOML table: {label}")
    return value


def contained_file(root: Path, relative: str) -> Path:
    """Require a regular non-symlink file below root, including its parents."""
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise InputError(f"unsafe repository path: {relative}")
    path = root / rel
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise InputError(f"repository file escapes root or is not regular: {relative}")
        current = path
        while current != root:
            if current.is_symlink():
                raise InputError(f"repository symlink is not accepted: {relative}")
            current = current.parent
        if resolved.stat().st_size > FILE_LIMIT:
            raise InputError(f"repository file exceeds {FILE_LIMIT} bytes: {relative}")
    except (OSError, RuntimeError) as error:
        raise InputError(f"cannot read repository file {relative}: {error}") from error
    return resolved


def read_toml(root: Path, relative: str) -> dict[str, Any]:
    try:
        return tomllib.loads(contained_file(root, relative).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise InputError(f"invalid TOML in {relative}: {error}") from error


def repository(value: str) -> Path:
    try:
        root = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise InputError(f"invalid repository path: {error}") from error
    if not root.is_dir():
        raise InputError("--repo must name a directory")
    package = table(read_toml(root, "Cargo.toml").get("package", {}), "package")
    if package.get("name") != "ruv-fann":
        raise InputError("--repo is not a ruv-fann checkout (Cargo package name mismatch)")
    contained_file(root, "scripts/verify-packages.py")
    contained_file(root, "README.md")
    return root


def doctor(root: Path) -> dict[str, Any]:
    packages: list[dict[str, Any]] = []
    for workspace in WORKSPACES:
        prefix = "" if workspace == "." else workspace + "/"
        top = read_toml(root, prefix + "Cargo.toml")
        workspace_data = table(top.get("workspace", {}), f"{workspace}.workspace")
        members = workspace_data.get("members", ["."])
        if not isinstance(members, list) or len(members) > 128:
            raise InputError(f"invalid workspace members: {workspace}")
        if "package" in top and "." not in members:
            members = [".", *members]
        for member in members:
            if not isinstance(member, str) or any(char in member for char in "*?["):
                raise InputError(f"unsupported workspace member: {member!r}")
            manifest = prefix + ("" if member == "." else member + "/") + "Cargo.toml"
            data = read_toml(root, manifest)
            package = data.get("package")
            if not isinstance(package, dict):
                raise InputError(f"workspace member has no package: {manifest}")
            version = package.get("version")
            if isinstance(version, dict) and version.get("workspace") is True:
                version = table(workspace_data.get("package", {}), "workspace.package").get("version")
            if not isinstance(package.get("name"), str) or not isinstance(version, str):
                raise InputError(f"package name/version missing: {manifest}")
            packages.append({"kind": "cargo", "name": package["name"], "version": version,
                             "manifest": manifest, "workspace": workspace})
    for manifest in ("package.json", "ruv-swarm/npm/package.json", "cuda-wasm/package.json"):
        try:
            data = json.loads(contained_file(root, manifest).read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("expected JSON object")
            packages.append({"kind": "npm", "name": data.get("name"),
                             "version": data.get("version"), "manifest": manifest,
                             "private": data.get("private", False), "engines": data.get("engines", {})})
        except (OSError, UnicodeError, ValueError) as error:
            raise InputError(f"invalid package JSON in {manifest}: {error}") from error
    return {"status": "passed", "operation": "doctor", "repo": str(root), "read_only": True,
            "python": sys.version.split()[0], "packages": packages,
            "tools": {name: shutil.which(name) for name in
                      ("git", "cargo", "rustc", "node", "npm", "claude", "wasm-pack", "pkg-config")},
            "notes": ["Tool paths show discovery only; no tool was executed.",
                      "Historical receipts are not fresh validation of this checkout.",
                      "GPU/browser/native OpenCV checks have additional prerequisites and separate gates."]}


def plan(root: Path, profile: str) -> dict[str, Any]:
    script = str(contained_file(root, "scripts/verify-packages.py"))
    base = [sys.executable, "-B", script]
    if profile in ("inventory", "docs"):
        commands = [base + ["--" + profile]]
    else:
        workspace = PROFILES[profile]
        read_toml(root, ("" if workspace == "." else workspace + "/") + "Cargo.toml")
        commands = [base + [mode, "--workspace", workspace] for mode in ("--check", "--test")]
    return {"status": "planned", "operation": "plan", "repo": str(root), "profile": profile,
            "read_only": True, "commands": [{"argv": argv, "cwd": str(root)} for argv in commands],
            "environment": OFFLINE_ENV.copy(),
            "effects": "inventory/docs read files; code profiles build/test and write Cargo target directories",
            "limits": {"network": "Cargo offline; no installation, audit fetch, or publication",
                       "output_bytes_per_command": OUTPUT_LIMIT},
            "trust": "Commands execute checkout code and build scripts; this runner is not a sandbox."}


def stop_process(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def execute(argv: list[str], cwd: str, deadline: float,
            output_limit: int = OUTPUT_LIMIT) -> dict[str, Any]:
    """Capture bounded bytes and kill the entire POSIX process group on failure."""
    started = time.monotonic()
    result: dict[str, Any] = {"argv": argv, "cwd": cwd}
    if os.name != "posix":
        return {**result, "status": "failed", "reason": "bounded execution requires POSIX",
                "exit_code": None, "seconds": 0, "output": ""}
    environment = {**os.environ, **OFFLINE_ENV}
    try:
        process = subprocess.Popen(argv, cwd=cwd, env=environment, shell=False,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    except OSError as error:
        return {**result, "status": "failed", "reason": "startup", "exit_code": None,
                "seconds": round(time.monotonic() - started, 3), "output": str(error)}
    chunks = bytearray()
    reason = None
    try:
        with selectors.DefaultSelector() as selector:
            assert process.stdout is not None
            os.set_blocking(process.stdout.fileno(), False)
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    reason = "timeout"
                    break
                for key, _ in selector.select(min(remaining, 0.1)):
                    block = os.read(key.fileobj.fileno(), min(65_536, output_limit - len(chunks) + 1))
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    chunks.extend(block)
                    if len(chunks) > output_limit:
                        reason = "output_limit"
                        break
                if reason:
                    break
        if reason:
            stop_process(process)
        else:
            try:
                process.wait(timeout=max(0.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                reason = "timeout"
                stop_process(process)
        # A check must not leave detached children in its process group behind.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    except BaseException:
        stop_process(process)
        raise
    finally:
        if process.stdout:
            process.stdout.close()
    return {**result, "status": "passed" if reason is None and process.returncode == 0 else "failed",
            "reason": reason or (None if process.returncode == 0 else "nonzero_exit"),
            "exit_code": process.returncode, "seconds": round(time.monotonic() - started, 3),
            "output": bytes(chunks[:output_limit]).decode("utf-8", errors="replace"),
            "output_truncated": reason == "output_limit"}


def verify(root: Path, profile: str, timeout: float) -> dict[str, Any]:
    details = plan(root, profile)
    deadline = time.monotonic() + timeout
    results = []
    for command in details["commands"]:
        if time.monotonic() >= deadline:
            results.append({**command, "status": "failed", "reason": "timeout", "exit_code": None})
            break
        result = execute(command["argv"], command["cwd"], deadline)
        results.append(result)
        if result["status"] != "passed":
            break
    return {"status": "passed" if all(x["status"] == "passed" for x in results) else "failed",
            "operation": "verify", "repo": str(root), "profile": profile, "timeout_seconds": timeout,
            "environment": details["environment"], "results": results,
            "scope": "Only the listed commands ran. This is not a release, GPU, or full-repository proof."}


def main() -> int:
    parser = JsonParser(description=__doc__)
    parser.add_argument("operation", choices=("doctor", "plan", "verify"))
    parser.add_argument("--repo", default=".", help="trusted checkout root (default current directory)")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="inventory")
    parser.add_argument("--timeout", type=float, default=120, help="shared deadline, 0.1–1800 seconds")
    args = parser.parse_args()
    try:
        if not 0.1 <= args.timeout <= 1800:
            raise InputError("--timeout must be finite and between 0.1 and 1800 seconds")
        root = repository(args.repo)
        if args.operation == "doctor":
            result = doctor(root)
        elif args.operation == "plan":
            result = plan(root, args.profile)
        else:
            result = verify(root, args.profile, args.timeout)
        encoded = json.dumps(result, indent=2, allow_nan=False)
        # Child output is separately bounded; metadata-only reports need a cap too.
        if args.operation != "verify" and len(encoded.encode("utf-8")) > OUTPUT_LIMIT:
            raise InputError(f"metadata report exceeds {OUTPUT_LIMIT} bytes")
        print(encoded)
        return 0 if result["status"] in ("passed", "planned") else 1
    except (InputError, OSError, ValueError) as error:
        print(json.dumps({"status": "invalid_input", "operation": args.operation, "error": str(error)}))
        return 2
    except KeyboardInterrupt:
        print(json.dumps({"status": "interrupted", "operation": args.operation}))
        return 130


if __name__ == "__main__":
    sys.exit(main())
