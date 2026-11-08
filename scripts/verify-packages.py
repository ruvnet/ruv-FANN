#!/usr/bin/env python3
"""Run locked checks across this repository's separate workspaces.

Requires Python 3.11+. This never publishes, updates locks, or masks a failed check.
OpenCV requires system OpenCV/Clang development packages (or native-env.sh).
GPU execution and browser checks have separate explicit release gates.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import tomllib
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
WORKSPACES = (".", "ruv-swarm", "neuro-divergent", "cuda-wasm", "opencv-rust")
REGRESSIONS = {
    ".": [["--test", "inference_allocations"]],
    "ruv-swarm": [
        ["--workspace", "--bins"],
        ["-p", "swe-bench-adapter", "--test", "integration_test"],
    ],
    "cuda-wasm": [["--test", "safety_regressions", "--no-default-features", "--features", "serde"]],
    "opencv-rust": [["-p", "opencv-core", "--test", "matrix_safety"]],
}


def tracked(pattern):
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", pattern], cwd=ROOT,
        check=True, capture_output=True,
    )
    return [ROOT / p.decode() for p in result.stdout.split(b"\0") if p]


def inventory():
    packages = []
    for path in tracked("*Cargo.toml"):
        data = tomllib.loads(path.read_text())
        package = data.get("package")
        if not package:
            continue
        workspace = path.parent if "workspace" in data else next(
            (ROOT / name for name in sorted(WORKSPACES, key=len, reverse=True)
             if path.is_relative_to(ROOT / name)), ROOT,
        )
        parent = tomllib.loads((workspace / "Cargo.toml").read_text())
        version = package["version"]
        if isinstance(version, dict):
            version = parent["workspace"]["package"]["version"]
        packages.append({
            "kind": "cargo", "name": package["name"], "version": version,
            "manifest": str(path.relative_to(ROOT)),
            "workspace": str(workspace.relative_to(ROOT)),
            "publish": package.get("publish", True),
        })
    for path in tracked("*package.json"):
        data = json.loads(path.read_text())
        packages.append({
            "kind": "npm", "name": data.get("name", "(repository tooling)"),
            "version": data.get("version"), "manifest": str(path.relative_to(ROOT)),
            "private": data.get("private", False),
            "role": "fixture" if any(p in ("tests", "test-npm-install-v106", "examples")
                                     for p in path.parts) else "distribution or tooling",
        })
    return packages


def docs_check():
    """Check local root README targets and portable, accessible SVG source."""
    errors = []
    readme = ROOT / "README.md"
    for target in re.findall(r"!?\[[^\]]*\]\(([^)\s]+)\)", readme.read_text()):
        if target.startswith(("http:", "https:", "mailto:", "#")):
            continue
        path = (ROOT / target.split("#")[0]).resolve()
        if not path.exists():
            errors.append(f"README target does not exist: {target}")
    namespace = "{http://www.w3.org/2000/svg}"
    for path in (ROOT / "docs" / "assets").glob("*.svg"):
        try:
            data = path.read_text()
            tree = ET.fromstring(data)
            if tree.tag != namespace + "svg":
                errors.append(f"{path.name}: missing SVG namespace")
            ids = {el.get("id") for el in tree.iter() if el.get("id")}
            labels = tree.get("aria-labelledby", "").split()
            if not labels or any(label not in ids for label in labels):
                errors.append(f"{path.name}: accessible label missing")
            if tree.find(namespace + "title") is None or tree.find(namespace + "desc") is None:
                errors.append(f"{path.name}: title/description missing")
            if "prefers-reduced-motion" not in data:
                errors.append(f"{path.name}: reduced-motion fallback missing")
            for element in tree.iter():
                if element.tag.split("}")[-1] in ("script", "foreignObject"):
                    errors.append(f"{path.name}: active content is not allowed")
                for key, value in element.attrib.items():
                    if key.lower().startswith("on"):
                        errors.append(f"{path.name}: event handler is not allowed")
                    if key.split("}")[-1] == "href" and not value.startswith("#"):
                        errors.append(f"{path.name}: external resource is not allowed")
        except (ET.ParseError, OSError) as error:
            errors.append(f"{path.name}: {error}")
    for error in errors:
        print(error, file=sys.stderr)
    if not errors:
        print("README local links and SVG source checks passed.")
    return not errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--inventory", action="store_true")
    modes.add_argument("--check", action="store_true", help="check libraries, tests, examples and benchmarks")
    modes.add_argument("--test", action="store_true", help="test all default targets and safety regressions")
    modes.add_argument("--audit", action="store_true", help="audit each lock with cargo-audit")
    modes.add_argument("--docs", action="store_true", help="validate root links and SVG source")
    parser.add_argument("--workspace", choices=WORKSPACES, action="append")
    parser.add_argument("--report", type=Path, help="write command exit codes and durations as JSON")
    args = parser.parse_args()
    if args.inventory:
        print(json.dumps(inventory(), indent=2))
        return 0
    if args.docs:
        return 0 if docs_check() else 1
    results = []
    env = os.environ.copy()
    env.setdefault("CARGO_BUILD_JOBS", "2")
    env.setdefault("CARGO_PROFILE_DEV_DEBUG", "0")
    env.setdefault("CARGO_INCREMENTAL", "0")
    for workspace in args.workspace or WORKSPACES:
        if args.audit:
            command = ["cargo", "audit", "--file", "Cargo.lock", "--deny", "unsound"]
        else:
            command = ["cargo", "test" if args.test else "check", "--locked", "--workspace", "--all-targets"]
            if workspace in ("ruv-swarm", "neuro-divergent") or (args.check and workspace == "."):
                command += ["--all-features"]
            if workspace == "cuda-wasm":
                command += ["--no-default-features", "--features", "serde"]
        commands = [command]
        if args.test:
            commands += [["cargo", "test", "--locked"] + flags
                         for flags in REGRESSIONS.get(workspace, [])]
        for command in commands:
            print(f"\n[{workspace}] {shlex.join(command)}", flush=True)
            started = time.monotonic()
            try:
                result = subprocess.run(command, cwd=ROOT / workspace, env=env, check=False)
                code = result.returncode
            except OSError as error:
                print(error, file=sys.stderr)
                code = 127
            results.append({"workspace": workspace, "command": command, "exit_code": code,
                            "seconds": round(time.monotonic() - started, 3)})
    print(json.dumps(results, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2) + "\n")
    return 1 if any(result["exit_code"] for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
