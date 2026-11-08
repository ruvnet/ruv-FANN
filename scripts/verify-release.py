#!/usr/bin/env python3
"""Build and verify local Cargo release archives without publishing or editing live locks.

Python 3.11+ and modern Cargo with `package --workspace` are required.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
WORKSPACES = {".": 1, "ruv-swarm": 14, "neuro-divergent": 6, "cuda-wasm": 1, "opencv-rust": 4}
ALIASES = {"root": ".", "swarm": "ruv-swarm", "forecasting": "neuro-divergent", "cuda": "cuda-wasm", "opencv": "opencv-rust"}
EPHEMERAL = {".git", "target", "node_modules", "dist", ".hive-mind", ".swarm", ".claude-flow", ".ruflo", ".ruv-swarm", ".claude", ".claude.json", ".mcp.json", ".roo", ".roomodes", "__pycache__", ".pytest_cache"}
ARTIFACT = re.compile(r"(?:\.db(?:-wal|-shm|-journal)?|\.sqlite\d*(?:-wal|-shm|-journal)?|\.crate|\.rmeta|\.rlib|\.pyc)$")


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def dump_toml(data):
    """Write Cargo's TOML value types, preserving semantic tables/array tables."""
    lines = []
    def key(parts):
        return ".".join(json.dumps(part) for part in parts)
    def value(item):
        if isinstance(item, str): return json.dumps(item)
        if isinstance(item, bool): return str(item).lower()
        if isinstance(item, (int, float)): return repr(item)
        if isinstance(item, list): return "[" + ", ".join(value(entry) for entry in item) + "]"
        raise ValueError(f"Unsupported TOML scalar: {type(item).__name__}")
    def table(mapping, prefix=(), header=None):
        if header: lines.extend(["", header])
        for name, item in mapping.items():
            if not isinstance(item, dict) and not (isinstance(item, list) and item and isinstance(item[0], dict)):
                lines.append(f"{json.dumps(name)} = {value(item)}")
        for name, item in mapping.items():
            path = prefix + (name,)
            if isinstance(item, dict): table(item, path, "[" + key(path) + "]")
            elif isinstance(item, list) and item and isinstance(item[0], dict):
                for entry in item: table(entry, path, "[[" + key(path) + "]]")
    table(data)
    return "\n".join(lines) + "\n"


def external_lock(path):
    if not path.exists(): return set()
    return {(row["name"], row["version"], row["source"], row.get("checksum"))
            for row in tomllib.loads(path.read_text()).get("package", []) if row.get("source")}


def check_external_locks(before, after, verified_root=()):
    removed, added = before - after, after - before
    if removed or added - set(verified_root):
        raise ValueError(f"Registry lock identities changed: removed={sorted(removed)}, unapproved additions={sorted(added - set(verified_root))}")
    return sorted(added)


def snapshot(repo, destination, output):
    result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                            cwd=repo, check=True, capture_output=True)
    files, checksum = [], hashlib.sha256()
    for raw in sorted(set(result.stdout.split(b"\0")) - {b""}):
        relative = Path(os.fsdecode(raw))
        source = repo / relative
        if any(part in EPHEMERAL for part in relative.parts) or ARTIFACT.search(relative.name): continue
        if source.is_relative_to(output) or not source.exists(): continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            if not source.resolve().is_relative_to(repo.resolve()):
                raise ValueError(f"External source symlink: {relative}")
            # An absolute in-repository link must not retain a pointer to live
            # sources after copying. Rebase every internal link into the snapshot.
            linked = destination / source.resolve().relative_to(repo.resolve())
            relative_link = os.path.relpath(linked, target.parent)
            target.symlink_to(relative_link)
            identity = "symlink:" + relative_link
        elif source.is_file():
            shutil.copy2(source, target)
            identity = digest(target)
        else: continue
        files.append({"path": relative.as_posix(), "sha256": identity})
        checksum.update(relative.as_posix().encode() + b"\0" + identity.encode() + b"\n")
    return files, checksum.hexdigest()


def archive_info(path):
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            parts = PurePosixPath(member.name).parts
            if not parts or member.name.startswith("/") or "\\" in member.name or ":" in member.name or ".." in parts or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe archive member: {member.name}")
        prefixes = {PurePosixPath(member.name).parts[0] for member in members}
        if len(prefixes) != 1: raise ValueError("Archive must have one package directory")
        prefix = prefixes.pop()
        manifest = tomllib.loads(archive.extractfile(prefix + "/Cargo.toml").read().decode())
        package = manifest["package"]
        if prefix != f"{package['name']}-{package['version']}": raise ValueError("Archive identity mismatch")
        names = {str(PurePosixPath(member.name).relative_to(prefix)) for member in members if member.isfile()}
        forbidden = [name for name in names if any(part in EPHEMERAL for part in PurePosixPath(name).parts) or ARTIFACT.search(PurePosixPath(name).name)]
        if forbidden: raise ValueError(f"Runtime state/build artifacts in archive: {forbidden}")
        license_name = package.get("license", "")
        required = []
        if "MIT" in license_name: required.append(any(name in names for name in ("LICENSE-MIT", "LICENSE", "LICENSE.txt")))
        if "Apache-2.0" in license_name: required.append("LICENSE-APACHE" in names)
        if not required or not all(required): raise ValueError(f"Missing declared license payload for {package['name']}: {license_name}")
        return {"name": package["name"], "version": package["version"], "license": license_name,
                "license_files": sorted(name for name in names if name.startswith("LICENSE")),
                "member_count": len(names), "sha256": digest(path), "size": path.stat().st_size,
                "manifest": manifest}


def extract_root(archive, destination):
    info = archive_info(archive)
    if destination.exists(): raise ValueError("Root archive staging directory must be new")
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            target = destination / member.name
            if member.isdir(): target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(member.mode & 0o777)
    root = destination / f"{info['name']}-{info['version']}"
    if "workspace" in tomllib.loads((root / "Cargo.toml").read_text()):
        raise ValueError("Published root manifest unexpectedly retains a workspace")
    return root


def registry_overlay(root_archive, output, metadata, lock):
    """Use Cargo's supported directory-source replacement without root dev deps.

    Borrow cached package payloads through links, but derive their file checksums
    from the original reviewed .crate bytes. No Cargo-home files are edited.
    """
    vendor = output / "registry-source"
    root_info = archive_info(root_archive)
    root = vendor / f"{root_info['name']}-{root_info['version']}"
    if not root.exists():
        extract_root(root_archive, vendor)
        files = tree_identity(root)
        (root / ".cargo-checksum.json").write_text(json.dumps({"package": root_info["sha256"], "files": files}, sort_keys=True))
    expected = {(name, version, source): checksum for name, version, source, checksum in external_lock(lock)}
    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
    located = {(package["name"], package["version"], package.get("source")): Path(package["manifest_path"]).parent for package in metadata["packages"]}
    for identity, checksum in sorted(expected.items()):
        name, version, registry = identity
        if registry != "registry+https://github.com/rust-lang/crates.io-index": continue
        package = {"name": name, "version": version}
        destination = vendor / f"{package['name']}-{package['version']}"
        if destination.exists():
            recorded = json.loads((destination / ".cargo-checksum.json").read_text())
            if recorded["package"] != checksum: raise ValueError(f"Conflicting vendor checksum: {identity}")
            continue
        archive_candidates = list((cargo_home / "registry/cache").glob(f"*/{destination.name}.crate"))
        archive = next((path for path in archive_candidates if digest(path) == checksum), None)
        if archive is None: raise ValueError(f"Reviewed cache archive is unavailable: {identity}")
        cached = located.get(identity, cargo_home / "registry/src" / archive.parent.name / destination.name)
        files = {}
        with tarfile.open(archive) as stream:
            for member in stream.getmembers():
                if member.isfile():
                    relative = PurePosixPath(member.name).relative_to(destination.name).as_posix()
                    files[relative] = hashlib.sha256(stream.extractfile(member).read()).hexdigest()
        destination.mkdir()
        if cached.is_dir():
            for child in cached.iterdir():
                if child.name != ".cargo-checksum.json": (destination / child.name).symlink_to(child, target_is_directory=child.is_dir())
        else:
            with tarfile.open(archive) as stream:
                for member in stream.getmembers():
                    parts = PurePosixPath(member.name).parts
                    if not parts or parts[0] != destination.name or ".." in parts or "\\" in member.name or ":" in member.name or not (member.isfile() or member.isdir()):
                        raise ValueError(f"Unsafe registry archive member: {member.name}")
                    target = vendor / member.name
                    if member.isdir(): target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with stream.extractfile(member) as source_file, target.open("wb") as output_file: shutil.copyfileobj(source_file, output_file)
        (destination / ".cargo-checksum.json").write_text(json.dumps({"package": checksum, "files": files}, sort_keys=True))
    return root, root_info


def registry_flags(vendor):
    # CLI configuration is inherited by Cargo's verifier but cannot leak into a
    # published .cargo/config.toml inside a candidate archive.
    return ["--config", 'source.crates-io.replace-with="verified-release-candidates"',
            "--config", "source.verified-release-candidates.directory=" + json.dumps(str(vendor))]


def stage_candidate(archive, vendor):
    info = archive_info(archive)
    root = extract_root(archive, vendor / (".stage-" + info["name"]))
    checksum = {"package": info["sha256"], "files": tree_identity(root)}
    (root / ".cargo-checksum.json").write_text(json.dumps(checksum, sort_keys=True))
    destination = vendor / root.name
    if destination.exists(): raise ValueError(f"Candidate source already exists: {destination}")
    root.rename(destination)
    root.parent.rmdir()


def member_order(packages):
    pending = {package["name"]: package for package in packages}
    ordered = []
    while pending:
        ready = sorted(name for name, package in pending.items() if all(
            dependency["name"] not in pending or dependency["name"] == name for dependency in package["dependencies"]))
        if not ready: raise ValueError("Workspace publication dependency cycle")
        for name in ready: ordered.append(pending.pop(name))
    return ordered


def tree_identity(root):
    return {path.relative_to(root).as_posix(): digest(path) for path in root.rglob("*") if path.is_file()}


def redirect_root(manifest, extracted, name, version):
    data = tomllib.loads(manifest.read_text())
    changed = False
    def walk(table):
        nonlocal changed
        for key, item in table.items():
            if key in ("dependencies", "dev-dependencies", "build-dependencies") and isinstance(item, dict):
                for alias, dependency in list(item.items()):
                    actual = dependency.get("package", alias) if isinstance(dependency, dict) else alias
                    if actual != name or (isinstance(dependency, dict) and dependency.get("workspace")): continue
                    if isinstance(dependency, str): dependency = {"version": dependency}
                    if dependency.get("version", "").lstrip("^=") != version:
                        raise ValueError(f"Unexpected root version in {manifest}: {dependency}")
                    if extracted is None: dependency.pop("path", None)
                    else: dependency["path"] = os.path.relpath(extracted, manifest.parent).replace(os.sep, "/")
                    item[alias] = dependency; changed = True
            elif isinstance(item, dict): walk(item)
    walk(data)
    if changed: manifest.write_text(dump_toml(data))
    return changed


def publication_order(archives):
    by_name = {row["name"]: row for row in archives}
    graph = {}
    for name, row in by_name.items():
        dependencies = set()
        def walk(table):
            for kind, values in table.items():
                if kind in ("dependencies", "dev-dependencies", "build-dependencies") and isinstance(values, dict):
                    for alias, spec in values.items():
                        actual = spec.get("package", alias) if isinstance(spec, dict) else alias
                        if actual in by_name and actual != name: dependencies.add(actual)
                elif isinstance(values, dict): walk(values)
        walk(row["manifest"]); graph[name] = dependencies
    result = []
    while graph:
        ready = sorted(name for name, dependencies in graph.items() if not dependencies)
        if not ready: raise ValueError(f"Internal publication dependency cycle: {graph}")
        for name in ready:
            result.append({"name": name, "version": by_name[name]["version"], "archive_sha256": by_name[name]["sha256"]})
            del graph[name]
        for dependencies in graph.values(): dependencies.difference_update(ready)
    return result


def execute(command, cwd, env, output, receipt, label):
    log = output / "logs" / f"{len(receipt['commands']):02d}-{label}.log"
    started = time.time()
    print(f"Running {label}", flush=True)
    with log.open("w") as stream:
        try:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
            code = result.returncode
        except OSError as error:
            stream.write(str(error) + "\n")
            code = 127
    receipt["commands"].append({"command": command, "cwd": str(cwd), "exit_code": code,
                                "seconds": round(time.time() - started, 3), "log": str(log.relative_to(output))})
    if code: raise RuntimeError(f"{label} failed ({code}); see {log}")
    return log


def verify(args):
    repo, output = args.repo.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()): raise ValueError("Use a new empty output directory for each immutable receipt")
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(); (output / "archives").mkdir()
    requested = [ALIASES.get(name, name) for name in (args.workspace or WORKSPACES)]
    if any(name in requested for name in ("ruv-swarm", "neuro-divergent")) and "." not in requested: requested.insert(0, ".")
    selected = [name for name in WORKSPACES if name in requested]
    live_locks = {name: digest(repo / name / "Cargo.lock") for name in selected}
    receipt = {"schema_version": 1, "verified": False, "requested_workspaces": selected,
               "commands": [], "archives": [], "live_lock_sha256_before": live_locks}
    env = os.environ.copy()
    env.update(CARGO_BUILD_JOBS="1", CARGO_PROFILE_DEV_DEBUG="0", CARGO_INCREMENTAL="0")
    source = output / "source"
    try:
        files, identity = snapshot(repo, source, output)
        (output / "snapshot-files.json").write_text(json.dumps(files, indent=2) + "\n")
        receipt["snapshot_sha256"] = identity
        receipt["environment"] = {key: env[key] for key in ("CARGO_BUILD_JOBS", "CARGO_PROFILE_DEV_DEBUG", "CARGO_INCREMENTAL")}
        version_log = execute(["cargo", "--version"], source, env, output, receipt, "cargo-version")
        receipt["cargo_version"] = version_log.read_text().strip()
        version_log = execute(["rustc", "--version"], source, env, output, receipt, "rustc-version")
        receipt["rustc_version"] = version_log.read_text().strip()
        root_archive = None
        for workspace in selected:
            folder = source / workspace
            before = external_lock(folder / "Cargo.lock")
            extracted, immutable, root_info = None, None, None
            configuration = []
            if args.reuse_targets: target = repo / workspace / "target"
            else: target = (args.target_root or output / "targets").resolve() / ("root" if workspace == "." else workspace)
            stage_env = env | {"CARGO_TARGET_DIR": str(target)}
            if workspace in ("ruv-swarm", "neuro-divergent"):
                if root_archive is None: raise ValueError("Root archive must be verified before dependent families")
                execute(["cargo", "fetch", "--locked"], folder, stage_env, output, receipt, f"{workspace}-reviewed-fetch")
                original_log = execute(["cargo", "metadata", "--locked", "--format-version", "1"], folder, stage_env, output, receipt, f"{workspace}-reviewed-metadata")
                original = next(json.loads(line) for line in original_log.read_text().splitlines() if line.startswith('{"packages":'))
                if external_lock(folder / "Cargo.lock") != before: raise ValueError("Reviewed metadata changed the snapshot lock")
                extracted, root_info = registry_overlay(root_archive, output, original, folder / "Cargo.lock")
                immutable = tree_identity(extracted)
                for manifest in folder.rglob("Cargo.toml"):
                    redirect_root(manifest, None, root_info["name"], root_info["version"])
                configuration = registry_flags(output / "registry-source")
            metadata_log = execute(["cargo", *configuration, "metadata", "--format-version", "1"], folder, stage_env, output, receipt, f"{workspace.replace('/', '-')}-metadata")
            # Cargo progress goes to stderr; parse its sole JSON object line.
            metadata = next(json.loads(line) for line in metadata_log.read_text().splitlines() if line.startswith('{"packages":'))
            after = external_lock(folder / "Cargo.lock")
            approved = {(root_info["name"], root_info["version"], "registry+https://github.com/rust-lang/crates.io-index", root_info["sha256"])} if extracted else set()
            added = check_external_locks(before, after, approved)
            members = [package for package in metadata["packages"] if package["id"] in metadata["workspace_members"]]
            originals = members
            if len(originals) != WORKSPACES[workspace]: raise ValueError(f"Unexpected package inventory for {workspace}: {len(originals)}")
            if not extracted:
                execute(["cargo", "package", "--workspace", "--allow-dirty", "--locked"], folder, stage_env, output, receipt, f"{workspace.replace('/', '-')}-package")
            receipt.setdefault("lock_normalization", []).append({"workspace": workspace, "verified_root_registry_identity": added})
            for package in member_order(originals):
                if extracted:
                    execute(["cargo", *configuration, "package", "--package", package["name"], "--allow-dirty", "--locked"], folder, stage_env, output, receipt, f"{package['name']}-package")
                path = target / "package" / f"{package['name']}-{package['version']}.crate"
                info = archive_info(path)
                if (info["name"], info["version"]) != (package["name"], package["version"]):
                    raise ValueError(f"Archive does not match selected package: {path}")
                destination = output / "archives" / path.name
                if destination.exists(): raise ValueError(f"Duplicate candidate archive: {destination.name}")
                shutil.copy2(path, destination); destination.chmod(0o444)
                info.update(workspace=workspace, archive=str(destination.relative_to(output)), verified=True)
                receipt["archives"].append(info)
                if workspace == ".": root_archive = destination
                if extracted: stage_candidate(destination, output / "registry-source")
            if extracted and tree_identity(extracted) != immutable: raise ValueError("Verified root archive extraction was mutated")
        order = publication_order(receipt["archives"])
        (output / "publication-order.json").write_text(json.dumps(order, indent=2) + "\n")
        if len(receipt["archives"]) != sum(WORKSPACES[name] for name in selected): raise ValueError("Candidate archive count mismatch")
        receipt["verified"] = True
    finally:
        receipt["live_lock_sha256_after"] = {name: digest(repo / name / "Cargo.lock") for name in selected}
        if receipt["live_lock_sha256_after"] != live_locks:
            receipt["verified"] = False
            receipt["live_lock_error"] = "Live checkout lock changed during isolated verification"
        report = output / "receipt.json"
        report.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        (output / "receipt.sha256").write_text(digest(report) + "  receipt.json\n")
        for path in [report, output / "receipt.sha256", output / "publication-order.json"]:
            if path.exists(): path.chmod(0o444)
    return 0 if receipt["verified"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new empty directory for archives, source snapshot, logs, and immutable receipt")
    parser.add_argument("--workspace", choices=[*WORKSPACES, *ALIASES], action="append", help="repeat to select families; dependent families automatically verify root first")
    parser.add_argument("--repo", type=Path, default=ROOT, help=argparse.SUPPRESS)
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument("--reuse-targets", action="store_true", help="reuse each live workspace's target cache; run only after other Cargo jobs stop")
    targets.add_argument("--target-root", type=Path, help="alternate shared cache parent")
    args = parser.parse_args(argv)
    try: return verify(args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, StopIteration, tarfile.TarError) as error:
        print(f"Release verification failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
