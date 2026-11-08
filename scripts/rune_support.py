"""Bounded process and file primitives for the opt-in Rune adapter (Python3.11)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import stat
import subprocess
import time
import tomllib
from typing import Any


class Invalid(ValueError):
    """Invalid boundary input; no fallback is permitted."""


class GateError(RuntimeError):
    """A requested validation gate did not pass."""


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plain_file(root: Path, relative: str, limit: int = 8_388_608) -> Path:
    if "\\" in relative:
        raise Invalid("literal backslashes in selected filenames are unsupported by the Rune digest contract")
    try:
        relative.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise Invalid("selected filenames must be valid UTF-8") from error
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise Invalid(f"unsafe relative path: {relative}")
    path = root / rel
    current = path
    while current != root:
        if current.is_symlink():
            raise Invalid(f"symlink rejected: {relative}")
        current = current.parent
    try:
        mode = path.stat().st_mode
        if not stat.S_ISREG(mode) or not path.resolve(strict=True).is_relative_to(root):
            raise Invalid(f"not a contained regular file: {relative}")
        if path.stat().st_size > limit:
            raise Invalid(f"file exceeds {limit} bytes: {relative}")
    except OSError as error:
        raise Invalid(f"required file unavailable: {relative}: {error}") from error
    return path


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(plain_file(path.parent, path.name).read_text())
    if not isinstance(value, dict):
        raise Invalid(f"expected JSON object: {path.name}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def reject_ancestor_configs(directory: Path) -> None:
    for parent in (directory, *directory.parents):
        if any((parent / ".cargo" / name).exists() for name in ("config", "config.toml")):
            raise Invalid(f"ambient Cargo config would affect isolated execution: {parent / '.cargo'}")


def new_external(value: str, repo: Path, source: Path | None = None) -> Path:
    original = Path(value).expanduser().absolute()
    for parent in (original, *original.parents):
        if parent.is_symlink():
            raise Invalid(f"output ancestry contains a symlink: {parent}")
    output = original.resolve()
    if output.exists() or output.is_relative_to(repo) or repo.is_relative_to(output):
        raise Invalid("output/prefix must be a new directory outside the checkout")
    if source and (output.is_relative_to(source) or source.is_relative_to(output)):
        raise Invalid("tool prefix must be outside the upstream source clone")
    reject_ancestor_configs(output)
    output.mkdir(parents=True, mode=0o700)
    return output


def clean_env(work: Path) -> dict[str, str]:
    """Omit credentials, wrappers, profile/target overrides and user Cargo config.

    Only cached registry/git directories are shared. Credential/config files in
    the original Cargo home are never copied into this managed Cargo home.
    """
    cache = work / "cargo-home"
    cache.mkdir(mode=0o700)
    original = Path(os.environ.get("CARGO_HOME", str(Path.home() / ".cargo"))).resolve()
    for name in ("registry", "git"):
        target = original / name
        if target.is_dir():
            (cache / name).symlink_to(target, target_is_directory=True)
    temporary = work / "tmp"
    temporary.mkdir(mode=0o700)
    environment = {"PATH": os.environ.get("PATH", os.defpath), "LANG": "C.UTF-8",
                   "CARGO_HOME": str(cache), "CARGO_NET_OFFLINE": "true",
                   "CARGO_BUILD_JOBS": "1", "CARGO_INCREMENTAL": "0",
                   "CARGO_TERM_COLOR": "never", "TMPDIR": str(temporary),
                   "RUSTUP_HOME": os.environ.get("RUSTUP_HOME", str(Path.home() / ".rustup"))}
    selected = os.environ.get("RUSTUP_TOOLCHAIN")
    if selected:
        if any(char in selected for char in ("/", "\\", "\0")) or selected.startswith("-"):
            raise Invalid("RUSTUP_TOOLCHAIN must name an installed toolchain, not a path")
        environment["RUSTUP_TOOLCHAIN"] = selected
    return environment


class Runner:
    def __init__(self, output: Path, env: dict[str, str], timeout: float, limit: int = 262_144):
        self.output, self.env, self.limit = output, env, limit
        self.deadline = time.monotonic() + timeout
        self.commands: list[dict[str, Any]] = []
        (output / "logs").mkdir(exist_ok=True)

    def run(self, label: str, argv: list[str], cwd: Path,
            expected_error: str | None = None) -> str:
        if os.name != "posix":
            raise Invalid("bounded Rune execution requires POSIX")
        started = time.monotonic()
        if started >= self.deadline:
            raise GateError("shared command deadline exceeded")
        process = subprocess.Popen(argv, cwd=cwd, env=self.env, shell=False,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        data = bytearray()
        reason = None
        try:
            with selectors.DefaultSelector() as selector:
                assert process.stdout is not None
                os.set_blocking(process.stdout.fileno(), False)
                selector.register(process.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    remaining = self.deadline - time.monotonic()
                    if remaining <= 0:
                        reason = "timeout"
                        break
                    for key, _ in selector.select(min(remaining, 0.1)):
                        chunk = os.read(key.fileobj.fileno(), min(65_536, self.limit - len(data) + 1))
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        data.extend(chunk)
                        if len(data) > self.limit:
                            reason = "output_limit"
                            break
                    if reason:
                        break
            if not reason:
                try:
                    process.wait(timeout=max(0.001, self.deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    reason = "timeout"
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
            if process.stdout:
                process.stdout.close()
        text = bytes(data[:self.limit]).decode("utf-8", errors="replace")
        logfile = self.output / "logs" / f"{len(self.commands):02d}-{label}.log"
        logfile.write_text(text)
        passed = reason is None and ((process.returncode == 0 and expected_error is None) or
                                     (process.returncode != 0 and expected_error is not None
                                      and expected_error in text))
        self.commands.append({"label": label, "argv": argv, "cwd": str(cwd),
                              "exit_code": process.returncode, "status": "passed" if passed else "failed",
                              "expected_error": expected_error, "reason": reason,
                              "seconds": round(time.monotonic() - started, 3),
                              "log": str(logfile.relative_to(self.output)), "log_sha256": sha(logfile)})
        if not passed:
            raise GateError(f"{label} failed ({reason or process.returncode}); see {logfile}")
        return text


def toolchain(runner: Runner, cwd: Path) -> dict[str, Any]:
    paths = {}
    for name in ("rustc", "cargo"):
        if shutil.which("rustup", path=runner.env["PATH"]):
            resolved = runner.run(f"resolve-{name}", ["rustup", "which", name], cwd).strip()
            path = Path(resolved)
        else:
            found = shutil.which(name, path=runner.env["PATH"])
            if not found:
                raise Invalid(f"{name} must already be installed")
            path = Path(found)
        path = path.resolve(strict=True)
        if not path.is_file() or not os.access(path, os.X_OK):
            raise Invalid(f"{name} is not an executable regular file")
        paths[name] = {"path": str(path), "sha256": sha(path)}
    bindirs = list(dict.fromkeys(str(Path(value["path"]).parent) for value in paths.values()))
    runner.env["PATH"] = os.pathsep.join(bindirs + [runner.env["PATH"]])
    paths["rustc"]["identity"] = runner.run("rustc-identity", [paths["rustc"]["path"], "-vV"], cwd).strip()
    paths["cargo"]["identity"] = runner.run("cargo-identity", [paths["cargo"]["path"], "-V"], cwd).strip()
    if "\nhost: " not in paths["rustc"]["identity"]:
        raise GateError("rustc identity has no host triple")
    return paths


def source_files(repo: Path, config: dict[str, Any]) -> dict[str, Path]:
    """Select only actual core inputs; never recursively copy the repository."""
    files: dict[str, Path] = {}
    total, entries, deadline = 0, 0, time.monotonic() + 10
    def add(name: str) -> None:
        nonlocal total
        path = plain_file(repo, name, config["limits"]["file_bytes"])
        if name not in files:
            total += path.stat().st_size
            files[name] = path
        if len(files) > config["limits"]["source_files"] or total > config["limits"]["source_bytes"]:
            raise Invalid("source snapshot exceeds configured file/byte budget")
    for name in config["source_files"]:
        add(name)
    for directory, extensions in config["source_trees"].items():
        if Path(directory).is_absolute() or ".." in Path(directory).parts:
            raise Invalid("source tree must be below the selected checkout")
        base = repo / directory
        if base.is_symlink() or not base.is_dir():
            raise Invalid(f"missing/symlink source directory: {directory}")
        iterator = base.rglob("*") if directory == "src" else base.iterdir()
        for path in iterator:
            entries += 1
            if entries > config["limits"]["source_files"] * 4 or time.monotonic() > deadline:
                raise Invalid("source discovery exceeds its entry/time budget")
            if path.is_symlink() or (not path.is_file() and not path.is_dir()):
                raise Invalid(f"symlink/special source entry rejected: {path.relative_to(repo)}")
            if path.is_file() and path.suffix in extensions:
                name = str(path.relative_to(repo))
                add(name)
    for name in (".cargo/config", ".cargo/config.toml"):
        if (repo / name).exists() or (repo / name).is_symlink():
            if name.endswith("/config"):
                raise Invalid("legacy .cargo/config is unsupported; explicit .toml lint config required")
            path = plain_file(repo, name)
            data = tomllib.loads(path.read_text())
            if set(data) - {"build", "target"} or data.get("build", {}) != {}:
                raise Invalid("only the root's explicit lint-only Cargo config is supported")
            targets = data.get("target", {})
            if not isinstance(targets, dict):
                raise Invalid("Cargo target configuration must be a table")
            for target in targets.values():
                if not isinstance(target, dict) or set(target) != {"rustflags"}:
                    raise Invalid("Cargo target config must contain only lint flags")
                flags = target["rustflags"]
                if not isinstance(flags, list) or len(flags) % 2:
                    raise Invalid("Cargo lint flags must be level/name pairs")
                for level, lint in zip(flags[::2], flags[1::2]):
                    if level not in ("-A", "-W", "-D", "-F") or not isinstance(lint, str) or \
                            not lint.replace("::", "").replace("_", "").isalnum():
                        raise Invalid("non-lint rustflags are unsupported")
            add(name)
    return dict(sorted(files.items()))


def tree_digest(files: dict[str, Path]) -> str:
    digest = hashlib.sha256()
    # Rust PathBuf::Ord compares components: training/adam.rs precedes
    # training.rs, although a raw slash-separated string sorts them oppositely.
    for relative, path in sorted(files.items(), key=lambda item: Path(item[0]).parts):
        name, data = relative.encode(), path.read_bytes()
        digest.update(len(name).to_bytes(8, "little")); digest.update(name)
        digest.update(len(data).to_bytes(8, "little")); digest.update(data)
    return digest.hexdigest()


def validate_manifest(repo: Path) -> None:
    data = tomllib.loads(plain_file(repo, "Cargo.toml").read_text())
    package = data.get("package")
    if not isinstance(package, dict) or package.get("name") != "ruv-fann":
        raise Invalid("selected repository must be the root ruv-fann package")
    if package.get("build", False) is not False or (repo / "build.rs").exists():
        raise Invalid("unreviewed root build scripts are unsupported")
    def inspect(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "path":
                    if not isinstance(child, str):
                        raise Invalid("manifest path must be a string")
                    plain_file(repo, child)
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)
    inspect(data)
    if data.get("patch") or data.get("replace"):
        raise Invalid("root dependency patches/replacements require a reviewed probe-lock design")
    for section in ("dependencies", "dev-dependencies", "build-dependencies"):
        values = data.get(section, {})
        if not isinstance(values, dict):
            raise Invalid(f"manifest {section} must be a table")
        if any(isinstance(value, dict) and ("path" in value or "git" in value) for value in values.values()):
            raise Invalid("path/git dependencies are outside this root-library adapter contract")
    workspace = data.get("workspace", {})
    if not isinstance(workspace, dict):
        raise Invalid("manifest workspace must be a table")
    members = workspace.get("members", [])
    if members not in ([], ["."]):
        raise Invalid("adapter supports the standalone root workspace only")
