#!/usr/bin/env python3
"""Validate this plugin's portable files and intentionally limited components.

This stdlib structural check complements (and does not replace) the installed
Claude Code host's `claude plugin validate <directory>` command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


NAME = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?$")
MAX_FILES = 256
MAX_BYTES = 8_388_608
FILE_BYTES = 1_048_576


def frontmatter(text: str, location: str) -> dict[str, str]:
    """Read this package's deliberately simple, single-line YAML scalars."""
    lines = text.splitlines()
    if not lines or lines[0] != "---" or "---" not in lines[1:]:
        raise ValueError(f"{location}: missing YAML frontmatter")
    result: dict[str, str] = {}
    for line in lines[1:lines[1:].index("---") + 1]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"([a-z][a-z-]*):\s*(.+)", line)
        if not match or match[1] in result:
            raise ValueError(f"{location}: expected unique single-line scalar metadata")
        value = match[2].strip()
        if value.startswith('"'):
            value = json.loads(value)
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1].replace("''", "'")
        if not isinstance(value, str) or not value or value in ("|", ">"):
            raise ValueError(f"{location}: invalid metadata value")
        result[match[1]] = value
    description = result.get("description", "")
    if not description or len(description) > 1024:
        raise ValueError(f"{location}: description must contain 1–1024 characters")
    return result


def validate(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    files: list[Path] = []
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"symlink is not portable: {path.relative_to(root)}")
        if not path.is_file():
            continue
        files.append(path)
        size = path.stat().st_size
        total += size
        if len(files) > MAX_FILES or size > FILE_BYTES or total > MAX_BYTES:
            raise ValueError("plugin exceeds bounded file count/size budget")
        if any(part in ("node_modules", "__pycache__", ".git") for part in path.relative_to(root).parts):
            raise ValueError(f"unrelated/generated files in plugin: {path.relative_to(root)}")
        if path.name.startswith(".env") or path.suffix in (".pem", ".key"):
            raise ValueError(f"credential-shaped file in plugin: {path.relative_to(root)}")
    manifest = json.loads((root / ".claude-plugin/plugin.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("plugin manifest must be an object")
    name = manifest.get("name", "")
    version = manifest.get("version", "")
    if not isinstance(name, str) or not NAME.fullmatch(name) or len(name) > 64:
        raise ValueError("invalid plugin name")
    if not isinstance(version, str) or not SEMVER.fullmatch(version):
        raise ValueError("plugin version must be semantic versioning")
    if not isinstance(manifest.get("description"), str) or not manifest["description"]:
        raise ValueError("plugin description is missing")
    prohibited = {"hooks", "mcpServers", "lspServers", "settings"} & manifest.keys()
    if prohibited:
        raise ValueError(f"unexpected automatic/runtime components: {sorted(prohibited)}")
    for path in ("hooks", "settings.json", ".mcp.json", ".lsp.json"):
        if (root / path).exists():
            raise ValueError(f"unexpected runtime configuration: {path}")
    skills = sorted((root / "skills").glob("*/SKILL.md"))
    commands = sorted((root / "commands").glob("*.md"))
    agents = sorted((root / "agents").glob("*.md"))
    if not skills or not commands:
        raise ValueError("skills and explicit commands are required")
    for skill in skills:
        data = frontmatter(skill.read_text(encoding="utf-8"), str(skill.relative_to(root)))
        if data.get("name") != skill.parent.name or not NAME.fullmatch(data["name"]):
            raise ValueError(f"skill name must match its kebab-case directory: {skill.parent.name}")
        if len(data["name"]) > 64 or not re.search(r"\buse (?:when|for)\b", data["description"], re.I):
            raise ValueError(f"skill needs a bounded name and use-when trigger: {skill.parent.name}")
    for command in commands:
        data = frontmatter(command.read_text(encoding="utf-8"), str(command.relative_to(root)))
        if "allowed-tools" in data:
            raise ValueError(f"commands must inherit host permissions: {command.name}")
    for agent in agents:
        data = frontmatter(agent.read_text(encoding="utf-8"), str(agent.relative_to(root)))
        if data.get("name") != agent.stem:
            raise ValueError(f"agent name mismatch: {agent.name}")
        if {tool.strip() for tool in data.get("tools", "").split(",")} != {"Read", "Grep", "Glob"}:
            raise ValueError(f"reviewer must have only read tools: {agent.name}")
        if "model" in data or "permissionMode" in data:
            raise ValueError(f"reviewer must inherit host model and permissions: {agent.name}")
    for path in files:
        if path.suffix != ".md":
            continue
        body = path.read_text(encoding="utf-8")
        if len(body.splitlines()) >= 500:
            raise ValueError(f"Markdown file must remain under 500 lines: {path.relative_to(root)}")
        if re.search(r"!`", body):
            raise ValueError(f"unexpected automatic shell expansion: {path.relative_to(root)}")
        for target in re.findall(r"!?\[[^\]]*\]\(([^)\s]+)\)", body):
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
                continue
            destination = (path.parent / target.split("#", 1)[0]).resolve()
            if not destination.is_relative_to(root) or not destination.exists():
                raise ValueError(f"missing/nonportable local link in {path.relative_to(root)}: {target}")
        for target in re.findall(r"\$\{CLAUDE_PLUGIN_ROOT\}/([A-Za-z0-9_./-]+)", body):
            destination = (root / target).resolve()
            if not destination.is_relative_to(root) or not destination.exists():
                raise ValueError(f"missing plugin-root resource in {path.relative_to(root)}: {target}")
    return {"status": "passed", "scope": "structural plugin contract; native host validation is separate",
            "name": name, "version": version, "files": len(files), "bytes": total,
            "skills": [path.parent.name for path in skills],
            "commands": [path.stem for path in commands], "agents": [path.stem for path in agents]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = validate(args.plugin)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
