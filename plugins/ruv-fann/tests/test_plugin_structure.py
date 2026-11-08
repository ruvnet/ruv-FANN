#!/usr/bin/env python3
"""Portable package-boundary tests; native Claude host validation is separate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.dont_write_bytecode = True
PLUGIN = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("validate_plugin", PLUGIN / "scripts/validate_plugin.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class PackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="ruv-plugin-package-")
        self.root = Path(self.temp.name) / "fixture"
        self.manifest = self.root / ".claude-plugin/plugin.json"
        write(self.manifest, json.dumps({"name": "fixture", "version": "0.1.0",
                                         "description": "Test plugin"}))
        self.skill = self.root / "skills/example/SKILL.md"
        write(self.skill, '---\nname: example\ndescription: "Inspect a checkout. Use when reviewing a package."\n'
                          '---\nRead [details](references/details.md).\n')
        write(self.root / "skills/example/references/details.md", "A real reference.\n")
        write(self.root / "commands/doctor.md", '---\ndescription: Inspect the checkout.\n---\nRead metadata.\n')
        write(self.root / "agents/reviewer.md", '---\nname: reviewer\ndescription: Review code.\n'
                                               'tools: Read, Grep, Glob\n---\nReview only.\n')

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_portable_valid_package(self) -> None:
        data = MODULE.validate(self.root)
        self.assertEqual(data["status"], "passed")
        self.assertEqual(data["skills"], ["example"])
        self.assertEqual(data["commands"], ["doctor"])
        self.assertEqual(data["agents"], ["reviewer"])

    def test_dangling_reference_fails(self) -> None:
        (self.root / "skills/example/references/details.md").unlink()
        with self.assertRaisesRegex(ValueError, "missing/nonportable"):
            MODULE.validate(self.root)

    def test_reference_escape_fails(self) -> None:
        write(Path(self.temp.name) / "outside.md", "Outside the package\n")
        self.skill.write_text(self.skill.read_text() + "[escape](../../../outside.md)\n")
        with self.assertRaisesRegex(ValueError, "missing/nonportable"):
            MODULE.validate(self.root)

    def test_symlink_rejected(self) -> None:
        (self.root / "linked.md").symlink_to(self.skill)
        with self.assertRaisesRegex(ValueError, "symlink"):
            MODULE.validate(self.root)

    def test_skill_identity_mismatch_rejected(self) -> None:
        self.skill.write_text(self.skill.read_text().replace("name: example", "name: wrong"))
        with self.assertRaisesRegex(ValueError, "name must match"):
            MODULE.validate(self.root)

    def test_automatic_components_rejected(self) -> None:
        data = json.loads(self.manifest.read_text())
        data["hooks"] = {"PostToolUse": []}
        self.manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "automatic/runtime"):
            MODULE.validate(self.root)

    def test_review_agent_tool_escalation_rejected(self) -> None:
        path = self.root / "agents/reviewer.md"
        path.write_text(path.read_text().replace("Read, Grep, Glob", "Read, Grep, Glob, Bash"))
        with self.assertRaisesRegex(ValueError, "only read tools"):
            MODULE.validate(self.root)

    def test_unresolved_plugin_root_resource_rejected(self) -> None:
        self.skill.write_text(self.skill.read_text() + 'Read `${CLAUDE_PLUGIN_ROOT}/scripts/missing.py`.\n')
        with self.assertRaisesRegex(ValueError, "plugin-root resource"):
            MODULE.validate(self.root)

    def test_automatic_shell_expansion_rejected(self) -> None:
        self.skill.write_text(self.skill.read_text() + '!`curl example.invalid`\n')
        with self.assertRaisesRegex(ValueError, "automatic shell"):
            MODULE.validate(self.root)


class ActualPluginTests(unittest.TestCase):
    def test_actual_plugin_component_contract(self) -> None:
        result = MODULE.validate(PLUGIN)
        self.assertEqual(len(result["skills"]), 8)
        self.assertEqual(result["commands"], ["doctor", "release-check", "verify"])
        self.assertEqual(result["agents"], ["repository-reviewer"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
