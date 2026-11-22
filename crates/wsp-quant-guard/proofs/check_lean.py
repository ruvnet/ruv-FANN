#!/usr/bin/env python3
"""Compile the proof and reject unchecked or custom axiom dependencies."""

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def main():
    lean = sys.argv[1] if len(sys.argv) == 2 else "lean"
    source = Path(__file__).with_name("QuantCells.lean")
    content = source.read_text()
    if re.search(r"\b(?:sorry|admit|unsafe)\b", content):
        raise SystemExit("Unchecked proof token found")
    if re.search(r"^\s*axiom\s", content, re.MULTILINE):
        raise SystemExit("Custom axiom declaration found")
    version = subprocess.run([lean, "--version"], check=True,
                             capture_output=True, text=True).stdout.strip()
    run = subprocess.run([lean, str(source)], check=True,
                         capture_output=True, text=True)
    dependencies = set()
    reported = []
    for line in run.stdout.splitlines():
        match = re.match(r"'([^']+)' depends on axioms: \[([^]]*)\]$", line)
        if match:
            reported.append(match.group(1))
            dependencies.update(x.strip() for x in match.group(2).split(",") if x.strip())
        else:
            match = re.match(r"'([^']+)' does not depend on any axioms$", line)
            if match:
                reported.append(match.group(1))
            else:
                raise SystemExit("Unexpected Lean diagnostic: " + line)
    expected = re.findall(r"^#print axioms ([\w.]+)$", content, re.MULTILINE)
    if reported != expected or not reported:
        raise SystemExit("Missing or reordered axiom audit output")
    allowed = {"propext", "Classical.choice", "Quot.sound"}
    if dependencies - allowed:
        raise SystemExit("Unexpected axioms: " + repr(dependencies - allowed))
    if run.stderr.strip():
        raise SystemExit("Unexpected Lean stderr: " + run.stderr)
    print(json.dumps({
        "status": "passed",
        "lean_version": version,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "checked_theorems": len(reported),
        "theorems": reported,
        "axiom_dependencies": sorted(dependencies),
        "custom_axioms": [],
        "unchecked_placeholders": [],
        "scope": "Integer quantizer lemmas, anchored composition, scale 128 support arithmetic, affine separation, and diamond obstruction. Does not verify Rust code or hardware.",
    }, indent=2))


if __name__ == "__main__":
    main()
