#!/usr/bin/env python3
"""Inspect actual batch-export callees with a separately installed WABT wasm2wat.

This is an opcode/call-graph check, not a hardware throughput measurement.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def inspect(binary: Path, tool: str, expected: str, export: str, precision: str) -> dict:
    wat = subprocess.run(
        [tool, "--enable-all", "--no-debug-names", str(binary)],
        check=True, capture_output=True, text=True,
    ).stdout
    exports = {name: int(index) for name, index in re.findall(
        r'\(export "([^"]+)" \(func (\d+)\)\)', wat
    )}
    if export not in exports:
        raise ValueError(f"Missing real batch export: {export}")
    functions = {}
    for match in re.finditer(r"^  \(func \(;(\d+);\)", wat, re.MULTILINE):
        end = wat.find("\n  (", match.end())
        functions[int(match[1])] = wat[match.start(): end if end >= 0 else len(wat)]
    pending = [exports[export]]
    reachable = set()
    instructions = {}
    while pending:
        index = pending.pop()
        if index in reachable:
            continue
        reachable.add(index)
        body = functions.get(index, "")
        pending.extend(int(value) for value in re.findall(r"\b(?:return_call|call)\s+(\d+)\b", body))
        vector_ops = re.findall(r"\b(?:f64x2|f32x4)\.(?:mul|add)\b", body)
        if vector_ops:
            instructions[index] = {op: vector_ops.count(op) for op in sorted(set(vector_ops))}
    found = {op for counts in instructions.values() for op in counts}
    required = {"f64x2.mul", "f64x2.add"}
    if precision == "both":
        required |= {"f32x4.mul", "f32x4.add"}
    if expected == "simd" and not required <= found:
        raise ValueError(f"Batch export does not reach both real SIMD arithmetic operations: {found}")
    if expected == "scalar" and found:
        raise ValueError(f"Ordinary batch export unexpectedly reaches SIMD arithmetic: {found}")
    return {"binary": str(binary), "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "export": export, "export_function": exports[export],
            "reachable_function_count": len(reachable),
            "reachable_vector_arithmetic": instructions, "expected": expected,
            "direct_call_graph_only": True,
            "scope": "Actual exported batch function and transitively direct-called bodies; no inference from a build flag or unrelated SIMD exports."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--wasm2wat", default="wasm2wat")
    parser.add_argument("--expected", choices=["scalar", "simd"], required=True)
    parser.add_argument("--export", default="wasmpreparedbatch_run_into")
    parser.add_argument("--precision", choices=["f64", "both"], default="f64")
    args = parser.parse_args()
    result = inspect(args.binary, args.wasm2wat, args.expected, args.export, args.precision)
    result["wasm2wat_version"] = subprocess.run(
        [args.wasm2wat, "--version"], check=True, capture_output=True, text=True,
    ).stdout.strip()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
