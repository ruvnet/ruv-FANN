"""Independent BF16 fixture and real CLI validation. No PAN2 inference claim.

WSP_TENSOR_BIN selects an exact installed/packed binary; otherwise the binary is
resolved under CARGO_TARGET_DIR (relative to the standalone crate when relative).
Checks remain active under python -O. Every CLI process has a timeout.
"""
import hashlib
import json
import os
import pathlib
import struct
import subprocess
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]


def binary():
    if os.environ.get("WSP_TENSOR_BIN"):
        return pathlib.Path(os.environ["WSP_TENSOR_BIN"]).resolve(strict=True)
    target = pathlib.Path(os.environ.get("CARGO_TARGET_DIR", "target"))
    if not target.is_absolute():
        target = ROOT / target
    return (target / "release/ruv-fann-wsp-tensor").resolve(strict=True)


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(*args, ok=True):
    result = subprocess.run([str(binary()), *map(str, args)], capture_output=True,
                            text=True, timeout=60)
    check((result.returncode == 0) == ok,
          f"unexpected CLI exit {result.returncode}: {result.stderr}")
    return json.loads(result.stdout) if ok else result.stderr


def main():
    with tempfile.TemporaryDirectory() as directory:
        d = pathlib.Path(directory)
        base, candidate, restored = [d / n for n in ("base.safetensors", "candidate.safetensors", "restored.safetensors")]
        header = json.dumps({"weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}).encode()
        header += b" " * (-len(header) % 8)
        original = struct.pack("<Q", len(header)) + header + bytes.fromhex("803e003f")
        base.write_bytes(original)
        info = run("inspect", base)
        base_hash = hashlib.sha256(original).hexdigest()
        check(info["sha256"] == base_hash, "inspect hash differs from independent SHA256")
        patch = d / "patch.json"
        patch.write_text(json.dumps({"schema": "ruv-fann.tensor-patch/1", "base_sha256": base_hash,
                                     "edits": [{"tensor": "weight", "index": 0, "value": 0.75}]}))
        start = time.perf_counter()
        receipt = run("apply", base, candidate, patch)
        expected = original[:-4] + bytes.fromhex("403f003f")
        check(candidate.read_bytes() == expected, "candidate differs outside expected BF16 edit")
        check(receipt["base_sha256"] == base_hash, "receipt baseline hash mismatch")
        check(receipt["candidate_sha256"] == hashlib.sha256(expected).hexdigest(), "receipt candidate hash mismatch")
        check(receipt["schema"] == "ruv-fann.tensor-receipt/1", "receipt schema mismatch")
        check(receipt["behavioral_validation"] == "not_run", "unexpected behavioral claim")
        receipt_path = d / "receipt.json"
        receipt_path.write_text(json.dumps(receipt))
        run("rollback", candidate, restored, receipt_path)
        check(restored.read_bytes() == original == base.read_bytes(), "rollback/base byte mismatch")
        run("apply", base, candidate, patch, ok=False)
        check(candidate.read_bytes() == expected, "failed overwrite changed existing candidate")
        value = struct.unpack("<f", b"\0\0" + candidate.read_bytes()[-4:-2])[0]
        check(value == 0.75, "independent BF16 decoding failed")
        return {"schema": "ruv-fann.tensor-e2e/1", "fixture": "independent BF16 Safetensors",
                "binary": str(binary()), "binary_sha256": hashlib.sha256(binary().read_bytes()).hexdigest(),
                "roundtrip": "passed", "overwrite_rejected": True, "base_sha256": base_hash,
                "candidate_sha256": receipt["candidate_sha256"],
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 3),
                "pan2_inference": "not_run", "behavioral_improvement": "not_established"}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
