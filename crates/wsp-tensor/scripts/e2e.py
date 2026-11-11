"""Independent BF16 fixture and CLI validation. No PAN2 inference claim."""
import hashlib, json, pathlib, struct, subprocess, tempfile, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / "target/release/ruv-fann-wsp-tensor"

def run(*args, ok=True):
    r = subprocess.run([str(BIN), *map(str, args)], capture_output=True, text=True)
    assert (r.returncode == 0) == ok, r.stderr
    return json.loads(r.stdout) if ok else r.stderr

def main():
    with tempfile.TemporaryDirectory() as directory:
        d = pathlib.Path(directory)
        base, candidate, restored = [d / n for n in ("base.safetensors", "candidate.safetensors", "restored.safetensors")]
        # BF16 [0.25, 0.5], generated independently of the Rust writer.
        header = json.dumps({"weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}).encode()
        header += b" " * (-len(header) % 8)
        base.write_bytes(struct.pack("<Q", len(header)) + header + bytes.fromhex("803e003f"))
        original = base.read_bytes()
        info = run("inspect", base)
        patch = d / "patch.json"
        patch.write_text(json.dumps({"schema": "ruv-fann.tensor-patch/1", "base_sha256": info["sha256"], "edits": [{"tensor": "weight", "index": 0, "value": 0.75}]}))
        start = time.perf_counter()
        receipt = run("apply", base, candidate, patch)
        assert candidate.read_bytes()[-4:] == bytes.fromhex("403f003f")
        receipt_path = d / "receipt.json"
        receipt_path.write_text(json.dumps(receipt))
        run("rollback", candidate, restored, receipt_path)
        assert restored.read_bytes() == original == base.read_bytes()
        run("apply", base, candidate, patch, ok=False)
        # Decode the actual edited BF16 bytes independently.
        value = struct.unpack("<f", b"\0\0" + candidate.read_bytes()[-4:-2])[0]
        assert value == 0.75
        return {"schema": "ruv-fann.tensor-e2e/1", "fixture": "independent BF16 Safetensors", "roundtrip": "passed", "overwrite_rejected": True, "base_sha256": hashlib.sha256(original).hexdigest(), "candidate_sha256": receipt["candidate_sha256"], "elapsed_ms": round((time.perf_counter()-start)*1000,3), "pan2_inference": "not_run", "behavioral_improvement": "not_established"}

if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
