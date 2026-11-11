"""Repeated synthetic file-I/O benchmark; no neural inference/speedup claim."""
import argparse
import hashlib
import json
import pathlib
import platform
import sys
import statistics
import struct
import subprocess
import tempfile
import time

from e2e import binary, check


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sizes-mib", type=int, nargs="+", default=[16, 128])
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args(argv)
    if not 1 <= args.repeats <= 100 or args.timeout < 1 or any(not 1 <= size <= 4096 for size in args.sizes_mib):
        parser.error("require repeats 1..100, sizes 1..4096 MiB and positive timeout")
    executable = binary()
    rows = []
    for mib in args.sizes_mib:
        with tempfile.TemporaryDirectory() as directory:
            d = pathlib.Path(directory)
            base = d / "base"
            size = mib * 1024 * 1024
            header = json.dumps({"weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
                                 "padding": {"dtype": "U8", "shape": [size], "data_offsets": [4, size + 4]}}).encode()
            header += b" " * (-len(header) % 8)
            with base.open("wb") as stream:
                stream.write(struct.pack("<Q", len(header)) + header + bytes.fromhex("803e003f"))
                block = bytes(1024 * 1024)
                for _ in range(mib):
                    stream.write(block)
            with base.open("rb") as stream:
                base_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            patch = d / "patch.json"
            patch.write_text(json.dumps({"schema": "ruv-fann.tensor-patch/1", "base_sha256": base_hash,
                                         "edits": [{"tensor": "weight", "index": 0, "value": 0.75}]}))
            samples = []
            candidate_hash = None
            for i in range(args.repeats):
                out = d / ("candidate-" + str(i))
                start = time.perf_counter()
                result = subprocess.run([str(executable), "apply", str(base), str(out), str(patch)],
                                        check=True, capture_output=True, text=True, timeout=args.timeout)
                elapsed = time.perf_counter() - start
                receipt = json.loads(result.stdout)
                with out.open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                check(receipt["candidate_sha256"] == actual and receipt["base_sha256"] == base_hash,
                      "benchmark output/receipt hash mismatch")
                check(candidate_hash in (None, actual), "repeated candidates differ")
                candidate_hash = actual
                samples.append(elapsed)
                out.unlink()  # Excluded from the next timed call; bound disk use.
            rows.append({"payload_mib": mib, "file_bytes": base.stat().st_size,
                         "raw_seconds": samples, "median_seconds": statistics.median(samples),
                         "min_seconds": min(samples), "max_seconds": max(samples),
                         "median_payload_mib_per_sec": mib / statistics.median(samples),
                         "base_sha256": base_hash, "candidate_sha256": candidate_hash})
    result = {"scope": "synthetic Safetensors file I/O; warm OS cache; includes CLI spawn/copy/hash/edit; no fsync durability or inference-throughput claim",
              "binary": str(executable), "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
              "repeats": args.repeats, "host": platform.platform(), "python": sys.version,
              "temporary_directory": tempfile.gettempdir(), "rows": rows}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
