"""Physical I/O checkpoint benchmark, not neural inference throughput."""
import hashlib, json, pathlib, resource, struct, subprocess, tempfile, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / "target/release/ruv-fann-wsp-tensor"
rows = []
for mib in (16, 128):
    with tempfile.TemporaryDirectory() as directory:
        d = pathlib.Path(directory)
        base, out = d / "base", d / "candidate"
        size = mib * 1024 * 1024
        h = json.dumps({"weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}, "padding":{"dtype":"U8","shape":[size],"data_offsets":[4,size+4]}}).encode()
        h += b" " * (-len(h)%8)
        with base.open("wb") as f:
            f.write(struct.pack("<Q",len(h))+h+bytes.fromhex("803e003f"))
            block = bytes(1024*1024)
            for _ in range(mib): f.write(block)
        digest = hashlib.file_digest(base.open("rb"),"sha256").hexdigest()
        p = d / "patch.json"
        p.write_text(json.dumps({"schema":"ruv-fann.tensor-patch/1","base_sha256":digest,"edits":[{"tensor":"weight","index":0,"value":0.75}]}))
        start = time.perf_counter()
        subprocess.run([str(BIN),"apply",str(base),str(out),str(p)],check=True,stdout=subprocess.DEVNULL)
        elapsed = time.perf_counter()-start
        rows.append({"payload_mib":mib,"seconds":round(elapsed,4),"effective_mib_per_sec":round(mib/elapsed,2),"child_peak_rss_kib_including_spawn_overhead":resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss})
print(json.dumps({"scope":"synthetic Safetensors file I/O; warm OS cache; single sample per size", "rows":rows},indent=2))
