"""Paired PAN2 generation on a separately provisioned, trusted Linux CUDA host.

Requires native directories named transformer, a pinned clean PAN2 checkout and
--hf-home containing offline hub/assets directories only (no credentials). Hashes bind bytes, not authority;
this is not a sandbox, physics score, or inference speed benchmark.
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import signal
import stat
import platform
import sys
import subprocess
import time

PIN = "0fb907b6e16a82ead6d691d1de2eef6299c71e9a"
MAX_JSON = 1024 * 1024


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def regular(path):
    if not stat.S_ISREG(path.lstat().st_mode):
        raise RuntimeError(f"requires a regular, non-symlink file: {path}")
    return path


def json_snapshot(path):
    regular(path)
    with path.open("rb") as stream:
        data = stream.read(MAX_JSON + 1)
    if len(data) > MAX_JSON:
        raise RuntimeError("JSON input exceeds 1 MiB")
    return json.loads(data), data


def read_json(path):
    return json_snapshot(path)[0]


def inventory(root, *, cache=False):
    """Hash all files, including config/index files; never read devices/FIFOs.

    HF snapshots legitimately link into their own blobs directory. Those links
    are recorded and must stay inside this explicitly selected cache root.
    """
    if not root.is_dir():
        raise RuntimeError(f"missing directory: {root}")
    result = {}
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        name = str(path.relative_to(root))
        if stat.S_ISLNK(mode):
            target = path.resolve(strict=True)
            if not cache or not target.is_relative_to(root) or not target.is_file():
                raise RuntimeError(f"unsafe symlink: {path}")
            result[name] = {"sha256": digest(target), "link": os.readlink(path)}
        elif stat.S_ISREG(mode):
            result[name] = {"sha256": digest(path)}
        elif not stat.S_ISDIR(mode):
            raise RuntimeError(f"special file is not an evaluation input: {path}")
    if not result:
        raise RuntimeError(f"empty input directory: {root}")
    return result


def model_cache(root):
    # HF_HOME also commonly contains authentication tokens. Require a dedicated
    # offline data-only cache; never open or emit hashes of credential files.
    if not root.is_dir() or any(path.name not in {"hub", "assets"} or not path.is_dir()
                                or path.is_symlink() for path in root.iterdir()):
        raise RuntimeError("--hf-home requires data-only hub/assets directories; no credentials")
    return inventory(root, cache=True)


def checkpoint(root):
    # Pinned generate.py uses <argument>/transformer unless basename=transformer.
    # Restrict this harness to one unambiguous native Safetensors layout. Its
    # fallback loader otherwise accepts .pt/.bin or DCP after a shard load fails.
    if root.name != "transformer":
        raise RuntimeError("native checkpoint directory must be named transformer")
    files = inventory(root)
    shards = [name for name in files if name.endswith(".safetensors")]
    if not shards or any("/" in name for name in shards):
        raise RuntimeError("requires top-level Safetensors shards")
    if any(pathlib.Path(name).suffix in {".pt", ".bin"} or pathlib.Path(name).name in
           {".metadata", "metadata", "_metadata", ".metadata.tmp"} for name in files):
        raise RuntimeError("alternate checkpoint loader formats are not allowed")
    if (root.parent / "model_state.pt").exists():
        raise RuntimeError("parent model_state.pt could bypass the selected shards")
    return files


def verify_receipt(receipt, baseline, candidate, base_hash, candidate_hash):
    if (not isinstance(receipt, dict) or receipt.get("schema") != "ruv-fann.tensor-receipt/1"
            or receipt.get("base_sha256") != base_hash
            or receipt.get("candidate_sha256") != candidate_hash):
        raise RuntimeError("receipt does not match checkpoint pair")
    edits = receipt.get("edits")
    if not isinstance(edits, list) or not 1 <= len(edits) <= 128:
        raise RuntimeError("receipt requires 1..128 undo edits")
    if baseline.stat().st_size != candidate.stat().st_size:
        raise RuntimeError("edited shard size changed")
    undo = []
    for edit in edits:
        if not isinstance(edit, dict):
            raise RuntimeError("invalid undo edit")
        offset, before, after = (edit.get(k) for k in ("offset", "before", "after"))
        if (type(offset) is not int or offset < 8 or not isinstance(before, list)
                or not isinstance(after, list) or len(before) not in (2, 4)
                or len(before) != len(after) or before == after
                or any(type(x) is not int or not 0 <= x <= 255 for x in before + after)):
            raise RuntimeError("invalid undo edit")
        undo.append((offset, bytes(before), bytes(after)))
    undo.sort()
    restored = hashlib.sha256()
    with candidate.open("rb") as stream:
        header = stream.read(8)
        if len(header) != 8:
            raise RuntimeError("truncated shard")
        data_start = 8 + int.from_bytes(header, "little")
        stream.seek(0)
        position = 0
        for offset, before, after in undo:
            if offset < max(position, data_start) or offset + len(after) > candidate.stat().st_size:
                raise RuntimeError("overlapping or out-of-payload undo edit")
            remaining = offset - position
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise RuntimeError("truncated shard")
                restored.update(chunk)
                remaining -= len(chunk)
            if stream.read(len(after)) != after:
                raise RuntimeError("receipt edited bytes do not match candidate")
            restored.update(before)
            position = offset + len(after)
        while chunk := stream.read(1024 * 1024):
            restored.update(chunk)
    if restored.hexdigest() != base_hash:
        raise RuntimeError("receipt undo does not restore baseline")


def prompt_images(prompts, repo):
    """Resolve aliases as pinned inference.inputs does (relative to repo cwd)."""
    if isinstance(prompts, dict):
        keys = [key for key in ("entries", "prompts", "data") if key in prompts]
        if len(keys) != 1:
            raise RuntimeError("prompt wrapper requires one entry list")
        prompts = prompts[keys[0]]
    if not isinstance(prompts, list) or not prompts:
        raise RuntimeError("prompts require a nonempty entry list")
    images = {}
    for entry in prompts:
        if not isinstance(entry, dict):
            raise RuntimeError("prompt entries must be objects")
        for key in ("image_path", "image", "ref_image"):
            if entry.get(key):
                if not isinstance(entry[key], str):
                    raise RuntimeError("image reference must be a local path")
                path = pathlib.Path(entry[key])
                if not path.is_absolute():
                    path = repo / path
                regular(path)
                images[str(path.resolve(strict=True))] = digest(path)
    return images


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, timeout=30).strip()


def source_flags(repo):
    # git status/diff trust assume-unchanged and skip-worktree bits. A modified
    # launcher with either bit is not proof of the pinned executable checkout.
    flags = git(repo, "ls-files", "-v", "-z")
    if any(entry and not entry.startswith("H ") for entry in flags.split("\0")):
        raise RuntimeError("PAN2 index must not hide files with assume-unchanged/skip-worktree")
    return flags


def run_process(command, *, cwd, env, log, timeout):
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
        if code:
            raise RuntimeError(f"inference failed: exit {code}")
    finally:
        # Even a successful parent may leave workers alive. Kill the whole
        # session before hashing inputs or starting the other checkpoint.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def environment(args, label):
    env = os.environ.copy()
    for key in list(env):
        if (key in {"BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS", "CDPATH", "GLOBIGNORE",
                    "PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH",
                    "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH", "HF_STORED_TOKENS_PATH"}
                or key.startswith("BASH_FUNC_")):
            env.pop(key)
    env.update({"PAN_ROOT": str(args.repo), "SEED": str(args.seed), "NUM_GPUS": "1",
                "NUM_STEPS": "50", "FLOW_SHIFT": "9", "GUIDANCE_SCALE": "2.5",
                "VIDEO_HEIGHT": "704", "VIDEO_WIDTH": "1248", "FPS": "24",
                "VIDEO_LENGTH": "241", "TASK": "auto", "EXTRA_ARGS": "",
                "CONDA_ENV": "pan", "TRAINING_CONFIG": "config/sft_29B_720p.yaml",
                "NEGATIVE_PROMPT_FILE": "inference/prompts/negative_prompt.json",
                "HF_HOME": str(args.hf_home), "HF_HUB_CACHE": str(args.hf_home / "hub"),
                "HUGGINGFACE_HUB_CACHE": str(args.hf_home / "hub"),
                "TRANSFORMERS_CACHE": str(args.hf_home / "hub"),
                "HF_ASSETS_CACHE": str(args.hf_home / "assets"),
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    for key in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
                "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR", "QUACK_CACHE_DIR"):
        env[key] = str(args.output / (label + "-cache") / key.lower())
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "baseline", "candidate", "prompts", "output", "receipt", "hf-home"):
        parser.add_argument("--" + name, required=True, type=pathlib.Path)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args(argv)
    for name in ("repo", "baseline", "candidate", "prompts", "output", "receipt", "hf_home"):
        path = getattr(args, name).absolute()
        if path.is_symlink():
            raise RuntimeError("input/output root must not be a symlink")
        setattr(args, name, path.resolve())
    if args.timeout < 1:
        raise RuntimeError("timeout must be positive")
    roots = [args.repo, args.baseline, args.candidate, args.hf_home, args.output]
    for i, root in enumerate(roots):
        if any(root.is_relative_to(other) or other.is_relative_to(root) for other in roots[:i]):
            raise RuntimeError("repository, checkpoints, cache and output must be disjoint")
    if any(path.is_relative_to(args.output) for path in (args.prompts, args.receipt)):
        raise RuntimeError("output must not contain inputs")
    regular(args.prompts)
    regular(args.receipt)
    flags = source_flags(args.repo)
    head = git(args.repo, "rev-parse", "HEAD")
    if head != PIN:
        raise RuntimeError("PAN2 checkout does not match pinned revision")
    if git(args.repo, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("PAN2 checkout must be clean")
    # Ignored Python source can shadow a tracked module despite clean git status.
    if git(args.repo, "ls-files", "--others", "--ignored", "--exclude-standard", "--", "*.py", "*.so", "*.sh"):
        raise RuntimeError("ignored executable source in PAN2 checkout")
    before = {"baseline": checkpoint(args.baseline), "candidate": checkpoint(args.candidate)}
    if before["baseline"].keys() != before["candidate"].keys():
        raise RuntimeError("checkpoint file sets differ")
    changed = [key for key in before["baseline"] if before["baseline"][key] != before["candidate"][key]]
    if len(changed) != 1 or not changed[0].endswith(".safetensors"):
        raise RuntimeError("exactly one edited shard and otherwise identical files required")
    key = changed[0]
    receipt, receipt_bytes = json_snapshot(args.receipt)
    receipt_hash = hashlib.sha256(receipt_bytes).hexdigest()
    verify_receipt(receipt, args.baseline / key, args.candidate / key,
                   before["baseline"][key]["sha256"], before["candidate"][key]["sha256"])
    prompt_data, prompt_bytes = json_snapshot(args.prompts)
    prompts_hash = hashlib.sha256(prompt_bytes).hexdigest()
    images = prompt_images(prompt_data, args.repo)
    cache = model_cache(args.hf_home)
    source = git(args.repo, "ls-files", "-s")
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("CUDA executor unavailable: nvidia-smi missing")
    gpu = subprocess.check_output(["nvidia-smi"], text=True, timeout=30)
    args.output.mkdir(parents=True, exist_ok=False)
    prompts = args.output / "prompts.json"
    prompts.write_bytes(prompt_bytes)
    (args.output / "receipt.json").write_bytes(receipt_bytes)

    def unchanged():
        if (before != {"baseline": checkpoint(args.baseline), "candidate": checkpoint(args.candidate)}
                or digest(prompts) != prompts_hash or digest(args.prompts) != prompts_hash
                or digest(args.receipt) != receipt_hash
                or digest(args.output / "receipt.json") != receipt_hash
                or images != {path: digest(regular(pathlib.Path(path))) for path in images}
                or cache != model_cache(args.hf_home)
                or head != git(args.repo, "rev-parse", "HEAD")
                or source != git(args.repo, "ls-files", "-s")
                or flags != source_flags(args.repo)
                or git(args.repo, "diff", "HEAD", "--")
                or git(args.repo, "status", "--porcelain", "--untracked-files=all")
                or git(args.repo, "ls-files", "--others", "--ignored", "--exclude-standard", "--", "*.py", "*.so", "*.sh")):
            raise RuntimeError("inputs changed during evaluation")

    runs = {}
    for label in ("baseline", "candidate"):
        unchanged()
        start = time.perf_counter()
        with (args.output / (label + ".log")).open("x") as log:
            run_process(["bash", "--noprofile", "--norc", str(args.repo / "inference/inference.sh"),
                         str(getattr(args, label)), str(args.output / label), str(prompts)],
                        cwd=args.repo, env=environment(args, label), log=log, timeout=args.timeout)
        elapsed = time.perf_counter() - start
        unchanged()
        outputs = inventory(args.output / label)
        videos = {name: value["sha256"] for name, value in outputs.items() if name.endswith(".mp4")}
        if not videos:
            raise RuntimeError("inference produced no videos")
        if runs and videos.keys() != runs["baseline"]["videos"].keys():
            raise RuntimeError("baseline/candidate video sets differ")
        # Upstream saves the actual parsed arguments beside each video. Require
        # them so a stale/misdirected worker cannot qualify the other checkpoint.
        for name in videos:
            video = args.output / label / name
            if video.stat().st_size == 0:
                raise RuntimeError("empty video output")
            settings = read_json(video.with_name(video.stem + "_config.json"))
            actual = settings.get("arguments", {})
            expected = {"checkpoint_path": str(getattr(args, label)), "prompt_json": str(prompts),
                        "seed": args.seed, "num_inference_steps": 50, "flow_shift": 9,
                        "guidance_scale": 2.5, "video_height": 704, "video_width": 1248,
                        "fps": 24, "dtype": "bf16"}
            if not isinstance(actual, dict) or any(actual.get(k) != v for k, v in expected.items()):
                raise RuntimeError("output settings do not match selected checkpoint/sampler")
        runs[label] = {"seconds": elapsed, "videos": videos, "files": outputs}
    result = {"schema": "ruv-fann.pan2-paired-inference/1", "pan2_commit": head,
              "seed": args.seed, "prompts_sha256": prompts_hash, "receipt_sha256": receipt_hash,
              "checkpoints": before, "reference_images": images, "hf_home_files": cache,
              "gpu_inventory": gpu, "runs": runs,
              "harness_sha256": digest(pathlib.Path(__file__).resolve()),
              "host": platform.platform(), "harness_python": sys.version,
              "runtime_dependencies": "operator-provisioned pan Conda environment; not certified by this receipt",
              "timing_scope": "single cold per-run kernel cache; includes launcher/model loading; no speedup claim",
              "behavioral_validation": "requires independent scoring", "promotion": "not_performed"}
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
