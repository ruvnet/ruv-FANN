"""Compare one ruv-FANN Safetensors edit on a preprovisioned, offline CUDA host.

Use trusted materialized Qwen3.5 model trees and a preinstalled PyTorch,
Transformers and Accelerate runtime. No model downloads, provider calls, or model promotion.
The smoke records paired outputs and timings, not generalization or speedup.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

from pan2_compare import digest, inventory, json_snapshot, verify_receipt


PROMPTS = (
    "Answer with the integer only: 17 + 28.",
    "Answer with the integer only: 12 times 9.",
    "Answer with the integer only: 144 divided by 12.",
    "Answer with the integer only: 93 minus 46.",
    "Answer with one word only: the opposite of cold.",
    "Answer with one word only: the color of a ripe banana peel.",
    "Answer with the integer only: how many days are in two weeks?",
    "Answer with the integer only: how many sides does a hexagon have?",
    "Return only the sorted numbers: 5, 2, 9, 1.",
    "Return only the next number: 3, 6, 12, 24.",
    "Return only the first character of the word meadow.",
    "Return only the word that does not belong: apple, pear, hammer, plum.",
    "Return only a JSON object with the key count and the integer value 3.",
    "In one sentence, explain why metal feels colder than wood at room temperature.",
    "In one sentence, summarize: The bus was late, so Mira walked to work.",
    "In one sentence, explain the difference between a backup and a rollback.",
)
MAX_NEW_TOKENS = 64


def checkpoint(root):
    data_suffixes = {".json", ".jinja", ".md", ".txt", ".safetensors"}
    if not root.is_dir() or any(
            path.is_symlink() or not path.is_file()
            or (path.suffix not in data_suffixes and path.name not in {".gitattributes", "LICENSE"})
            for path in root.iterdir()):
        raise RuntimeError("requires a materialized top-level data-only model tree with no pickle or symlinks")
    files = inventory(root)
    config, _ = json_snapshot(root / "config.json")
    if (not isinstance(config, dict) or config.get("model_type") != "qwen3_5"
            or config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]
            or "quantization_config" in config):
        raise RuntimeError("requires the unquantized Qwen3.5 conditional generation layout")

    shards = {name for name in files if name.endswith(".safetensors")}
    index_name = "model.safetensors.index.json"
    if index_name in files:
        index, _ = json_snapshot(root / index_name)
        weights = index.get("weight_map") if isinstance(index, dict) else None
        if (not isinstance(weights, dict) or not weights
                or any(not isinstance(name, str) for name in weights.values())
                or set(weights.values()) != shards or "model.safetensors" in shards):
            raise RuntimeError("Safetensors shard inventory does not match the weight index")
    elif shards != {"model.safetensors"}:
        raise RuntimeError("requires model.safetensors or a matching shard index")
    return files


def validate_inputs(baseline, candidate, shard, receipt_path):
    checkpoints = {"baseline": checkpoint(baseline), "candidate": checkpoint(candidate)}
    base, edited = checkpoints.values()
    if base.keys() != edited.keys():
        raise RuntimeError("checkpoint file sets differ")
    if [name for name in base if base[name] != edited[name]] != [shard]:
        raise RuntimeError("the selected shard must be the only changed file")

    receipt, receipt_bytes = json_snapshot(receipt_path)
    if (not isinstance(receipt, dict)
            or set(receipt) != {"schema", "base_sha256", "candidate_sha256", "edits",
                                "behavioral_validation"}
            or receipt["behavioral_validation"] != "not_run"):
        raise RuntimeError("requires an unmodified ruv-FANN tensor edit receipt")
    verify_receipt(receipt, baseline / shard, candidate / shard, base[shard]["sha256"],
                   edited[shard]["sha256"])
    return checkpoints, hashlib.sha256(receipt_bytes).hexdigest()


def infer(baseline, candidate):
    # Set before importing HF or loading any model/tokenizer.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers
    from transformers import AutoTokenizer, GenerationConfig, Qwen3_5ForConditionalGeneration

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("requires a CUDA device with BF16 support")
    device = torch.device("cuda:0")
    gpu = torch.cuda.get_device_properties(device)
    tokenizer = AutoTokenizer.from_pretrained(
        baseline, local_files_only=True, trust_remote_code=False, token=False)
    inputs = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=True, return_tensors="pt")
        for prompt in PROMPTS
    ]
    if any(encoded["input_ids"].shape[1] > 256 for encoded in inputs):
        raise RuntimeError("fixed prompt token budget exceeded")

    runs = {}
    for label, model_path in (("baseline", baseline), ("candidate", candidate)):
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        # A concrete library class keeps model resolution out of repository code.
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, use_safetensors=True, token=False,
            dtype=torch.bfloat16, device_map={"": str(device)}, attn_implementation="sdpa")
        try:
            model.eval()
            torch.cuda.synchronize(device)
            load_seconds = time.perf_counter() - start
            generation = GenerationConfig(
                do_sample=False, num_beams=1, max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True, eos_token_id=model.generation_config.eos_token_id,
                pad_token_id=tokenizer.pad_token_id)
            records = []
            with torch.inference_mode():
                for encoded in inputs:
                    input_ids = encoded["input_ids"].to(device)
                    attention_mask = encoded["attention_mask"].to(device)
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    output = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                            generation_config=generation)
                    torch.cuda.synchronize(device)
                    seconds = time.perf_counter() - start
                    input_tokens = input_ids.shape[1]
                    new_ids = output[0, input_tokens:].tolist()
                    records.append({"input_tokens": input_tokens, "output_tokens": len(new_ids),
                                    "token_ids": new_ids, "seconds": seconds,
                                    "text": tokenizer.decode(new_ids, skip_special_tokens=True)})
                    del input_ids, attention_mask, output
            runs[label] = {"load_seconds": load_seconds, "prompts": records,
                           "generate_seconds": sum(row["seconds"] for row in records),
                           "output_tokens": sum(row["output_tokens"] for row in records),
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    runtime = {"gpu": gpu.name, "gpu_memory_bytes": gpu.total_memory,
               "torch": torch.__version__, "cuda_build": torch.version.cuda,
               "transformers": transformers.__version__, "python": sys.version.split()[0],
               "platform": platform.platform()}
    return runs, runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "receipt", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--shard", required=True, help="Exact top-level edited .safetensors name")
    parser.add_argument("--verify-only", action="store_true", help="Verify on CPU without inference")
    args = parser.parse_args(argv)
    for name in ("baseline", "candidate", "receipt", "output"):
        path = getattr(args, name)
        if path.is_symlink():
            raise RuntimeError("model, receipt and output paths must not be symlinks")
        setattr(args, name, path.resolve())
    if (args.baseline.is_relative_to(args.candidate)
            or args.candidate.is_relative_to(args.baseline)
            or args.output.is_relative_to(args.baseline) or args.output.is_relative_to(args.candidate)
            or args.output.exists()):
        raise RuntimeError("model trees must be disjoint and the output path must be new and external")
    if Path(args.shard).name != args.shard or not args.shard.endswith(".safetensors"):
        raise RuntimeError("selected shard must be a top-level Safetensors file name")

    checkpoints, receipt_hash = validate_inputs(args.baseline, args.candidate, args.shard, args.receipt)
    report = {
        "schema": "ruv-fann.qwen-paired-smoke/1",
        "status": "verified_only" if args.verify_only else "gpu_smoke",
        "edited_shard": args.shard,
        "checkpoint_files": checkpoints,
        "edit_receipt_sha256": receipt_hash,
        "harness_sha256": digest(Path(__file__).resolve()),
        "verification_helper_sha256": digest(Path(verify_receipt.__code__.co_filename).resolve()),
        "rollback_bytes_verified": True,
        "promotion": "not_performed",
    }
    if not args.verify_only:
        runs, runtime = infer(args.baseline, args.candidate)
        after = {"baseline": checkpoint(args.baseline), "candidate": checkpoint(args.candidate)}
        _, receipt_after = json_snapshot(args.receipt)
        if checkpoints != after or hashlib.sha256(receipt_after).hexdigest() != receipt_hash:
            raise RuntimeError("inputs changed during evaluation")
        report.update({
            "prompts": PROMPTS, "runtime": runtime, "runs": runs,
            "decode": {"strategy": "greedy", "thinking": False, "max_new_tokens": MAX_NEW_TOKENS},
            "behavioral_validation": "paired outputs only, no quality gate",
            "timing_scope": "sequential checkpoint loads, one generation per prompt, no speedup claim",
        })
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        result = Path(temporary) / "report.json"
        result.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.link(result, args.output)
    return report


if __name__ == "__main__":
    try:
        main()
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        sys.exit(f"qwen_compare: {error}")
