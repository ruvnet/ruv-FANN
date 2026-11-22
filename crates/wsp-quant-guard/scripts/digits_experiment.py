#!/usr/bin/env python3
"""Real digits task for exact integer weight editing; CPU only, no downloads.

Train one small ridge classifier, then edit integer weights without retraining.
The certificate scope is listed raw pixel vectors, not all images or semantics.
All choices use training/calibration data; the test split is reporting only.

Dependencies: numpy, scikit-learn (the digits dataset ships with sklearn).
Example: python scripts/digits_experiment.py --output /tmp/digits.json
"""
import argparse
import copy
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


def replay_stdlib():
    """Replay frozen cases without numpy, sklearn, training, or dataset access."""
    parser = argparse.ArgumentParser(description="Replay frozen digits edits using Python standard library")
    parser.add_argument("--replay-only", type=Path, required=True)
    parser.add_argument("--rust-cli", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = args.replay_only.read_bytes()
    records = [json.loads(line) for line in raw.splitlines() if line]
    started = time.perf_counter()
    completed = subprocess.run([args.rust_cli], input="".join(
        json.dumps(r["request"]) + "\n" for r in records), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=True)
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    if len(responses) != len(records):
        raise AssertionError("replay response count mismatch")

    def predict(model, point):
        result = []
        for row, (weights, bias) in enumerate(zip(model["weights"], model["bias"])):
            accumulator = bias
            for weight, value in zip(weights, point):
                accumulator += weight * value
                if not -(2**31) <= accumulator < 2**31:
                    raise AssertionError("replay accumulator overflow")
            result.append(min(model["output_max"][row], max(model["output_min"][row],
                          accumulator // model["scales"][row])))
        return result

    feasible = 0
    for record, response in zip(records, responses):
        request, expected = record["request"], record["expected"]
        if not response.get("ok") or response.get("found") != expected["found"]:
            raise AssertionError("replay feasibility mismatch")
        if not expected["found"]:
            continue
        feasible += 1
        new, base = response["model"], request["model"]
        if new != expected["model"]:
            raise AssertionError("replay selected a different deterministic model")
        for field in base:
            if field != "weights" and new[field] != base[field]:
                raise AssertionError("replay changed frozen model semantics")
        changed = sum(a != b for wa, wb in zip(base["weights"], new["weights"]) for a, b in zip(wa, wb))
        if changed != 1 or any(not base["weight_min"] <= v <= base["weight_max"] for row in new["weights"] for v in row):
            raise AssertionError("replay violates weight contract")
        for point in request["protected"]["points"]:
            if predict(new, point) != predict(base, point):
                raise AssertionError("replay changed a protected output")
        for target in request["targets"]:
            if predict(new, target["input"]) != target["output"]:
                raise AssertionError("replay missed target")
    result = {"schema": "wsp-quant-guard-digits-replay-v1", "cases": len(records),
              "feasible": feasible, "infeasible": len(records) - feasible,
              "all_matched": True, "fixture_sha256": hashlib.sha256(raw).hexdigest(),
              "elapsed_seconds": time.perf_counter() - started,
              "python": platform.python_version(), "platform": platform.platform(),
              "solve_total_us": sum(r.get("solve_us", 0) for r in responses),
              "verify_total_us": sum(r.get("verify_us", 0) for r in responses)}
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__" and "--replay-only" in sys.argv:
    replay_stdlib()
    sys.exit(0)

import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


def sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def outputs(model, inputs):
    x = np.asarray(inputs, dtype=np.int64)
    if x.ndim == 1:
        x = x[None, :]
    z = x @ np.asarray(model["weights"], dtype=np.int64).T + model["bias"]
    return np.clip(z // model["scales"], model["output_min"], model["output_max"])


def cell(model, row, code):
    scale = model["scales"][row]
    low, high = model["output_min"][row], model["output_max"][row]
    if not low <= code <= high:
        raise ValueError("target outside output range")
    return (-(2**31) if code == low else max(-(2**31), code * scale),
            2**31 - 1 if code == high else min(2**31 - 1, (code + 1) * scale - 1))


def restrict(lo, hi, coefficient, lower, upper):
    """Intersect integer [lo,hi] with lower <= coefficient*d <= upper."""
    coefficient, lower, upper = int(coefficient), int(lower), int(upper)
    if coefficient == 0:
        return (lo, hi) if lower <= 0 <= upper else (1, 0)
    if coefficient < 0:
        coefficient, lower, upper = -coefficient, -upper, -lower
    return max(lo, -((-lower) // coefficient)), min(hi, upper // coefficient)


def solve(model, protected, input_, desired, mode):
    """Complete search for a nonzero SINGLE WEIGHT edit under this contract.

    This is not a complete solver for arbitrary multiweight neural repair.
    Uses exact Python integers in all interval arithmetic. Every returned model
    is independently replayed, including ordered int32 accumulator prefixes.
    """
    # This dataset is small enough that every permitted weight vector is safe.
    # Refuse wider arithmetic domains rather than claim completeness after a
    # post hoc overflow check that might discard a feasible larger delta.
    for point in protected + [input_]:
        bound = sum(abs(v) for v in point) * max(abs(model["weight_min"]), abs(model["weight_max"]))
        if any(abs(b) + bound >= 2**31 for b in model["bias"]):
            raise ValueError("reference experiment requires int32 safety for full int8 weight range")
    before = outputs(model, input_)[0].tolist()
    changed = [row for row, (a, b) in enumerate(zip(before, desired)) if a != b]
    if len(changed) != 1:
        return None
    row = changed[0]
    w = model["weights"][row]
    bias = model["bias"][row]
    a_target = bias + sum(a * b for a, b in zip(w, input_))
    target_bounds = cell(model, row, desired[row])
    cached = []
    for point in protected:
        accumulator = bias + sum(a * b for a, b in zip(w, point))
        code = int(outputs(model, point)[0, row])
        low, high = cell(model, row, code)
        cached.append((point, accumulator, low, high))
    candidates = []
    for col, old in enumerate(w):
        lo, hi = model["weight_min"] - old, model["weight_max"] - old
        lo, hi = restrict(lo, hi, input_[col], target_bounds[0] - a_target,
                          target_bounds[1] - a_target)
        for point, accumulator, low, high in cached:
            if mode == "accumulator":
                lo, hi = restrict(lo, hi, point[col], 0, 0)
            elif mode == "cells":
                lo, hi = restrict(lo, hi, point[col], low - accumulator, high - accumulator)
            if lo > hi:
                break
        if lo > hi:
            continue
        delta = lo if lo > 0 else hi if hi < 0 else 0
        if delta:
            candidates.append((abs(delta), col, delta))
    for _, col, delta in sorted(candidates):
        edited = copy.deepcopy(model)
        edited["weights"][row][col] += delta
        if valid(edited, model, protected, input_, desired, mode):
            return {"model": edited, "edit": {"row": row, "col": col, "delta": delta}}
    return None


def valid(edited, original, protected, input_, desired, mode):
    if not np.array_equal(outputs(edited, input_)[0], desired):
        return False
    if protected and mode != "target_only":
        if not np.array_equal(outputs(edited, protected), outputs(original, protected)):
            return False
        if mode == "accumulator":
            delta = np.asarray(edited["weights"]) - original["weights"]
            if np.any(np.asarray(protected, dtype=np.int64) @ delta.T):
                return False
    for point in protected + [input_]:
        for weights, bias in zip(edited["weights"], edited["bias"]):
            accumulator = int(bias)
            if not -(2**31) <= accumulator < 2**31:
                return False
            for w, x in zip(weights, point):
                accumulator += int(w) * int(x)
                if not -(2**31) <= accumulator < 2**31:
                    return False
    return True


class RustCheck:
    def __init__(self, executable):
        self.process = subprocess.Popen(
            [executable], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        self.total = 0
        self.feasible = 0
        self.solve_us = []
        self.verify_us = []
        self.records = []
        self.same_candidate = 0

    def check(self, model, protected, point, desired, python_result):
        request = {"op": "solve", "model": model,
                   "protected": {"points": protected, "boxes": []},
                   "targets": [{"input": point, "output": desired}]}
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Rust CLI terminated: " + self.process.stderr.read())
        response = json.loads(line)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError("Rust CLI rejected valid experiment: " + str(response))
        if isinstance(response, dict) and "result" in response:
            response = response["result"]
        found = response is not None and response.get("found", True)
        if found != (python_result is not None):
            raise AssertionError("Rust/Python disagree on single-coordinate feasibility")
        if found:
            edited = response["model"]
            if not valid(edited, model, protected, point, desired, "cells"):
                raise AssertionError("Rust returned invalid protected or target behavior")
            delta = np.asarray(edited["weights"]) - model["weights"]
            if np.count_nonzero(delta) != 1:
                raise AssertionError("Rust patch violates single-coordinate contract")
            if edited == python_result["model"]:
                self.same_candidate += 1
            self.feasible += 1
        if response is not None:
            self.solve_us.append(response.get("solve_us", 0))
            self.verify_us.append(response.get("verify_us", 0))
        self.total += 1
        # Full replay inputs contain every success and one failure for each
        # observed protected-set size, keeping the portable fixture small.
        if found or not any(not r["expected"]["found"] and
                            len(r["request"]["protected"]["points"]) == len(protected)
                            for r in self.records):
            self.records.append({"request": copy.deepcopy(request),
                                 "expected": {"found": found, "model": response.get("model")}})
        return response

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=10)
        if self.process.returncode:
            raise RuntimeError(self.process.stderr.read())


def setup(seed):
    start = time.perf_counter()
    x, y = load_digits(return_X_y=True)
    indices = np.arange(len(y))
    train, other = train_test_split(indices, train_size=.55, stratify=y, random_state=seed)
    calibration, other = train_test_split(
        other, train_size=1 / 3, stratify=y[other], random_state=seed + 1)
    edit, test = train_test_split(other, train_size=.5, stratify=y[other], random_state=seed + 2)
    feature_mask = np.ptp(x[train], axis=0) > 0
    x = x[:, feature_mask].astype(np.int64)
    started = time.perf_counter()
    classifier = Ridge(alpha=1.0)
    classifier.fit(x[train], np.eye(10)[y[train]])
    training_seconds = time.perf_counter() - started
    multiplier = 127.0 / np.abs(classifier.coef_).max()
    weights = np.rint(classifier.coef_ * multiplier).astype(np.int64).tolist()
    bias = np.rint(classifier.intercept_ * multiplier).astype(np.int64).tolist()
    prefix = {"dataset": "sklearn.load_digits", "features": np.flatnonzero(feature_mask).tolist(),
              "feature_semantics": "raw integer pixels 0..16; no mutable upstream network"}
    model = {"weights": weights, "bias": bias, "scales": [1] * 10,
             "output_min": [-32768] * 10, "output_max": [32767] * 10,
             "weight_min": -128, "weight_max": 127, "prefix_id": sha(prefix)}
    scale_grid = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    calibration_scores = []
    for scale in scale_grid:
        model["scales"] = [scale] * 10
        score = float(np.mean(outputs(model, x[calibration]).argmax(1) == y[calibration]))
        calibration_scores.append({"scale": scale, "accuracy": score})
    reference = calibration_scores[0]["accuracy"]
    scale = max(c["scale"] for c in calibration_scores if c["accuracy"] >= reference - .01)
    model["scales"] = [scale] * 10
    details = {
        "seed": seed, "split_sizes": {"train": len(train), "calibration": len(calibration),
                                        "edit": len(edit), "test": len(test)},
        "split_sha256": sha({"train": train.tolist(), "calibration": calibration.tolist(),
                             "edit": edit.tolist(), "test": test.tolist()}),
        "dataset_sha256": hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest(),
        "prefix": prefix, "input_dimensions": x.shape[1], "classes": 10,
        "weight_multiplier": float(multiplier), "calibration_scores": calibration_scores,
        "selected_scale": scale,
        "scale_rule": "coarsest declared scale within 1 percentage point of scale1 calibration accuracy",
        "training_seconds": training_seconds, "setup_seconds": time.perf_counter() - start,
        "base_model_sha256": sha(model),
        "float_test_accuracy": float(np.mean(classifier.predict(x[test]).argmax(1) == y[test])),
    }
    return model, x, y, calibration, edit, test, details


def experiment(model, x, y, anchors, edit, test, mode, rust):
    current = copy.deepcopy(model)
    anchor_points = x[anchors].tolist()
    protected = list(anchor_points)
    baseline_codes = outputs(model, x[test])
    baseline_labels = baseline_codes.argmax(1)
    anchor_codes = outputs(model, anchor_points)
    initial_codes = outputs(model, x[edit])
    attempts = edit[initial_codes.argmax(1) != y[edit]]
    accepted, audit = [], []
    solve_times = []
    reference_times = []
    max_protected_code_changes = 0
    for index in attempts:
        point = x[index].tolist()
        codes = outputs(current, point)[0].tolist()
        true_label = int(y[index])
        if int(np.argmax(codes)) == true_label:
            audit.append({"dataset_index": int(index), "status": "already_correct_after_prior_edit"})
            continue
        desired = codes[:]
        desired[true_label] = max(v for r, v in enumerate(codes) if r != true_label) + 1
        used_protection = [] if mode == "target_only" else protected
        started = time.perf_counter()
        result = solve(current, used_protection, point, desired, mode)
        reference_elapsed = time.perf_counter() - started
        reference_times.append(reference_elapsed)
        if rust and mode == "cells":
            response = rust.check(current, used_protection, point, desired, result)
            solve_times.append(response.get("solve_us", 0) / 1_000_000)
            if response["found"]:
                changes = np.argwhere(np.asarray(response["model"]["weights"]) - current["weights"])
                row, col = (int(v) for v in changes[0])
                delta = response["model"]["weights"][row][col] - current["weights"][row][col]
                result = {"model": response["model"], "edit": {"row": row, "col": col, "delta": delta}}
        else:
            solve_times.append(reference_elapsed)
        if result is None:
            audit.append({"dataset_index": int(index), "status": "no_single_coordinate_solution"})
            continue
        old_hash = sha(current)
        current = result["model"]
        protected.append(point)
        accepted.append({"dataset_index": int(index), "label": true_label,
                         "expected_codes": desired, "base_sha256": old_hash,
                         "result_sha256": sha(current), "edit": result["edit"]})
        protected_changed = int(np.count_nonzero(np.any(outputs(current, anchor_points) != anchor_codes, axis=1)))
        max_protected_code_changes = max(max_protected_code_changes, protected_changed)
        if mode != "target_only":
            if protected_changed:
                raise AssertionError("accepted protected code drift")
            for prior in accepted:
                if outputs(current, x[prior["dataset_index"]])[0].tolist() != prior["expected_codes"]:
                    raise AssertionError("accepted prior target code drift")
        audit.append({"dataset_index": int(index), "status": "accepted"})
    test_codes = outputs(current, x[test])
    predicted = test_codes.argmax(1)
    retained = sum(int(np.argmax(outputs(current, x[a["dataset_index"]])[0])) == a["label"] for a in accepted)
    retained_exact = sum(outputs(current, x[a["dataset_index"]])[0].tolist() == a["expected_codes"] for a in accepted)
    # Undo the actual patches in reverse order and require byte equivalent model data.
    restored = copy.deepcopy(current)
    for record in reversed(accepted):
        change = record["edit"]
        restored["weights"][change["row"]][change["col"]] -= change["delta"]
    if restored != model:
        raise AssertionError("rollback failed")
    return {
        "method": mode, "protected_anchor_count": len(anchors),
        "engine": "rust_with_python_crosscheck" if rust and mode == "cells" else "python",
        "protected_feature_rank": int(np.linalg.matrix_rank(x[anchors].astype(float))),
        "input_dimensions": x.shape[1], "initial_edit_pool_errors": len(attempts),
        "solver_calls": len(solve_times), "accepted_effective_edits": len(accepted),
        "already_correct_from_other_edits": sum(a["status"] == "already_correct_after_prior_edit" for a in audit),
        "final_retained_target_labels": retained, "final_retained_target_codes": retained_exact,
        "max_changed_anchor_codes_after_any_patch": max_protected_code_changes,
        "final_changed_anchor_codes": int(np.count_nonzero(np.any(outputs(current, anchor_points) != anchor_codes, axis=1))),
        "test_accuracy_before": float(np.mean(baseline_labels == y[test])),
        "test_accuracy_after": float(np.mean(predicted == y[test])),
        "test_cases_helped": int(np.sum((baseline_labels != y[test]) & (predicted == y[test]))),
        "test_cases_harmed": int(np.sum((baseline_labels == y[test]) & (predicted != y[test]))),
        "test_changed_exact_codes": int(np.count_nonzero(np.any(test_codes != baseline_codes, axis=1))),
        "final_changed_weights": int(np.count_nonzero(np.asarray(current["weights"]) - model["weights"])),
        "synthesis_total_seconds": sum(solve_times),
        "python_reference_total_seconds": sum(reference_times),
        "synthesis_median_seconds": float(np.median(solve_times)) if solve_times else 0,
        "rollback_equal": True, "final_model_sha256": sha(current),
        "accepted_edits": accepted, "attempts": audit,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anchors", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--rust-cli", help="optional persistent JSONL Rust solver executable")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay-output", type=Path, help="portable full JSONL Rust inputs plus expected results")
    args = parser.parse_args()
    model, x, y, calibration, edit, test, details = setup(args.seed)
    correct = calibration[outputs(model, x[calibration]).argmax(1) == y[calibration]]
    if any(n <= 0 or n > len(correct) for n in args.anchors):
        raise ValueError("anchor counts must be positive and available from calibration")
    rust = RustCheck(args.rust_cli) if args.rust_cli else None
    runs = []
    try:
        for n in args.anchors:
            for mode in ("accumulator", "cells", "target_only"):
                runs.append(experiment(model, x, y, correct[:n], edit, test, mode, rust))
    finally:
        if rust:
            rust.close()
    report = {
        "schema": "wsp-quant-guard-digits-v1", "setup": details, "runs": runs,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "sklearn": sklearn.__version__, "platform": platform.platform()},
        "rust_crosscheck": {"enabled": rust is not None, "cases": rust.total if rust else 0,
                            "feasible": rust.feasible if rust else 0,
                            "same_candidate": rust.same_candidate if rust else 0,
                            "solve_total_us": sum(rust.solve_us) if rust else 0,
                            "verify_total_us": sum(rust.verify_us) if rust else 0},
        "scope": "single integer weight edits to a fixed linear readout; exact listed input code preservation",
        "limitations": ["one fixed split; no claim of statistical significance or new theorem",
                        "no GPU or hardware deployment measurement",
                        "no claim about unconstrained multiweight feasibility",
                        "anchor protection does not imply unseen data protection",
                        "this is constrained synthesis, not an LLM weight discovery benchmark"],
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    else:
        print(rendered)
    if args.replay_output:
        if not rust:
            raise ValueError("--replay-output requires --rust-cli")
        args.replay_output.parent.mkdir(parents=True, exist_ok=True)
        args.replay_output.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rust.records))


if __name__ == "__main__":
    main()
