#!/usr/bin/env python3
"""Independent integer oracle and end to end research acceptance. Stdlib only."""
import argparse
import copy
import hashlib
import itertools
import json
import platform
import random
import subprocess
import time
from pathlib import Path


class Client:
    def __init__(self, executable):
        self.process = subprocess.Popen([str(executable)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True)

    def call(self, **request):
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise AssertionError("Rust checker terminated without a response")
        return json.loads(line)

    def close(self):
        self.process.stdin.close()
        assert self.process.wait(timeout=10) == 0


def predict(model, inputs):
    output = []
    for row, weights in enumerate(model["weights"]):
        acc = model["bias"][row]
        for w, x in zip(weights, inputs):
            product = w * x
            if not -(2**31) <= product < 2**31:
                raise OverflowError("product")
            acc += product
            if not -(2**31) <= acc < 2**31:
                raise OverflowError("partial sum")
        q = acc // model["scales"][row]
        output.append(max(model["output_min"][row], min(model["output_max"][row], q)))
    return output


def model(weights, scale=4, bias=0, lower=-8, upper=7, minimum=-3, maximum=3):
    return dict(weights=[list(weights)], bias=[bias], scales=[scale],
                output_min=[lower], output_max=[upper], weight_min=minimum,
                weight_max=maximum, prefix_id="fixture:integer-features:v1")


def apply(model, edits):
    candidate = copy.deepcopy(model)
    for edit in edits:
        candidate["weights"][edit["row"]][edit["col"]] = edit["value"]
    return candidate


def exhaustive(model, protected_points, target):
    """Independent complete enumeration of the declared one weight search class."""
    baseline = [predict(model, x) for x in protected_points]
    for row, weights in enumerate(model["weights"]):
        for col, old in enumerate(weights):
            for value in range(model["weight_min"], model["weight_max"] + 1):
                if value == old:
                    continue
                candidate = apply(model, [dict(row=row, col=col, value=value)])
                if predict(candidate, target["input"]) != target["output"]:
                    continue
                if [predict(candidate, x) for x in protected_points] == baseline:
                    return True
    return False


def solver_oracle(client, cases=400):
    rng = random.Random(20261122)
    feasible = checked_patches = box_cases = 0
    for case in range(cases):
        m = model([rng.randrange(-3, 4) for _ in range(2)], rng.randrange(1, 9),
                  rng.randrange(-8, 9), -2, 2)
        points = [[rng.randrange(-3, 4) for _ in range(2)] for _ in range(case % 5)]
        boxes = []
        if case % 4 == 0:
            lo = [rng.randrange(-2, 2) for _ in range(2)]
            hi = [x + 1 for x in lo]
            grid = list(itertools.product(*[range(a, b + 1) for a, b in zip(lo, hi)]))
            if len({tuple(predict(m, x)) for x in grid}) == 1:
                boxes = [dict(lower=lo, upper=hi)]
                box_cases += 1
            else:
                grid = []
        else:
            grid = []
        protected = dict(points=points, boxes=boxes)
        target = dict(input=[rng.randrange(-4, 5) for _ in range(2)],
                      output=[rng.randrange(-2, 3)])
        expected = exhaustive(m, points + grid, target)
        result = client.call(op="solve", model=m, protected=protected, targets=[target])
        assert result["ok"], (case, result)
        assert result["found"] == expected, (case, m, protected, target, result)
        feasible += expected
        if expected:
            candidate = apply(m, result["patch"]["edits"])
            assert candidate == result["model"]
            assert predict(candidate, target["input"]) == target["output"]
            for point in points + grid:
                assert predict(candidate, point) == predict(m, point)
            valid = client.call(op="verify", model=m, protected=protected,
                                patch=result["patch"], certificate=result["certificate"])
            assert valid == {"ok": True, "verified": True}
            bad = copy.deepcopy(result["certificate"])
            bad["candidate_hash"] = "0" * 64
            invalid = client.call(op="verify", model=m, protected=protected,
                                  patch=result["patch"], certificate=bad)
            assert not invalid["ok"]
            checked_patches += 1
    return dict(cases=cases, feasible=feasible, exact_agreements=cases,
                constant_box_cases=box_cases, independently_verified_patches=checked_patches,
                altered_certificates_rejected=checked_patches)


def hundred_edits(client):
    width = 128
    rng = random.Random(8)
    weights = [rng.choice([-8, -7, -6, -5, 1, 2, 3, 4]) for _ in range(width)]
    m = model(weights, scale=16, bias=0, lower=-128, upper=127,
              minimum=-128, maximum=127)
    anchors = [[0] * width]
    for i in range(width):
        h = [0] * width
        h[i] = 1
        anchors.append(h)
    protected = dict(points=anchors, boxes=[])
    steps = [[dict(row=0, col=i, value=weights[i] + 1)] for i in range(100)]
    result = client.call(op="sequence", model=m, protected=protected,
                         steps=steps, rollback=True)
    assert result["ok"]
    current = copy.deepcopy(m)
    targets = []
    protected_checks = target_checks = 0
    for i, record in enumerate(result["records"]):
        assert record["accepted"], record
        current = apply(current, record["patch"]["edits"])
        for anchor in anchors:
            assert predict(current, anchor) == predict(m, anchor)
            protected_checks += 1
        h = [0] * width
        h[i] = 16
        targets.append((h, [weights[i] + 1]))
        for target, desired in targets:
            assert predict(current, target) == desired
            target_checks += 1
    assert result["final_model"] == current
    assert result["rolled_back"] == 100
    assert result["after_rollback"] == m
    assert result["baseline_restored"]
    assert result["rollback_head"] != result["final_head"]
    latencies = sorted(r["verify_us"] for r in result["records"])
    return dict(dimensions=width, protected_points=len(anchors),
                augmented_protected_rank=width + 1, accepted_distinct_edits=100,
                distinct_targets_corrected_and_retained=100,
                original_protected_checks=protected_checks,
                accumulated_target_checks=target_checks, protected_mismatches=0,
                rolled_back=100, baseline_restored=True,
                changed_parameter_fraction=100 / width,
                certificate_check_us_p50=latencies[len(latencies)//2],
                certificate_check_us_p95=latencies[94],
                scope="Constructed synthetic layer. 100 scalar target outputs, not 100 learned capabilities.")


def counterexamples(client):
    # Reusing original slack independently would approve both +1 patches;
    # the cumulative second patch crosses the protected cell boundary.
    m = model([6], scale=8, minimum=-128, maximum=127)
    result = client.call(op="sequence", model=m, protected=dict(points=[[1]], boxes=[]),
                         steps=[[dict(row=0,col=0,value=7)],
                                [dict(row=0,col=0,value=8)]], rollback=True)
    assert result["ok"] and result["records"][0]["accepted"]
    assert not result["records"][1]["accepted"]
    assert result["records"][1]["unchanged_head"] and result["baseline_restored"]
    # Real edit [0.5,-0.25] annihilates h=(1,2); rounding to integers
    # (ties to even) annihilates the edit as well, losing target change.
    real_delta, anchor, target = [0.5,-0.25], [1,2], [2,0]
    assert sum(a*b for a,b in zip(real_delta,anchor)) == 0
    rounded = [round(x) for x in real_delta]
    assert sum(a*b for a,b in zip(real_delta,target)) == 1
    assert sum(a*b for a,b in zip(rounded,target)) == 0
    # A different exact float null direction rounds to a leaking integer row.
    delta, h = [0.6,-0.2], [1,3]
    assert abs(sum(a*b for a,b in zip(delta,h))) < 1e-15
    integer_leak = sum(round(a)*b for a,b in zip(delta,h))
    assert integer_leak == 1
    return dict(cumulative_violation_rejected=True, rejected_update_atomic=True,
                float_nullspace_rounding_target_lost=True,
                float_nullspace_rounding_accumulator_leak=integer_leak)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    client = Client(args.binary.resolve())
    try:
        result = dict(schema="wsp-quant-acceptance-v1", seed=20261122,
                      python=platform.python_version(), machine=platform.machine(),
                      oracle=solver_oracle(client), sequential=hundred_edits(client),
                      counterexamples=counterexamples(client))
    finally:
        client.close()
    result["elapsed_seconds"] = time.perf_counter() - started
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["binary_sha256"] = hashlib.sha256(args.binary.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
