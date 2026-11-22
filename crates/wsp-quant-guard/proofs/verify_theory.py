#!/usr/bin/env python3
"""Independent finite checks of the analytic claims, using Python integers.

This is an oracle for bounded cases, not a universal proof or a benchmark of
learned model accuracy.  It imports no implementation code from the crate.
"""

from fractions import Fraction
from itertools import product
import json
import math
import random


def quantize(accumulator, scale, low=-128, high=127):
    return max(low, min(high, accumulator // scale))


def cell(output, scale, low, high, minimum, maximum):
    assert scale > 0 and low <= output <= high
    if low == high:
        return minimum, maximum
    lower = minimum if output == low else scale * output
    upper = maximum if output == high else scale * (output + 1) - 1
    return max(minimum, lower), min(maximum, upper)


def ceil_div(numerator, denominator):
    return -((-numerator) // denominator)


def direction_interval(accumulator, coefficient, lower, upper):
    if coefficient == 0:
        return (-10**30, 10**30) if lower <= accumulator <= upper else None
    if coefficient > 0:
        return (ceil_div(lower - accumulator, coefficient),
                (upper - accumulator) // coefficient)
    return (ceil_div(upper - accumulator, coefficient),
            (lower - accumulator) // coefficient)


def check_cells():
    checks = 0
    for scale in range(1, 17):
        for low, high in [(-4, 4), (-4, -4), (0, 7)]:
            for q in range(low, high + 1):
                lower, upper = cell(q, scale, low, high, -1024, 1024)
                for accumulator in range(-1024, 1025):
                    assert (quantize(accumulator, scale, low, high) == q) == (
                        lower <= accumulator <= upper)
                    checks += 1
    return checks


def check_capacity():
    checks = 0
    cases = []
    for dimension in range(1, 6):
        for scale in range(2, 9):
            minimum_support = {}
            for weights in product(range(scale), repeat=dimension):
                # Protected features include zero plus every basis vector.
                assert quantize(0, scale) == 0
                assert all(quantize(weight, scale) == 0 for weight in weights)
                target_output = quantize(sum(weights), scale)
                support = sum(weight != 0 for weight in weights)
                minimum_support[target_output] = min(
                    support, minimum_support.get(target_output, dimension + 1))
                checks += 1
            upper = dimension * (scale - 1) // scale
            assert set(minimum_support) == set(range(upper + 1))
            for target_output, support in minimum_support.items():
                expected = 0 if target_output == 0 else ceil_div(
                    target_output * scale, scale - 1)
                assert support == expected
            cases.append({"dimension": dimension, "scale": scale,
                          "maximum_target_output": upper,
                          "minimum_support": minimum_support})
    return checks, cases


def check_direction_intervals():
    rng = random.Random(290471)
    checks = 0
    for _ in range(10000):
        accumulator = rng.randint(-80, 80)
        coefficient = rng.randint(-20, 20)
        lower = rng.randint(-80, 80)
        upper = lower + rng.randint(0, 100)
        interval = direction_interval(accumulator, coefficient, lower, upper)
        for k in range(-32, 33):
            actual = lower <= accumulator + k * coefficient <= upper
            predicted = interval is not None and interval[0] <= k <= interval[1]
            assert actual == predicted
            checks += 1
    return checks


def check_affine_separator():
    protected = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    target = (1, 1)
    checks = 0
    witnesses = {}
    for scale in range(1, 6):
        for vx, vy in product(range(-8, 9), repeat=2):
            accumulators = [vx * x + vy * y for x, y in protected]
            minimum, maximum = min(accumulators), max(accumulators)
            target_accumulator = vx * target[0] + vy * target[1]
            predicted = (maximum - minimum <= scale - 1 and
                         target_accumulator - maximum >= 1)
            biases = [bias for bias in range(-32, 33)
                      if all(quantize(a + bias, scale) == 0 for a in accumulators)
                      and quantize(target_accumulator + bias, scale) >= 1]
            assert predicted == bool(biases)
            if biases and scale not in witnesses:
                witnesses[scale] = {"weights": [vx, vy], "bias": biases[0]}
            checks += 1
    assert 1 not in witnesses and 2 not in witnesses
    assert 3 in witnesses
    return checks, witnesses


def counterexamples():
    baseline, scale, step = 1, 4, 1
    # Each increment passes a stale baseline upper slack check of 2.
    assert step <= scale - 1 - baseline
    sequence = [quantize(baseline + n * step, scale) for n in range(4)]
    assert sequence == [0, 0, 0, 1]

    before = [Fraction(49, 100), Fraction(49, 100)]
    delta = [Fraction(3, 5), -Fraction(3, 5)]
    after = [value + change for value, change in zip(before, delta)]
    nearest = lambda x: math.floor(x + Fraction(1, 2))
    assert sum(before) == sum(after)
    integer_before = [nearest(x) for x in before]
    integer_after = [nearest(x) for x in after]
    assert sum(integer_before) == 0 and sum(integer_after) == 1
    return {
        "stale_slack": {"baseline_accumulator": baseline, "scale": scale,
                        "increment": step, "outputs": sequence},
        "requantized_nullspace": {
            "real_before": [str(x) for x in before],
            "real_delta": [str(x) for x in delta],
            "integer_before": integer_before,
            "integer_after": integer_after,
            "protected_feature": [1, 1]},
    }


def heterogeneous_sequence():
    rng = random.Random(110226)
    dimension, scale = 128, 128
    bin_codes = [-1] * 64 + [0] * 64
    rng.shuffle(bin_codes)
    original = [scale * q + rng.randint(0, 15) for q in bin_codes]
    current = list(original)
    protected = [quantize(w, scale) for w in original]
    target_before = quantize(sum(original), scale)
    edited_entries = 0
    cumulative_abs_delta = 0
    target_outputs = [target_before]
    for step in range(100):
        remaining = scale
        for i in range(dimension):
            upper = min(127, scale * (protected[i] + 1) - 1)
            increment = min(remaining, upper - current[i])
            assert increment >= 0
            if increment:
                current[i] += increment
                remaining -= increment
                edited_entries += 1
                cumulative_abs_delta += increment
            if remaining == 0:
                break
        assert remaining == 0
        assert all(-128 <= w <= 127 for w in current)
        assert [quantize(w, scale) for w in current] == protected
        target_after = quantize(sum(current), scale)
        assert target_after == target_before + step + 1
        target_outputs.append(target_after)
    return {"dimension": dimension, "scale": scale,
            "protected_points": dimension + 1,
            "augmented_protected_rank": dimension + 1,
            "accepted_productive_patches": 100,
            "protected_output_mismatches": 0,
            "target_before": target_before, "target_after": target_outputs[-1],
            "changed_coordinate_events": edited_entries,
            "cumulative_l1_weight_change": cumulative_abs_delta,
            "seed": 110226,
            "scope": "Designed synthetic witness, not trained model accuracy"}


def main():
    cell_checks = check_cells()
    capacity_checks, capacity_cases = check_capacity()
    direction_checks = check_direction_intervals()
    affine_checks, affine_witnesses = check_affine_separator()
    result = {
        "status": "passed",
        "oracle": "Python arbitrary precision integers independent of Rust implementation",
        "cell_equivalence_cases": cell_checks,
        "capacity_exhaustive_vectors": capacity_checks,
        "capacity_configurations": len(capacity_cases),
        "direction_membership_cases": direction_checks,
        "affine_direction_configurations": affine_checks,
        "minimum_diamond_scale": 3,
        "diamond_witnesses": affine_witnesses,
        "counterexamples": counterexamples(),
        "heterogeneous_sequence": heterogeneous_sequence(),
        "formal_scope": "Bounded checks supplement paper proofs; no novelty certification",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
