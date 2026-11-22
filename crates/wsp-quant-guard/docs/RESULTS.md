# Research results: certified quantized edits

Date: 2026-11-22. Status: reproducible research prototype; publication novelty
unestablished. Source repository: ruvnet/ruv-FANN. Coordination record:
https://github.com/ruvnet/core-memory/issues/240.

## Main finding

Requiring an unchanged quantized output is less restrictive than requiring an
unchanged integer accumulator. The difference can support useful sparse edits.
We derived exact capacity and minimum support formulas for a full rank
construction, implemented a checked integer solver and patch ledger, formally
checked selected lemmas, and tested a trained digit classifier.

The research method combines explicit specifications, parallel conjecture and
literature review, counterexample search, independent arithmetic oracles,
Lean checking, and executable experiments. It does not use the Riemann
hypothesis itself as an assumption or transfer a number theoretic proof into
neural computation.

## Mathematical result and its boundary

Protect zero and each coordinate basis vector in d dimensions. For a zero
baseline, fixed bias and positive integer scale s at least 2, every integer
weight vector in [0,s−1]^d preserves protected output zero. At the all ones
target, every integer output from 0 through floor(d(s−1)/s) is reachable,
provided representable weights and no relevant output saturation.

The exact minimum number of changed weights needed for positive output r is
ceil(rs/(s−1)). Exact preservation of the protected raw accumulators forces
all weight and bias edits to zero. Thus full protected rank does not imply
that all quantized behavior preserving edits are impossible.

A second derived condition relates integer directional width to cell width.
A target outside a protected convex hull can still be uneditable at a given
integer quantizer scale. The four point diamond example is impossible at
scale 2 and feasible at scale 3. Both assertions have Lean proofs.

These statements are elementary consequences of established mathematics.
They are useful characterizations in this experiment, not established novel
theorems. The [proof note](../proofs/THEORY.md) supplies complete paper arguments,
formal coverage boundaries and counterexamples. The prior art search found
existing null space editing, quantized repair and provable region repair.

## Synthetic sequential experiment

The production code accepted 100 distinct one weight replacements in a
128 dimensional int8 bounded layer. A separate Python integer interpreter
recomputed all protected and prior target outputs after every step.

| Measurement | Result |
|:--|:--|
| Original protected inputs | 129, comprising zero and every basis vector |
| Augmented protected rank | 129 |
| Distinct effective accepted patches | 100 |
| Distinct scalar targets corrected and retained | 100 |
| Original protected output comparisons | 12,900 |
| Accumulated target comparisons | 5,050 |
| Protected output mismatches | 0 |
| Successful rollbacks | 100 |
| Original model restored | Exactly |

The construction deliberately provides headroom. Each patch changes one of
128 weights, but all patches together change 100/128 weights, or 78.125%.
Calling the entire sequence a tiny cumulative model edit would be misleading.
The 100 targets are distinct scalar constraints, not 100 learned capabilities.

In a local optimized build, certificate recomputation in this fixture had a
median around 40 microseconds and a 95th percentile around 50 microseconds.
Those measurements exclude model loading, synthesis and transport, and are
host dependent. The JSON evidence is authoritative for a specific run.

## Real digit classification experiment

The bundled public scikit learn digits dataset was partitioned once with
seed 42: 988 training, 269 calibration, 270 edit and 270 untouched test cases.
Training data selected 60 varying raw integer pixel features. One ridge
classifier was fitted and quantized to int8 weight values. Calibration alone
selected scale 32 using a predefined rule: choose the coarsest listed scale
within one percentage point of the scale 1 calibration accuracy.

All three methods used the same starting model, initial edit pool errors and
single coordinate search class. Guarded methods added successful target code
vectors to their protected set after each edit. The cells method used actual
Rust returned models with an independent Python solver and prediction check.
The two controls used the Python implementation. The comparator is a strict
single coordinate accumulator preservation baseline, not a reproduction of
AlphaEdit or a complete multiweight null space algorithm.

| Original anchors | Method | Accepted edits | Retained target labels | Changed anchors | Untouched test correct |
|:--|:--|:--|:--|:--|:--|
| 4 | Accumulator preservation | 4 | 4 | 0 | 137/270 |
| 4 | Output cell preservation | 9 | 9 | 0 | 249/270 |
| 4 | Target only | 15 | 6 | 4 | 157/270 |
| 16 | Accumulator preservation | 2 | 2 | 0 | 251/270 |
| 16 | Output cell preservation | 5 | 5 | 0 | 252/270 |
| 16 | Target only | 15 | 6 | 16 | 157/270 |
| 64 | Accumulator preservation | 2 | 2 | 0 | 251/270 |
| 64 | Output cell preservation | 2 | 2 | 0 | 251/270 |
| 64 | Target only | 15 | 6 | 64 | 157/270 |

Baseline test accuracy was 251/270, or 92.96%. Both guarded methods retained
the entire output vector on every accepted target through the sequence, and
all original protected output vectors were unchanged after every edit.

With 16 anchors, cells admitted 2.5 times as many retained corrections as the
strict comparator: five versus two. This is a count on one small experiment,
not a general speed, accuracy or capability multiplier. At 64 anchors the
advantage disappeared. The 64 anchor feature matrix has rank 50 out of 60,
so these data do not prove that a general multiweight null space is empty.

## Negative results and business implications

The four anchor cell run harmed two previously correct unseen test examples.
The strict accumulator baseline harmed 114, despite preserving every anchor.
The target only control lost earlier repairs and damaged all anchor sets.
These are strong warnings against advertising finite example certificates
as model wide preservation.

The 16 anchor cell run helped two untouched test cases and harmed one,
for a net change from 251 to 252 correct. This is insufficient for statistical
superiority. We did not tune on that test split or hide the less favorable
protection sizes.

Training the tiny baseline ridge model took approximately 2.2 milliseconds
in the retained evidence run. Cumulative Rust synthesis was approximately 12 to 67
milliseconds depending on the anchor count. The experiment does not show
that this workflow is cheaper than retraining. A cost matched retraining
study remains necessary, especially for larger networks.

The candidate value for ruv-FANN is a checkable patch admission component
for small integer models. A deployment host could use it to admit bounded
parameter changes and reject changes that violate explicitly protected
outputs. Actual device kernels, persistent revision state, authorization,
target objective binding and disjoint quality gates must be added by that host.

## Verification evidence

* 24 Rust tests cover boundaries, transaction behavior, CLI work limits,
  753 complete tiny search comparisons and the 100 patch construction.
  The solver test generates 1,000 cases; 247 have varying baseline box
  outputs and are excluded from the constant output box comparison.
* The independent Python acceptance script agrees on all 400 additional
  search fixtures, including 63 constant output boxes. It independently
  verifies 165 accepted patches and rejects 165 altered certificates.
* The complete digit study cross checks 67 Rust feasibility decisions, with
  16 feasible results and identical selected candidates. A portable frozen
  replay contains all 16 successes and 12 unsuccessful cases.
* Python theory checks cover 590,112 cell equivalences, 72,079 complete
  weight vectors over 35 capacity configurations, 650,000 directional
  memberships and 1,445 affine separator configurations.
* Seventeen Lean declarations compile with Lean 4.19.0 and only standard
  foundation axioms. The audit rejects unchecked placeholders or custom axioms.
* The scoped Ruflo 3.25.6 security scan reported zero findings. Independent
  source and boundary review found no high or critical defect. The scanner is
  not a Rust dependency advisory audit or proof of complete security.

The Lean proof does not verify the Rust implementation, an arbitrary scale
capacity theorem in full, or machine code. The implementation supports
arbitrary positive integer scales under its checked arithmetic limits.

## Reproduction and next decision

See the [crate instructions](../README.md), `evidence/*.json`, the frozen
digit inputs and the source manifest. CI runs the Rust gates, independent
acceptance, frozen digit replay, Python theory checks and pinned Lean audit.
Independent ruOS execution is recorded separately from local measurements.

Decision: retain an experimental module and a candidate research claim.
Next action: test one existing quantized sensor or routing model with its
actual deployment arithmetic and a fixed, unseen retention set. The largest
risk is preserving a narrow anchor list while harming unrepresented behavior.
Mitigate through covered input regions where tractable and an independent
distributional retention gate.

Acceptance for the next stage: produce at least one retained useful correction,
zero violations of the declared protected contract, no breach of a predefined
unseen quality limit, exact rollback on the real runtime, and measured full
adaptation cost against the matched retraining control.
