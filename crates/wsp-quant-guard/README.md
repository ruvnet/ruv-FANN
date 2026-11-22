# Certified quantized weight edits

An experimental ruv-FANN research crate for changing integer weights while
preserving specified quantized outputs exactly. The crate is an opt in reference
implementation. It is not connected to a production inference engine and is not
published to a package registry.

## Research result

Full rank protected features can rule out every exact accumulator preserving
edit while still allowing changes that preserve their quantized output codes.
This gives a useful, narrowly specified alternative to requiring a zero weight
perturbation on every protected representation.

For example, use weights `(3,2)`, scale `8`, and floor division. Protect the
inputs `(1,0)` and `(0,1)`, both of which output `0`. Replacing the weights by
`(4,4)` leaves those outputs unchanged but changes the output for `(1,1)` from
`0` to `1`. The protected feature matrix already has full rank.

This is a derived separation between two contracts, not a claim that
quantization cells, constrained repair, or null space editing were invented
here. [Prior art](docs/PRIOR_ART.md) includes AlphaEdit, BetaEdit, QNNRepair and
PREPARED. [Theory](proofs/THEORY.md) states the assumptions, capacity and minimum
support results, counterexamples, and exact formal proof scope.

## Measured evidence

* A constructed 128 dimensional integer layer accepted 100 distinct effective
  one weight patches. Python independently checked 12,900 protected outputs
  and 5,050 accumulated target outputs, all identical to their required values.
  All 100 new scalar target outputs remained satisfied. Exact rollback restored
  the original model. This is synthetic feasibility evidence, not 100 learned
  capabilities.
* On a quantized handwritten digit classifier, the cell constrained method
  retained 9, 5 and 2 corrections with 4, 16 and 64 protected examples.
  The same single coordinate search under exact accumulator preservation
  retained 4, 2 and 2. Every protected code vector remained unchanged for both
  guarded methods. More protection consumed available editing capacity.
* At 16 anchors, untouched test accuracy changed from 251/270 to 252/270:
  two test cases improved and one regressed. This net gain of one correct
  example does not establish a general accuracy gain.
  With 4 anchors, cell editing reduced it to 249/270 despite certified anchor
  preservation. No model wide safety claim follows.
* Seventeen Lean lemmas compile with the standard foundation axioms only.
  They cover selected quantizer and integer feasibility mathematics, not the
  complete Rust implementation or a deployed hardware kernel.

The [research report](docs/RESULTS.md) contains control outcomes, measurements,
limitations and reproducibility references. Evidence JSON records exact source
and input fingerprints. Timing is a local measurement, not a hardware promise.

## Execute

From the repository root, using Rust 1.88 and Python 3.12:

```bash
cargo test --release --locked --manifest-path crates/wsp-quant-guard/Cargo.toml
cargo build --release --locked --manifest-path crates/wsp-quant-guard/Cargo.toml
python3 crates/wsp-quant-guard/scripts/acceptance.py \
  --binary crates/wsp-quant-guard/target/release/ruv-fann-wsp-quant-guard \
  --output /tmp/wsp-acceptance.json
python3 crates/wsp-quant-guard/proofs/verify_theory.py
python3 crates/wsp-quant-guard/proofs/check_lean.py /path/to/lean
```

The pinned proof compiler is Lean 4.19.0. The proof imports only its standard
library. It does not download Mathlib or use a model API.

Reproduce the saved digit tasks without installing a machine learning library:

```bash
python3 crates/wsp-quant-guard/scripts/digits_experiment.py \
  --replay-only crates/wsp-quant-guard/evidence/digits-replay.jsonl \
  --rust-cli crates/wsp-quant-guard/target/release/ruv-fann-wsp-quant-guard \
  --output /tmp/wsp-digits-replay.json
```

The complete digit study additionally uses NumPy and scikit learn; see the
version record in `evidence/digits-local.json`. It uses bundled public data,
fixed stratified partitions, and calibration only to select the output scale.
The test partition never participates in editing or scale selection.

## Execution contract

Each row implements `clamp(floor((w dot h + bias) / scale), low, high)`.
The scale is a positive integer. Weights are restricted to a declared subset
of int8. Features and biases are int32. Every product and every ordered
accumulation prefix must fit int32. Wide checked arithmetic verifies that
contract; wraparound is rejected. The current patch API changes weights only.

Protection can describe finite feature vectors or integer boxes. An admitted
box must have a constant baseline output vector. Affine extrema then prove
that every integer point in that box remains inside the same output cells.

The sparse solver is complete only for one changed weight within its declared
resource limits. It searches exact integer intervals rather than rounding a
float solution. A `found:false` result does not rule out a multiweight repair.

The JSONL binary exposes `predict`, `solve`, `check`, `verify`, and `sequence`.
It reads bounded JSON and emits one JSON response per line. It has no network
client, arbitrary command execution, or automatic model file writes.

## Trust and integration boundaries

Certificates bind the baseline, current parent, patch, candidate, protected
domains, quantization parameters and caller supplied prefix identity. Applying
a patch recomputes preservation before changing state. A rejected patch leaves
the ledger unchanged. Rollback restores weight values while advancing the
revision, so an old patch remains stale within that ledger.

Hashes identify artifacts; they are not signatures or authorization. External
storage and cross process replay prevention belong to the caller. The prefix
identity is an assertion supplied by the caller, not proof that an external
feature generator is unchanged. Target satisfaction constrains the solver and
is independently tested; the preservation certificate itself covers only the
declared protected outputs.

Any integration must bind the real feature generator, scales, zero points,
rounding, downstream parameters, device kernel and revision store. The reference
floor quantizer is not interchangeable with a backend that rounds to nearest.
No ESP32, Hailo, transformer, or complete neural runtime is certified here.

## Decision and acceptance

Keep the component experimental. The next useful gate is a device matched
comparison on an existing small quantized network, with a disjoint retention
set and a matched retraining control. Broad novelty, lower total adaptation
cost, and general preservation remain unestablished.

Acceptance: the Rust tests, independent Python oracle, saved digit replay and
Lean audit must pass; altered certificates and stale patches must fail; every
reported retained correction must survive all subsequent accepted patches;
rollback must restore the original model exactly.
