# Bounded MetaHarness optimization

Use the selected checkout's private `tools/metaharness` package. It pins
`metaharness` 0.4.21 and `@metaharness/darwin` 0.13.0 in its own lockfile. The
reviewed upstream source is `ruvnet/metaharness` commit
`b27878cf39d3ada6b2c4d1bcc3e9258a70c6cfd6`. A source review commit is not proof
of the bytes in a published npm package; retain npm integrity and installed
distribution hashes separately.

## Inspect and install within the authorized task

```sh
npm ci --prefix tools/metaharness --ignore-scripts --omit=optional
node tools/metaharness/src/cli.mjs plan --repo .
npm test --prefix tools/metaharness
```

POSIX (Linux or macOS) and Node 22.13+ are required. Installation affects only
this private tool. Lifecycle
scripts and unused optional native/model packages are omitted. Do not update the
root package graph or substitute an unpinned installer. Package identity comes
from installed metadata and hashes; this MetaHarness CLI does not implement a
reliable `--version` check.

## Run a real, bounded loop

```sh
node tools/metaharness/src/cli.mjs optimize --repo . --variant simd --model medium --baseline-batch-rows 128 --generations 2 --children 2 --seed 7 --samples 20 --total-rows 512 --output /tmp/ruv-fann-metaharness
```

Use `--baseline-batch-rows` for the current setting (1–128, default one). Retain
that baseline if a different candidate does not clear fresh confirmation.
Choose a new external directory. The supported presets are `tiny`, `medium` and
`large`, with ordinary (`core`) or SIMD WASM. Precision is Float64 and activation
is Linear. Search changes only the integer `batchRows` parameter from 1 to 128;
the model, weights, row-specific inputs and total rows stay fixed. Last partial
batches use their exact active lengths. Every candidate must match the legacy
reference output, including signed zero, before timing.

The tool calls real `evolveNumeric`, not an imitation of Darwin. It limits
generations to three, children to two, concurrency to one, and independently
bounds dispatches. Parent expansion matters: the maximum is
`1 + children + 2*children*(generations-1)`. Repeated genomes reuse a verified
observation and retain that observation's identity in the archive. Do not count
cache hits as independent timing measurements.

The selected search maximum is only a candidate. Fresh interleaved paired blocks
compare it against the baseline. A recommendation requires the 95% bootstrap
lower bound to exceed 1.05. A valid `not_improved` result retains the baseline;
it is not an infrastructure failure. Do not require a positive speedup in CI.

Read `receipt.json`, archive/trial data and command logs. Require real artifact
and source identities, finite positive timing samples, equal workload/output
hashes, zero output error, bounded dispatches, final input verification, and fresh
confirmation evidence. Source-aware WASM receipt verification runs before and
after the experiment. Tool policy/evaluator sources are fingerprinted too.

## Interpret measurements

Timings cover the entire fixed-row workload through public JavaScript checks,
copies and tail calls. Model/workspace construction, reference checks and fixture
generation are outside the hot timing block. One observed construction cost and
estimated storage are separate fields. Nanosecond values are derived from
`performance.now()` milliseconds, not a claim of nanosecond clock resolution.

Bootstrap intervals describe sampled timing-block variation on this host and
workload. They do not include drift across days, other models/hardware, cold
starts, GPU execution or forecasting accuracy. Record actual runtime, host,
variant, model, row count and compiler/artifact identities before comparing runs.

`score --repo . --output NEWDIR` invokes MetaHarness's actual static scorer on a
bounded metadata projection. Its heuristics and cost estimates are separate from
build/runtime/performance evidence. Do not add fake files or change safety
settings to improve a heuristic score.

## Boundaries and CI

The child runs with a deadline, output cap, sanitized environment and process
group cleanup. No arbitrary evaluator shell command, provider call, source edit,
configuration auto-apply, promotion or publication is exposed. This is not an OS
sandbox for hostile code. Existing host trust and repository permissions apply.

The standalone/reusable MetaHarness workflow installs the exact lock, runs
boundary tests, executes actual core/SIMD trials, validates evidence and preserves
the checkout. Shared-runner CI checks correctness and bounded execution, not a
hardware-specific latency target. Rune remains the separate native frozen-build
gate; this loop does not replace it or the full package release checks.
