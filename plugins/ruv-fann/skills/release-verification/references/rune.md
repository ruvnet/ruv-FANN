# Pinned Rune builds and CI evidence

Use the selected checkout's `scripts/rune.py` and `config/rune.json`. The package
is **rune-toolchain**, binary **rune**, from `https://github.com/ruvnet/rune`.
The Rune scripting language uses the same binary name. A matching `--version`
or a binary on PATH is insufficient; this helper requires its own pinned-source
build receipt and checks the executable SHA-256.

The current pin is version 0.2.0, commit
`92425512f0f742e21b747f58f7555104446fc9de`. Read the live configuration before
execution and stop if it differs from the intended reviewed pin. The tool needs
Rust 1.88+, Python 3.11+ and POSIX. This does not raise the core library's MSRV.

## Inspect and prepare

```sh
python3 scripts/rune.py plan --repo .
python3 scripts/rune.py prepare --repo . --output /tmp/ruv-rune-prefetch
```

Choose new task-owned external directories; preserve prior receipts. `plan`
does not execute code. `prepare` copies the actual unmodified root Cargo/source
inputs under `workspace/core/` and a private executable probe at `workspace/`.
It excludes private coordination state and rejects symlinks, escaping manifests,
unsupported Cargo config and oversize inputs. Both the probe and core sources
are inside Rune's digest boundary. The probe lock binds its resolved dependency
graph; `core/Cargo.lock` separately retains the library's original graph.

## Explicit setup

When dependency setup is within the task's authorization, clone/fetch the exact
configured commit into a separate directory. Check its HEAD and Cargo.lock hash
against `config/rune.json`. Never use an unpinned registry installer or substitute
the scripting-language package.

With that clone at `/tmp/ruv-rune-source`, prefetch the three committed graphs:

```sh
cargo fetch --locked --manifest-path /tmp/ruv-rune-source/Cargo.toml
cargo fetch --locked --manifest-path Cargo.toml
cargo fetch --locked --manifest-path /tmp/ruv-rune-prefetch/workspace/Cargo.toml
python3 scripts/rune.py setup --source /tmp/ruv-rune-source --prefix /tmp/ruv-rune-tool --timeout 1200
```

`setup` checks tracked upstream build-input bytes against Git objects, rejects
dirty/unpinned inputs, builds offline, and records source, lock, compiler and
binary identity in `tool-receipt.json`. The receipt is local evidence, not a
signed upstream attestation. Its directory and executable remain trusted host
inputs. Missing cached dependencies fail; verification never fetches a fallback.

## Execute and check actual evidence

```sh
python3 scripts/rune.py verify --repo . --tool-receipt /tmp/ruv-rune-tool/tool-receipt.json --output /tmp/ruv-rune-evidence --timeout 1200
```

The helper freezes a model-free plan, replays it, verifies a clean rebuild and
records Rune's explanation. It independently checks plan identity, compiler,
target, source/lock digests, offline/model-call receipt fields, expected product
name, executable SHA and the stored content-addressed object. It runs that exact
executable, which calls production f32/f64 batch and calibration APIs, then runs
release-source tests for numerical behavior, allocations and work budgets.

Read `receipt.json`, command arguments, exit codes and bounded logs. Success
requires status `passed`, a passing runtime probe and correctness result, and
all three rejection checks: `stale_source`, `plan_tamper`, `object_tamper`.
Recheck unchanged original locks and input identities. A timeout, missing tool,
truncated output or unexecuted prerequisite is a failure. Do not overwrite a
failed receipt; fix the cause and use a new evidence directory.

The helper uses a separate Cargo home with cached dependencies, omits ambient
credentials/configuration/flag overrides, and bounds process groups, time and
output. Build scripts and proc-macros still execute trusted host code. Source
copying and Cargo offline are not an operating-system sandbox.

## CI/CD and scope

`.github/workflows/rune.yml` runs relevant push/PR checks and is reusable from
the full acceptance workflow. Setup is explicitly separated from offline
verification. Failure propagates; JSON/log evidence is uploaded even on failure.
The job has read-only repository permission and no model/provider secret.

Reproducing one selected native product is not verification of every artifact,
other package families, browser/WASM execution or peak memory. Performance needs
the separate benchmark protocol. This integration does not invoke `rune improve`,
edit code, sign/promote a candidate, publish a release or create a trust anchor.
