# Claude Code skills and repository plugin

The [`ruv-fann` plugin](../plugins/ruv-fann/README.md) packages maintained
repository workflows for Claude Code. It contains eight package-specific
skills, three explicitly invoked commands, a read-only reviewer agent, and a
Python standard-library helper. Its current version is `0.1.3`; installing it
does not change any Rust/npm package version.

## Start a local session

From this checkout's root:

```sh
claude plugin validate ./plugins/ruv-fann
claude --plugin-dir ./plugins/ruv-fann
```

Inside Claude Code:

```text
/ruv-fann:doctor
/ruv-fann:repo-map
/ruv-fann:verify docs
```

The folder is loaded for that session. For an unpacked standalone distribution,
pass its absolute directory to both commands. Persistent installation depends
on the installed CLI's supported marketplace workflow; inspect `claude plugin
--help` rather than editing a plugin cache or replacing your existing catalog.

This checkout also supplies a local marketplace at
[`plugins/.claude-plugin/marketplace.json`](../plugins/.claude-plugin/marketplace.json).
For an opt-in persistent project installation, run from the intended project:

```sh
claude plugin marketplace add /absolute/path/to/checkout/plugins --scope project
claude plugin install ruv-fann@ruvnet-ruv-fann --scope project
```

This registers the catalog and writes the selected project's plugin setting.
The plugin does not run those configuration changes itself. The catalog's
`./ruv-fann` source resolves from `plugins/`, not from `.claude-plugin/`. A
standalone plugin ZIP does not include that separate marketplace catalog.

The `.claude-plugin/plugin.json` manifest uses Claude Code's format. The files
are not a ChatGPT account plugin or a remote MCP service.

The helper requires Python 3.11+. `doctor` reports tool availability and package
identities; it does not install missing tools. A complete source checkout is
required to run its repository checks. Current native/browser prerequisites
remain in the relevant package README and [modernization guide](modernization.md).
Existing user permissions, MCP servers, hooks, and provider settings remain
under the user's control.

## Pick the relevant skill

| Skill | Concrete workflow |
| --- | --- |
| [repo-map](../plugins/ruv-fann/skills/repo-map/SKILL.md) | Inspect the five Cargo workspaces and choose the smallest relevant gate |
| [core-inference](../plugins/ruv-fann/skills/core-inference/SKILL.md) | Check model boundaries, immutable prepared snapshots, and CPU/browser/native compute behavior |
| [forecasting](../plugins/ruv-fann/skills/forecasting/SKILL.md) | Train supported RNN/baseline models, evaluate past-only splits, and calibrate residual intervals |
| [swarm-mcp](../plugins/ruv-fann/skills/swarm-mcp/SKILL.md) | Exercise real local orchestration metadata, WASM, and MCP framing |
| [cuda-translation](../plugins/ruv-fann/skills/cuda-translation/SKILL.md) | Verify the external compiler prerequisite and genuine source translation |
| [opencv](../plugins/ruv-fann/skills/opencv/SKILL.md) | Select portable, native C++, or generated WASM image paths |
| [security-performance](../plugins/ruv-fann/skills/security-performance/SKILL.md) | Test trust boundaries and compare measured performance with references |
| [release-verification](../plugins/ruv-fann/skills/release-verification/SKILL.md) | Verify immutable archive identities and the dependency publication order |

Each skill describes when to use it, the sources to read, the commands to run,
and what successful evidence must contain. Supporting references are loaded
only when needed. These files are scoped to this repository's implemented APIs;
they do not copy the older generic skills already present under `.claude/`.

## Current API guidance

The core skill now covers `PreparedNetwork<T>`, `InferenceWorkspace<T>`, and
`PreparationLimits`: owned immutable model parameters, separate caller scratch,
checked canonical unit biases, and fixed-capacity f32/f64 batch workspaces.
The generated WASM API exposes bounded Float64 batches with independent sample
SIMD lanes. Preparation/workspace allocation and warm inference are distinct
measurement phases; packing alone does not establish faster execution.

The forecasting skill now covers explicit `ConformalCalibrator` fitting and
application. It documents corrected residual ranks, independent series/model/
horizon groups, `as_of`/forecast-origin leakage guards, sample and memory limits,
and the statistical assumptions behind empirical coverage. This separate
calibration API does not enable automatic prediction-interval builders,
bootstrap/quantile methods, or probabilistic sampling.

## MetaHarness optimization

The security/performance skill includes a
[MetaHarness procedure](../plugins/ruv-fann/skills/security-performance/references/metaharness.md).
It uses the pinned public Darwin numeric engine to tune a supported runtime
configuration against the actual generated WASM. It does not invent a model
client, copy MetaHarness implementations, or replace the existing Rune gate.

```sh
npm ci --prefix tools/metaharness --ignore-scripts --omit=optional
node tools/metaharness/src/cli.mjs plan --repo .
node tools/metaharness/src/cli.mjs score --repo . --output /tmp/ruv-fann-score
node tools/metaharness/src/cli.mjs optimize --repo . --variant simd --model medium --baseline-batch-rows 128 --samples 20 --total-rows 512 --output /tmp/ruv-fann-optimize
```

Use new external output directories. `score` is a static heuristic assessment of
a bounded public metadata projection. Set `--baseline-batch-rows` to the current
setting, from 1 to 128; it defaults to one. `optimize` executes a different path:
real `evolveNumeric`, actual core/SIMD WASM inference, full output comparisons,
bounded candidate dispatches and new paired confirmation samples. All variants
process the same total input rows with identical weights; row grouping is the
only gene. Duplicate observations are reused without claiming extra independent
measurements. The ordinary and SIMD variants are separate experiments.

Supported presets are Linear Float64 networks: tiny (5 → 7 → 3), medium
(100 → 50 → 25 → 10), and large (784 → 128 → 64 → 10). The search range is
1–128 rows per batch. The parent bounds wall time and output, uses a scrubbed
child environment, and checks source/artifact/tool identities before and after.
Every candidate must retain exact finite reference outputs, including tails.

The fresh confirmation requires a 95% timing-block bootstrap lower speed ratio
above 1.05 for a recommendation. `not_improved` retains the baseline and is a
valid outcome. It is not a cross-machine guarantee or a forecasting-accuracy
evaluation. Construction/storage observations are separate from warm throughput.
The loop emits an archive and `receipt.json`, and never changes source or applies
the chosen setting to production.

The private tool has its own exact npm lock. It requires POSIX (Linux or macOS)
and Node 22.13+, with no lifecycle
install scripts, no unused optional native/model packages, and no provider key.
Its [standalone/reusable workflow](../.github/workflows/metaharness.yml) checks
real execution on supported Node versions without requiring a speed gain on a
shared runner. Existing Rust/npm package locks and release policies are separate.

## Rune build verification

The release and security/performance skills now include a
[detailed Rune procedure](../plugins/ruv-fann/skills/release-verification/references/rune.md).
It uses `rune-toolchain` 0.2.0 from a fixed `ruvnet/rune` commit, rather than an
arbitrary `rune` binary on PATH. This separate development tool needs Rust 1.88+,
Python 3.11+ and POSIX; the core library's MSRV remains unchanged.

First inspect the intended checkout:

```sh
python3 scripts/rune.py plan --repo .
python3 scripts/rune.py prepare --repo . --output /tmp/ruv-rune-prefetch
```

`prepare` copies the unmodified core into an isolated tree and adds a private
probe executable that calls the real inference and calibration APIs. Both are
inside Rune's source digest boundary. It excludes private agent state. Source
symlinks, escaping dependencies and unsupported configuration fail validation.

In an external task directory, fetch the exact source pin from `config/rune.json`:

```sh
git init /tmp/ruv-rune-source
git -C /tmp/ruv-rune-source remote add origin https://github.com/ruvnet/rune.git
git -C /tmp/ruv-rune-source fetch --depth 1 origin 92425512f0f742e21b747f58f7555104446fc9de
git -C /tmp/ruv-rune-source checkout --detach FETCH_HEAD
cargo fetch --locked --manifest-path /tmp/ruv-rune-source/Cargo.toml
cargo fetch --locked --manifest-path Cargo.toml
cargo fetch --locked --manifest-path /tmp/ruv-rune-prefetch/workspace/Cargo.toml
python3 scripts/rune.py setup --source /tmp/ruv-rune-source --prefix /tmp/ruv-rune-tool --timeout 1200
python3 scripts/rune.py verify --repo . --tool-receipt /tmp/ruv-rune-tool/tool-receipt.json --output /tmp/ruv-rune-evidence --timeout 1200
```

Use new directories for each attempt. Dependency fetching is an explicit setup
step; `setup` and verification use Cargo offline. The local tool receipt binds
the pinned source, compiler and binary hash. Verification freezes and replays a
plan, rebuilds in a clean target directory, checks the executable and stored
object, executes the digest-bound probe, and runs separate release-source tests.
It must also reject stale source, a modified plan and object corruption.

Read the resulting `receipt.json` and bounded command logs. Reproducibility and
correctness are distinct from performance measurements, all-package archive
verification, browser execution and registry publication. The helper makes no
model calls or automatic edits/promotions. Cargo offline controls dependency
resolution; trusted build scripts still run on the host.

The [Rune CI workflow](../.github/workflows/rune.yml) runs relevant push/PR checks
and is reused by the complete release acceptance workflow. It pins setup inputs,
propagates failures, uploads evidence and leaves the checkout unchanged. Existing
workflow enablement and publication settings remain under their existing policy.

## Bounded verification

The helper accepts commands `doctor`, `plan`, and `verify`, with options after
the command. Supply one profile and an explicit checkout path:

```sh
python3 plugins/ruv-fann/scripts/repoctl.py doctor --repo .
python3 plugins/ruv-fann/scripts/repoctl.py plan --repo . --profile inventory
python3 plugins/ruv-fann/scripts/repoctl.py verify --repo . --profile inventory --timeout 120
python3 plugins/ruv-fann/scripts/repoctl.py verify --repo . --profile docs --timeout 120
```

Profiles are `inventory`, `docs`, `core`, `swarm`, `forecasting`, `cuda`, and
`opencv`. The first two inspect the current package inventory and documentation.
The five family profiles run the repository driver's scoped workspace checks
and tests. They do not imply browser/native GPU execution, an audit, or verified
archive packaging. Consult the corresponding detailed skill for those gates.

All helper output is JSON. Exit `0` means the requested check passed, `1` means
verification failed, and `2` means input was invalid. `plan` emits exact argument
arrays and working directories without executing them. `verify` enforces a
process-group deadline and output budget, and supplies Cargo offline mode.
Missing tools/dependencies, exhausted budgets, timeouts, and failing commands
remain failures. There is no automatic retry through an unpinned installer.

Ordinary build caches/output may be created by authorized checks. The helper is
not an operating-system sandbox; run untrusted source in a suitably isolated
host. Do not use plugin instructions as a substitute for the host's filesystem,
network, provider, or approval boundaries.

## Test through ruOS and Ruflo Ai Team

A reproducible end-to-end run has distinct stages:

1. Read ruOS desktop status and recent activity. Select an owned live desktop
   using current heartbeat evidence and record its machine identity.
2. Create a private Ruflo Ai Team run with bounded tasks, owners, acceptance
   criteria, runtime, and a fixed concurrency limit. Actual platform workers
   perform implementation; Ruflo's run/task tools record coordination.
3. Transfer the exact reviewed plugin and needed package payload into a new
   isolated desktop directory. Compare SHA-256 values before execution; preserve
   the selected desktop's existing checkouts and settings.
4. Inspect installed tool help/version. Validate the plugin with the actual
   Claude Code CLI and load it through that CLI's supported flow. Discover its
   skills/commands and execute a harmless workflow. A manifest parser fixture
   alone is not host discovery evidence.
5. Run helper inventory/documentation checks against the intended checkout and
   actual package API/MCP checks where included. Record command arguments, exit
   codes, assertions, hashes, and bounded logs. Missing host authentication is
   a specific limitation of a model-backed invocation, not a passing test.
6. Review artifacts independently, complete only tasks whose acceptance checks
   passed, and export the private Ruflo evidence bundle. Match its recorded
   results to the actual command receipts and tested payload hashes.

No service endpoint or credential is shipped in the plugin. Claude Code does
not automatically inherit connected ChatGPT apps; use the host's independently
configured integrations. Inspect installed Ruflo runtime help before dispatching
any runtime command. A team record, spawned process, or "ready" provisioning
flag does not prove completed execution.

## Review and distribution

The bundled `ruv-fann:repository-reviewer` agent has only `Read`, `Grep`, and
`Glob` tools. It reports defects and evidence gaps, without writing files or
running shell commands. It does not select a different model or broaden
permissions. The three commands are explicit invocations and have no dynamic
shell snippets that interpolate untrusted arguments.

Before distribution, validate the manifest with the installed Claude Code
version, check every local Markdown link/reference, run the helper tests, and
inspect the final archive. Include only plugin files and license text. Exclude
credentials, symlinks, dependency directories, runtime state, and build caches.
Rebuild the archive after any source change and record its new SHA-256. Plugin
validation and host loading should target the same bytes that are distributed.

The plugin performs no commits, pushes, registry publication, paid-provider
configuration, desktop provisioning, or external outreach. Release verification
continues to follow the existing [release process](release-process.md). Distinct
evidence is required for publication, physical GPU performance, model accuracy,
and any claimed state-of-the-art capability.

## Execution evidence

The historical **0.1.0** [end-to-end report](claude-plugin-e2e.md) and
[machine-readable receipt](claude-plugin-e2e.json) record the actual host version,
tested file hashes, executed workflows, command results, and remaining limits.
They do not certify the changed **0.1.3** plugin payload. Use those records for the tested state; the procedure above is not itself proof
that a future installation or changed package passes.

## Format references

The package follows Anthropic's official Claude Code documentation, reviewed
2026-11-08:

- [Plugin manifest reference](https://code.claude.com/docs/en/plugins-reference)
- [Plugin components](https://code.claude.com/docs/en/plugins/components)
- [Create plugins](https://code.claude.com/docs/en/plugins/create)
- [Skills](https://code.claude.com/docs/en/skills)
- [Subagents](https://code.claude.com/docs/en/sub-agents)

The manifest uses stable metadata fields and standard discovery directories.
Literal `${CLAUDE_PLUGIN_ROOT}` references in skill/command/agent Markdown are
substituted by Claude Code when the content loads; tool-shell environments do
not automatically export that variable. Always quote the substituted path.
