# ruv-FANN for Claude Code

A focused Claude Code plugin for working on this repository. It provides eight
detailed skills, three explicit commands, a read-only reviewer, and a bounded
Python verification helper. Version `0.1.4` describes the plugin, independently
of the Rust and npm package versions.

## Load it

Use a Claude Code release that supports plugins. From the repository root:

```sh
claude plugin validate ./plugins/ruv-fann
claude --plugin-dir ./plugins/ruv-fann
```

`--plugin-dir` loads this folder for that Claude Code session. It does not edit
your global plugin catalog or install a ChatGPT app. With a standalone copy,
pass the absolute path to its `ruv-fann` directory instead. Consult your installed
CLI's `claude plugin --help` for its persistent installation workflow.

For an opt-in project installation from this checkout's local marketplace:

```sh
claude plugin marketplace add /absolute/path/to/checkout/plugins
claude plugin install ruv-fann@ruvnet-ruv-fann --scope project
```

The marketplace command registers the local catalog. The install command writes
the selected project's plugin setting; run it from the project you intend to
configure. Neither command is executed automatically by the plugin. The plugin
ZIP contains the plugin folder; this repository's separate marketplace file
is `plugins/.claude-plugin/marketplace.json`.

In the session, try:

```text
/ruv-fann:doctor
/ruv-fann:repo-map
/ruv-fann:verify docs
/ruv-fann:release-check
```

The plugin can inspect another checkout when you supply its path. The checkout
must contain this repository's package manifests and verification scripts.
Read local repository instructions before any execution. Python 3.11+ runs the
bundled helper; Node, Cargo, native OpenCV, and browser dependencies are needed
only by the corresponding selected checks.

## Skills

| Skill | Use it for |
| --- | --- |
| [repo-map](skills/repo-map/SKILL.md) | Locate package families, choose a workflow, and inspect prerequisites |
| [core-inference](skills/core-inference/SKILL.md) | CPU/prepared inference, independent workspaces, checked I/O, and compute boundaries |
| [forecasting](skills/forecasting/SKILL.md) | Supported RNN/baseline training, past-only evaluation, conformal calibration, and checkpoints |
| [swarm-mcp](skills/swarm-mcp/SKILL.md) | Implemented swarm metadata, dense WASM inference, and the supported MCP surface |
| [cuda-translation](skills/cuda-translation/SKILL.md) | Translate the supported CUDA subset into Rust/WGSL with a real compiler |
| [opencv](skills/opencv/SKILL.md) | Portable matrices, native C++ integration, and browser image operations |
| [security-performance](skills/security-performance/SKILL.md) | Review trust boundaries and design measured, reproducible performance checks |
| [release-verification](skills/release-verification/SKILL.md) | Verify package payloads, source identity, audits, and release evidence |

The skill bodies load when needed; supporting references add package-specific
commands and acceptance criteria. The plugin keeps these instructions separate
from this repository's older generic Claude templates.

Version `0.1.4` adds the ruv-FANN/RuVector/MetaHarness integration procedure. It
retains bounded optimization, pinned Rune build verification, and prepared
inference instructions for native f32/f64 batches and checked Float64 WASM SIMD
batches. Calibrated interval coverage still depends on residual
assumptions and held-out evaluation. Automatic interval builders remain unsupported.
The repository's dated `docs/claude-plugin-e2e.*` receipts establish the historical
`0.1.0` host run; they do not certify the changed `0.1.4` payload.

## Commands and reviewer

`/ruv-fann:doctor` inspects manifests and available tools without changing files.
`/ruv-fann:verify <profile>` plans and executes one bounded profile, defaulting
to `inventory`. `/ruv-fann:release-check` reviews release evidence and guides
fresh archive verification when it is part of the requested work.

The `ruv-fann:repository-reviewer` subagent reads code and evidence and returns
findings. Its tools are restricted to `Read`, `Grep`, and `Glob`; it cannot run
commands or apply fixes. It inherits the host's model selection.

You can use the helper without Claude Code:

```sh
python3 plugins/ruv-fann/scripts/repoctl.py doctor --repo .
python3 plugins/ruv-fann/scripts/repoctl.py plan --repo . --profile docs
python3 plugins/ruv-fann/scripts/repoctl.py verify --repo . --profile docs --timeout 120
```

Options follow the subcommand. Profiles are `inventory`, `docs`, `core`,
`swarm`, `forecasting`, `cuda`, and `opencv`. Results are JSON on stdout; exit
codes are `0` for success, `1` for a failed check, and `2` for invalid input.
Cargo verification runs offline. Missing dependencies remain an explicit failed
prerequisite; the helper does not fetch replacements. Heavy workspace checks
can exceed the chosen deadline. Browser/GPU execution, audits, and verified
release archives have separate commands explained by their skills.

## ruOS and Ruflo Ai Team

Use ruOS for actual execution on an explicitly selected owned desktop, with an
isolated task directory, finite timeout, and verified transferred file hashes.
Use Ruflo Ai Team for private task ownership, run budgets, and evidence export.
Its task records do not themselves execute agents. Platform workers and desktop
commands produce the artifacts and exit codes that justify completion.

These connected services belong to the host where you run them. Installing this
Claude plugin does not install or authenticate ruOS or Ruflo Ai Team, and it
does not embed their credentials or invent remote MCP endpoints. The detailed
repository integration guide is `docs/claude-plugin.md` in the checkout.

## Scope

The plugin adds no hooks, global settings, provider credentials, permission
bypass flags, automatic installers, daemons, or publication actions. Verification
can create ordinary tool caches/build outputs in the selected checkout. Agent
instructions are guidance; process deadlines and input limits are enforced by
the helper, while filesystem and network isolation depend on the host.

Passing inventory or plugin validation proves only that specific check.
GPU source compilation is distinct from execution; a software GPU is distinct
from physical hardware throughput. Supported forecasting remains the actual
RNN and documented baselines. Task assignment is distinct from external model
execution. A release candidate is distinct from a published version.

MIT licensed; see [LICENSE](LICENSE).
