# Claude Code skills and repository plugin

The [`ruv-fann` plugin](../plugins/ruv-fann/README.md) packages maintained
repository workflows for Claude Code. It contains eight package-specific
skills, three explicitly invoked commands, a read-only reviewer agent, and a
Python standard-library helper. Its current version is `0.1.1`; installing it
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

## Version 0.1.1 API guidance

The core skill now covers `PreparedNetwork<T>`, `InferenceWorkspace<T>`, and
`PreparationLimits`: owned immutable model parameters, separate caller scratch,
checked canonical unit biases, and scalar packed kernels. Preparation/workspace
allocation and warm inference are distinct measurement phases; no SIMD or
XNNPACK backend claim follows from packing.

The forecasting skill now covers explicit `ConformalCalibrator` fitting and
application. It documents corrected residual ranks, independent series/model/
horizon groups, `as_of`/forecast-origin leakage guards, sample and memory limits,
and the statistical assumptions behind empirical coverage. This separate
calibration API does not enable automatic prediction-interval builders,
bootstrap/quantile methods, or probabilistic sampling.

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
They do not certify the changed **0.1.1** plugin payload. Use those records for the tested state; the procedure above is not itself proof
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
