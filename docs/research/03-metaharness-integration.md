# Metaharness Integration Analysis for ruv-FANN

**Status**: Research report (swarm: metaharness integration analyst)
**Date**: 2026-08-08
**Source repo reviewed**: `ruvnet/metaharness` (shallow clone at `/workspace/ruvnet/metaharness`)
**Target repo**: `ruvnet/ruv-FANN` (`/home/user/ruv-FANN`)

---

## 1. Metaharness architecture

Metaharness ("Mint a custom AI agent harness from any repo") is **not an agent
framework — it is a factory for agent frameworks**. `npx metaharness <name>`
scaffolds a repo-aware harness: agents, skills, slash commands, MCP server,
scoped memory namespace, governance policy, witness-signed provenance, and an
npm-publishable CLI. The model is treated as replaceable; the harness is the
product (`README.md`).

### 1.1 Three-layer model (`docs/ARCHITECTURE.md`)

| Layer | Contents |
|---|---|
| 3 — user surface | `packages/create-agent-harness` (the `metaharness` CLI, `npx metaharness`), `harness` subcommands, `.claude-plugin/plugin.json` |
| 2 — adapters/app | 9 host adapters (`packages/host-claude-code`, `host-codex`, `host-copilot`, `host-github-actions`, `host-hermes`, `host-openclaw`, `host-opencode`, `host-pi-dev`, `host-rvm`), `packages/sdk`, `packages/vertical-base`, `packages/vertical-trading` |
| 1 — kernel | Rust core (`crates/kernel`: claims, hooks, intel, mcp, memory, routing, witness, federation) with `crates/kernel-wasm` (wasm-bindgen) and `crates/kernel-napi` (NAPI-RS) targets, loaded via `packages/kernel-js` (native → wasm → js fallback) |

Layer 1 never imports Layers 2/3 (ADR-002 boundary). ADRs are the design record:
245+ files in `docs/adrs/` (`INDEX.md`, "append, do not renumber").

### 1.2 Key packages

| Package | Path | Role |
|---|---|---|
| `metaharness` / `create-agent-harness` | `packages/create-agent-harness/` | Harness generator CLI: scaffold, `--wizard`, `analyze-repo`, `score`, eject, federate, genome tooling |
| `@metaharness/darwin` | `packages/darwin-mode/` | Darwin Mode: gradient-free evolution of harness policy files (ADR-070…146) |
| `@metaharness/router` | `packages/router/` | Cost-optimal model routing from eval logs: k-NN + trained KRR + optional native FastGRNN (ADR-040/043) |
| `@metaharness/flywheel` | `packages/flywheel/` | The promotion loop as a library: run → measure → mutate → verify → promote, Ed25519 receipts, replayable lineage |
| `@metaharness/weight-eft` | `packages/weight-eft/` | LoRA distillation of the cheap tier from the agent's own solved trajectories (ADR-198) |
| bench / DRACO | `packages/bench/draco/` | Cross-domain deep-research benchmark; emitted the routing dataset (`runs/routing-dataset.json`) that trained the router (ADR-037…040) |
| `kimi-k3-harness` | `kimi-k3-harness/` | Flagship *generated* harness (for kimi-k3-in-c): architect/implementer/reviewer/test-writer agents, code-index MCP, plus Darwin + Flywheel wired to a Rust→WASM kernel bench (measured 5.07x GOPS lift, `kimi-k3-harness/README.md`) — the closest existing analog to what a ruv-FANN harness would be |
| `score` command | `packages/create-agent-harness/src/repo-scorecard.ts`, `score.ts` | `npx metaharness score <repo> [--json]` — ADR-041 scorecard: harness fit, build likelihood, tool safety, estimated cost/run; reads the repo, never executes it |
| evals | `packages/evals-extract`, `evals-hle`, `evals-math`, `evals-sql`, `evals-toolcall`, `evals-servedmodel` | Domain eval adapters that feed the flywheel/router |

`kimi-k3-harness` demonstrates the full stack on a numerics repo: `npm run
build:wasm` (a Rust int8-matvec kernel replica), `npm run flywheel` (policy
tuning with a frozen conjunctive gate: ≥2% lift, no cost regression,
correctness vs f64 golden reference, anchor shape must not regress), `npm run
evolve` (Darwin real-sandbox). This is the template for ruv-FANN: **treat a
performance-critical inner loop as the fitness target**.

---

## 2. Darwin Mode deep dive (`packages/darwin-mode/`)

Lineage: Darwin Gödel Machine — mutate the *harness source*, empirically
validate each variant; model weights stay frozen (ADR-070). Dependency-free,
Node >= 20 built-ins only.

### 2.1 The loop (`src/evolve.ts`, 544 lines)

```
profile → baseline → (mutate → sandbox → score → archive)* → promote/select
```

- **Profile** (`src/repo_profiler.ts`): distils the repo into a `RepoProfile`
  — package manager, **test command**, source files (walks `.ts/.tsx/.js/.jsx/.json/.md`,
  skipping `node_modules/.git/.metaharness/dist`), and risk files
  (`/(\.env|secret|credential|token|key|deploy|release|infra)/i`).
- **Baseline** (`src/generator.ts`, `src/templates.ts`): emits the seven
  mutation-surface files.
- **Mutate** (`src/mutator.ts`): `DeterministicMutator` — a seeded perturbation
  of exactly ONE surface file per child (default path; no network, no key).
  LLM mutators are library-only drop-ins behind the same gate:
  `src/openrouter-mutator.ts`, `src/requesty-mutator.ts`, `src/ruvllm-mutator.ts`.
- **Sandbox** (`src/sandbox.ts`): safety gate first, then runs the profile's
  `testCommand` via `execFile` (**no shell**, argv-split) with a **scrubbed
  env** — only `PATH`, `NODE_ENV=test`, `METAHARNESS_VARIANT`, `METAHARNESS_TASK`
  pass through, so no secrets/tokens/proxy vars leak into a variant. 120s
  default timeout, 8 MiB output cap, never throws (failures become `RunTrace`s).
  Alternative substrates: `--sandbox mock` (`src/mock-sandbox.ts`, ADR-102,
  offline/fast) and `--sandbox agent` (`src/tier2-sandbox.ts`, ADR-106, executes
  the real surface code).
- **Score** (`src/scorer.ts`, ADR-072): pure function over traces.
- **Archive** (`src/archive.ts`, ADR-073): a persisted TREE (`archive.json`),
  parent→child; non-promoted variants retained, selection samples the WHOLE
  archive so a weak ancestor can seed a strong branch (escapes hill-climbing).
- **Select**: `--selection score|quality-diversity|behavioral-diversity|niche-steering|clade|pareto`
  (`src/phenotype.ts`, `src/clade.ts`, `src/pareto.ts`), plus `--crossover`
  (ADR-089), `--epistasis` (learned linkage, `src/epistasis.ts`, ADR-093),
  `--risk-budget` (SGM, ADR-090), `--fdr` (Benjamini-Hochberg promotion gate,
  ADR-096), `--curriculum` (`src/curriculum.ts`, ADR-097).

CLI: `metaharness-darwin evolve <repo> [--generations 3] [--children 4]
[--concurrency 4] [--seed 0] [--bench suite.json] ...`. Artifacts land in
`<repo>/.metaharness/` (`archive.json`, `lineage.json`, `variants/`, `runs/`,
`reports/winner.json`).

### 2.2 The seven mutation surfaces (the "genes", `src/safety.ts`)

A variant directory may contain ONLY these files (`FILE_BY_SURFACE` /
`APPROVED_FILES`), each pure policy logic over injected data:

| Surface | File | Governs |
|---|---|---|
| `planner` | `planner.ts` | task → ordered plan steps |
| `contextBuilder` | `context_builder.ts` | ranks candidate files vs the task |
| `reviewer` | `reviewer.ts` | flags changed files vs risk list + test outcome |
| `retryPolicy` | `retry_policy.ts` | whether/how to retry on failure class |
| `toolPolicy` | `tool_policy.ts` | allow-list + ordering over command kinds |
| `memoryPolicy` | `memory_policy.ts` | is an outcome worth remembering |
| `scorePolicy` | `score_policy.ts` | *proposed* score weights (the frozen kernel scorer decides promotion — a variant can never re-grade itself) |

Beyond file surfaces, the SWE-bench work evolves a **structured config genome**
(`packages/darwin-mode/bench/swebench/evolve-config.mjs`, ADR-184/187/188):
`{ mode: single|cascade|ecascade|xbo|xcascade|bo3, baseModel, escalateModel,
maxSteps, temp }` — with real resolve-rate fitness and an ADR-072 cost breaker.
This config-genome pattern (not the seven-file pattern) is the right shape for
evolving ruv-swarm/claude-flow configs.

### 2.3 Safety gate (`src/safety.ts`, ADR-071)

Two independent checks, both code: `inspectVariant(dir)` before any execution
(only the 7 approved filenames; no nested dirs/symlinks; blocked-filename
substrings including `.env`, `secret`, `token`, `id_rsa`, `package.json`,
lockfiles; caps: 32 files, 256 KB/file) and `validateGeneratedCode()` before
LLM-generated code is written. Blocked content patterns reject `process.env`
(incl. computed forms), `child_process`/spawn/exec, dynamic
`require()`/`import()`, `eval`/`new Function`, `fetch`/XHR/WebSocket,
restricted `node:` builtins, prototype/global escapes, shell strings
(`curl|wget|ssh|sudo|chmod|rm -rf`...), and secret handling. A disqualified
variant never runs (reserved exit code 99) and scores `safetyScore 0`.

### 2.4 Fitness and promotion (`src/scorer.ts`, ADR-072)

```
baseScore  = 0.35*taskSuccess + 0.20*testPassRate + 0.15*traceQuality
           + 0.10*costEfficiency + 0.10*latencyEfficiency + 0.10*safetyScore
finalScore = baseScore - 0.30*secretExposure - 0.25*destructiveAction
           - 0.20*hallucinatedFile - 0.15*toolLoop - 0.10*costOverrun
```

Promotion requires all four: `finalScore > parent + 0.05`, `safetyScore >= 0.95`,
`testPassRate >= parent's`, zero blocked actions (`safetyScore == 1.0`).

### 2.5 Scaffold wiring (ADR-147)

Every `npx metaharness` scaffold ships Darwin by default
(`packages/create-agent-harness/src/index.ts`: `--darwin`/`--no-darwin`,
`DARWIN_VERSION`): it injects `devDependencies["@metaharness/darwin"]`, the
scripts **`npm run evolve`** (real substrate — runs the harness's own test
command per variant) and **`npm run evolve:dry`** (mock substrate, fully
offline), and a real `.claude/skills/evolve/SKILL.md`. Default path is
air-gapped and key-free (DeterministicMutator).

Measured results (`packages/darwin-mode/README.md`, `LEARNINGS.md`,
`bench/results/RESULTS.md`): SWE-bench Lite conformant 34.0% (102/300,
~$0.005/instance, DeepSeek-V4-Flash), best-of-3 + judge ~52%, Test-Driven
Repair 68.3% when the acceptance test is given (ADR-175).

---

## 3. Router and the flywheel

### 3.1 `@metaharness/router` (`packages/router/src/`)

Dependency-free, no network, no model files — bring your own embeddings.

- **k-NN router** (`src/index.ts`): `new Router({ qualityBar, candidates })`
  where each candidate is `{ id, costPerMTok, examples: [{ embedding, quality }] }`
  from your eval logs. `route(queryEmbedding)` predicts per-candidate quality
  via k-NN and returns the **cheapest candidate predicted to clear the bar**
  (`{ id, predictedQuality, costPerMTok, metBar }`). Cached norms give 2.4k-30k
  routes/s.
- **Trained KRR router** (`src/train.ts`, ADR-043): `trainRouter(rows, prices,
  { qualityBar })` with `rows: [{ embedding, scores: { modelId: quality } }]`.
  Kernel ridge regression, cosine kernel, lambda fit by leave-one-out CV —
  the regularised generalisation of k-NN that wins as the corpus grows.
  Serialises to portable JSON (`toJSON()`/`fromJSON`).
- **Native FastGRNN** (`src/native.ts`): optional `@ruvector/tiny-dancer` peer
  (Rust/NAPI, `.safetensors`), same dataset, `resolveRouterBackend('auto')`.

**Data format the flywheel must produce** (the whole contract):

```json
{ "embedding": [/* d floats, any embedding model */],
  "scores": { "z-ai/glm-4.7": 0.72, "anthropic/claude-opus-4": 0.91 } }
```

plus a `prices` map `{ modelId: costPerMTok }`. DRACO's committed instance is
`packages/bench/draco/runs/routing-dataset.json`. ADR-040 defines the ladder
(always_cheap / always_frontier / router / **oracle** upper bound; report
router quality as % of oracle; the DRACO learning curve was still rising at
n≈20, so value compounds with data).

### 3.2 OpenRouter fit

OpenRouter appears as the standard multi-model access path:
`packages/darwin-mode/src/openrouter-mutator.ts` reads **`OPENROUTER_API_KEY`
from the environment** (dev fallback: `$TMPDIR/.orkey`), POSTs to
`https://openrouter.ai/api/v1/chat/completions`, tracks usage/cost telemetry,
and degrades to a safe no-op on network failure. Model IDs in router examples
are OpenRouter-style (`provider/model`). The ruv-FANN integration follows the
same rule: **key only ever via `OPENROUTER_API_KEY` env — never hardcoded,
never committed** (matches ruv-FANN CLAUDE.md security rules; note the
`/tmp/.orkey` fallback should NOT be used in ruv-FANN CI).

### 3.3 `@metaharness/flywheel` (`packages/flywheel/`)

The reusable promotion loop: `runFlywheelGenerations({ rootPolicy, proposer,
evaluator, promotionRule, holdout, anchor, maxGenerations, signer })`. Your
evaluator projects any benchmark onto `Score = { primary, noopRate, costPerWin,
regressed }`. Anti-Goodhart by construction: candidates must clear a holdout
AND a frozen **anchor** never optimized against; the gate (`meetsPromotionRule`)
is frozen and fingerprinted; every promotion is Ed25519-signed and
`verifyReplayBundle()` lets an outside auditor replay the lineage with zero
trust. `weight-eft` (`packages/weight-eft/`) is the complementary weights-side
lever: exports gold-resolved trajectories to standard SFT (OpenAI chat JSONL,
tool_calls preserved) + on-policy DPO pairs, with train/eval instance-ID
disjointness, a reward-hacking filter, and a `weightAdapter` gene
(`null|'sft'|'sft-dpo'`) that Darwin can prune if the tune overfits (ADR-198).
GPU-gated; $0 by default (emits a training plan).

---

## 4. Integration design for ruv-FANN

ruv-FANN today: Rust workspace (root `Cargo.toml`, `benches/neural_network.rs`),
`ruv-swarm/` (Rust crates + `ruv-swarm/benches/{agent_spawn,message_passing,
orchestration,wasm}_bench.rs` + `ruv-swarm/npm` JS package), `neuro-divergent`,
`cuda-wasm`, root `package.json` (`engines: node >=18.20.8`,
`ruv-swarm: file:ruv-swarm/npm`). Claude-flow config lives in `CLAUDE.md`
(topology, maxAgents, strategy, consensus, hooks).

### 4.a Mint a ruv-FANN harness

```bash
# One-time, from the repo root (no repo code is executed by analysis)
npx metaharness score . --json > docs/research/metaharness-scorecard.json   # fit/cost preview (ADR-041)
npx metaharness ruv-fann-harness --template vertical:coding --host claude-code \
    --target ./ruv-fann-harness --with-wasm ./ruv-swarm/crates/ruv-swarm-wasm
```

- Or `harness analyze-repo . --scaffold ruv-fann-harness` to derive agents/
  skills from the actual layout (`packages/create-agent-harness/src/analyze-repo.ts`).
- Keep the scaffold in its own top-level directory `ruv-fann-harness/`
  (a generated harness is a self-contained npm package, like `kimi-k3-harness/`
  in the metaharness repo), publishable later as `@ruvnet/ruv-fann-harness`
  so the whole org gets `npx @ruvnet/ruv-fann-harness`.
- Trim per README guidance: keep architect/implementer/reviewer/test-writer,
  the code-index MCP, and push-guarded git perms; delete unused verticals.
  `harness doctor` / `harness validate` keep it healthy.
- Darwin ships wired by default (ADR-147): `npm run evolve` / `npm run evolve:dry`
  work out of the box inside the scaffold.

### 4.b Wire Darwin Mode to ruv-FANN's benches

Two tiers, mirroring `kimi-k3-harness`:

1. **Policy-file evolution (default, $0, offline)** — run
   `metaharness-darwin evolve` against `ruv-swarm/npm` with the profile's test
   command = `npm test` there. Script at `/scripts/darwin-evolve.sh`:

   ```bash
   #!/usr/bin/env bash
   set -euo pipefail
   npx metaharness-darwin evolve ./ruv-swarm/npm \
     --generations "${GENERATIONS:-3}" --children 4 --concurrency 4 \
     --seed "${SEED:-0}" --tie faster --selection quality-diversity \
     "$@"   # artifacts: ruv-swarm/npm/.metaharness/{archive,lineage,reports}
   ```

   Add `.metaharness/` to `.gitignore` except `reports/winner.json`.

2. **Config-genome evolution (the real prize)** — adapt
   `packages/darwin-mode/bench/swebench/evolve-config.mjs` into
   `/scripts/darwin-evolve-config.mjs` with a **ruv-swarm genome**:

   ```
   GENOME  = { topology: hierarchical|mesh|hierarchical-mesh,
               maxAgents: 4..15, strategy: specialized|balanced,
               consensus: raft|byzantine, model tiers per agent role,
               hookCadence, memoryNamespace policy }
   FITNESS = ruv-FANN bench outcomes: cargo bench (benches/neural_network.rs,
             ruv-swarm/benches/*) throughput deltas + `npm test` pass rate in
             ruv-swarm/npm + wall-clock/cost — projected onto the ADR-072
             weighted score; promotion via the flywheel's frozen gate with
             holdout = a held-out bench set, anchor = a pinned criterion
             baseline JSON (never optimized against).
   ```

   Winning genomes are materialized as `/config/swarm-genome.json` (the
   authoritative evolved config) and the values mirrored into `CLAUDE.md`'s
   Project Config block by a human-reviewed PR (ADR-166 human review gates:
   config that steers agents should not self-apply).

3. Optional LLM mutator: export `OPENROUTER_API_KEY` and set
   `DARWIN_MUTATOR_MODEL` (default `google/gemini-2.5-flash` per ADR-085) —
   library path only; the default deterministic path needs no key.

### 4.c The flywheel loop

```
        ┌──────────────────────────────────────────────────────────┐
        │  1. RUN     tasks via OpenRouter (OPENROUTER_API_KEY env)│
        │  2. LOG     trajectory + eval → eval-log JSONL           │
        │  3. TRAIN   trainRouter(rows, prices) → router.json      │
        │  4. ROUTE   route(embed(task)) → cheapest adequate model │
        │  5. EVOLVE  (periodic) Darwin on the config genome       │
        └───────────────▲──────────────────────────┬───────────────┘
                        └───── promoted winners ───┘
```

Concrete pieces (all following ruv-FANN file conventions):

| Artifact | Location | Content |
|---|---|---|
| Eval logger | `/scripts/flywheel/log-eval.mjs` | Append `{ taskId, embedding, scores: {model: quality}, tokens, costUsd, ts }` per completed task to `/config/flywheel/eval-log.jsonl`. Quality = bench/test pass signal (ADR-040 style score matrix). Embeddings from any local model (metaharness uses MiniLM in-browser, ADR-025; a small ONNX/wasm embedder keeps this offline). |
| Router trainer | `/scripts/flywheel/train-router.mjs` | Reads `eval-log.jsonl` + `/config/flywheel/prices.json` → `trainRouter()` → writes `/config/flywheel/router.json` (`TrainedRouter.toJSON()`). Report LOO quality and % of oracle. |
| Route hook | `/scripts/flywheel/route-task.mjs` | `TrainedRouter.fromJSON(router.json).route(embedding)` → prints model ID; called from claude-flow's pre-task hook so spawned agents get the cheapest adequate tier (compose with the existing 3-tier routing in CLAUDE.md ADR-026: the trained router replaces the static tier table for Tier 2/3 selection). |
| Darwin cadence | `/scripts/darwin-evolve-config.mjs` (weekly/`npx @claude-flow/cli hooks` post-milestone) | Re-evolve the genome against the refreshed eval corpus; flywheel gate + anchor; emit `/config/swarm-genome.json` + signed replay bundle to `/config/flywheel/replay-bundle.json`. |
| Docs | `/docs/research/03-metaharness-integration.md` (this file), operational runbook later in `/docs/` | — |

Rules baked in: key only via `OPENROUTER_API_KEY`; eval logs contain no
secrets (Darwin's scrubbed-env sandbox already guarantees variants can't see
the key); router training is pure TS/offline; oracle/holdout kept disjoint
from training rows (`assertTrainEvalDisjoint` pattern from weight-eft).
Later, once the archive holds enough gold-resolved trajectories,
`metaharness weight-eft export` can distil the cheap tier (GPU-gated,
strictly optional).

### 4.d Order of execution

1. Scaffold + trim harness (4.a) — zero risk, no keys.
2. `npm run evolve:dry` then `evolve` on `ruv-swarm/npm` (4.b-1) — offline.
3. Stand up eval logging + router training (4.c) — needs `OPENROUTER_API_KEY`
   only at RUN time; training/routing are offline.
4. Config-genome evolution with flywheel gate (4.b-2) — after ≥ ~20 logged
   rows (the DRACO n where routing already reached 92% of oracle).

---

## 5. Risks and compatibility

- **Node versions**: every metaharness package pins `engines: node >=20`
  (workspace `package.json`, per-package engines); ruv-FANN's root
  `package.json` pins `>=18.20.8`. Run harness/darwin/router under Node 20+
  (CI matrix should add a Node 20 lane); do not lower metaharness's floor.
- **Native/toolchain deps**: the kernel resolves native → wasm → **js
  fallback** (published beta ships js — `kimi-k3-harness/README.md`), so no
  Rust toolchain is required to *use* a scaffold. Building kernel-wasm/napi
  needs `wasm-pack`/`napi` + Rust — only if ruv-FANN vendors the kernel.
  `@ruvector/tiny-dancer` (native router) is an optional peer; ADR-043 notes
  its Rust trainer had stubbed BPTT/persistence — prefer the pure-TS KRR
  router. ruv-FANN already carries `better-sqlite3` (native) — unrelated but
  the same node-gyp constraints apply.
- **Sandbox requirements**: Darwin's sandbox is process-level (`execFile`,
  no shell, scrubbed env, static safety gate) — not container isolation. It
  runs the repo's real test command, so `cargo test`/`cargo bench` fitness
  runs need the Rust toolchain present and are slow (criterion benches minutes
  per variant; use `--concurrency` low, or the `--sandbox mock` substrate for
  smoke). The SWE-bench-grade evolution used Docker + GCP fleet
  (`bench/swebench/evolve-config.mjs`: Firestore, 32-vCPU quota) — that
  infrastructure is NOT portable into ruv-FANN CI; keep config-genome fitness
  on local benches instead.
- **What won't work in CI**: (1) anything needing `OPENROUTER_API_KEY` —
  keep LLM-mutator and live-run lanes out of PR CI, or gate behind a secret
  on protected branches only (never the `/tmp/.orkey` fallback); (2) DRACO
  grounding/faithfulness dimensions (network + LLM judge) — run `--no-judge`
  offline dimensions only; (3) GPU lanes (`weight-eft train` real runs);
  (4) long criterion-bench evolution (minutes x children x generations —
  make it a scheduled/manual workflow, not per-PR). Per-PR CI can safely run:
  `harness validate`, `npm run evolve:dry` (mock, offline, deterministic),
  router training + LOO eval on the committed dataset, and
  `verifyReplayBundle` on the checked-in bundle.
- **Determinism/noise**: ADR-137/138 quantify fitness noise — bench-based
  fitness needs repeated runs or the `--fdr`/`--risk-budget` gates to avoid
  promoting noise; criterion variance on shared CI runners will be worse than
  on the reference container.
- **Licensing/versions**: MIT throughout; pin `@metaharness/darwin` (scaffold
  pins `^0.8.0`, ADR-147 shipped at `metaharness@0.2.0`) and record versions
  in `/config/flywheel/` artifacts for reproducibility.

---

## Appendix: primary sources read

- `README.md`, `docs/ARCHITECTURE.md`, `docs/USERGUIDE.md`, `docs/adrs/INDEX.md`
- Darwin: `packages/darwin-mode/{README.md,src/{evolve,safety,sandbox,scorer,mutator,openrouter-mutator,repo_profiler}.ts,bench/swebench/evolve-config.mjs}`; ADR-070/071/072/073/147/175/184
- Router: `packages/router/{README.md,src/{index,train,native}.ts}`; ADR-040/043; `packages/bench/draco/{README.md,runs/routing-dataset.json}`
- Flywheel/Weight-EFT: `packages/flywheel/README.md`, `packages/weight-eft/README.md`; ADR-198
- Generator/score: `packages/create-agent-harness/src/{index.ts,repo-scorecard.ts,analyze-repo.ts}`; `kimi-k3-harness/README.md`
