# Flywheel + Darwin Optimization Harness

A dependency-free (Node 18+ built-ins only) run → measure → mutate → verify → promote
loop for ruv-FANN, backed by OpenRouter.

- **`flywheel.js`** — runs the task suite against a tiered model list (cheap / mid /
  frontier), scores each response with per-task heuristics (optional LLM judge), and
  appends JSONL eval records to `.claude-flow/optimization/eval-log.jsonl` (gitignored).
- **`darwin-evolve.js`** — gradient-free, elitist evolution of the flywheel *genome*
  (routing thresholds, per-tier temperature / max_tokens, retry backoff, escalation
  policy, judge weight). Each candidate genome is evaluated by the flywheel on a small
  task batch; only measurable improvements are kept. History goes to
  `.claude-flow/optimization/generations.jsonl`; the winning genome is written back to
  the config.
- **`../../config/optimization/darwin.config.json`** — tiers (OpenRouter model IDs),
  the genome, mutation bounds, evolution settings, and the built-in 7-task suite
  (code transforms, bug-fix reasoning, FANN / forecasting domain Q&A).

## Setup

```bash
export OPENROUTER_API_KEY=...   # required; never committed, never printed
```

Both scripts fail fast with a clear message if the key is unset. The key is read only
from `process.env.OPENROUTER_API_KEY`.

## Usage (no npm install needed)

```bash
# Verify connectivity with one trivial call to the cheapest tier
node scripts/optimization/flywheel.js --smoke

# Run the full suite with the current genome
node scripts/optimization/flywheel.js

# Options: --limit 3 (first N tasks), --tier cheap (force a tier),
#          --judge (blend in LLM-judge scores), --tag my-experiment

# Evolve the genome (5 generations x 4-task batches by default)
node scripts/optimization/darwin-evolve.js --generations 5

# Preview mutations without API calls or config writes
node scripts/optimization/darwin-evolve.js --dry-run --generations 3 --seed 42
```

## How the flywheel feeds Darwin

`darwin-evolve.js` imports `runSuite()` from `flywheel.js`. Per generation it clones
the elite genome, applies 1–2 bounded mutations, runs the flywheel on a task batch,
and computes `fitness = meanScore − costWeight × per-task cost`. A candidate is
promoted only if it strictly beats the elite (elitist selection), so the config can
only ratchet toward cheaper-and-better routing. Every eval record and every
generation outcome is JSONL-logged, so lift is auditable from the raw logs.

**Safety gate:** the genome is data-only and validated against a strict key whitelist
with numeric bounds; model references must match `[\w.:/-]+`. Neither script imports
`child_process` or evaluates strings — evolution can never produce anything that
executes shell commands.

## Mapping to metaharness (for a later full integration)

| Here | metaharness | Notes |
| --- | --- | --- |
| `flywheel.js` `runSuite` + JSONL logs | `@metaharness/flywheel` (`run.ts`, `receipts.ts`) | Same run→measure→log loop; metaharness adds frozen conjunctive promotion gates, holdout/anchor anti-Goodhart checks, and Ed25519-signed replayable lineage. Swap `fitness()` + the promote step for `meetsPromotionRule` when integrating. |
| `darwin-evolve.js` mutation operators + elitist keep | `@metaharness/darwin` (`evolve.ts`, `mutator.ts`, `safety.ts`) | Same "mutate one surface, keep only measured wins" lineage (Darwin Gödel Machine style). metaharness mutates whole harness policies in a sandbox; here we mutate a routing/params genome. Our `assertSafeGenome` mirrors its `safety.ts` gate. |
| Genome `routing.{mid,frontier}Threshold` | `@metaharness/router` | Router learns cheapest-model-that-clears-a-quality-bar from `{embedding, quality}` eval examples. `eval-log.jsonl` records (task, tier, score, cost) are exactly the training data a learned router needs — a later step can replace the static thresholds with `Router.route()`. |

## Outputs

- `.claude-flow/optimization/eval-log.jsonl` — one record per task run
  (`taskId`, `tier`, `model`, `score`, `heuristicScore`, `judgeScore`, `costUsd`,
  `latencyMs`, `escalations`, `responsePreview`).
- `.claude-flow/optimization/generations.jsonl` — one record per generation
  (`mutations`, `candidateFitness`, `eliteFitness`, `promoted`, `seed`).

Both live under `.claude-flow/optimization/`, which is gitignored.
