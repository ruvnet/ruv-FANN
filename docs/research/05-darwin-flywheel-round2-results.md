# Darwin/Flywheel Optimization — Round 2: Island-Model Evolution Results

**Date:** 2026-08-09
**Branch:** `claude/ruvnet-sota-research-j9qmeo`
**Predecessor:** [Round 1 results](./04-darwin-flywheel-run-results.md) (null result — score-saturated suite, noise-driven promotions reverted)

## 1. What changed after round 1

Round 1 failed honestly: every task scored ~1.0 on heuristics, so Darwin selection
operated purely on cost sampling noise and "promoted" functionally inert mutations
(including disabling escalation). Round 2 made three corrections:

1. **Hardened task suite** — the 7 tasks were re-scored with stricter heuristics and
   4 harder tasks were added (complexity 0.45–0.8: FANN backprop derivative with an
   exact numeric answer, MASE computation/ranking, a concurrency check-then-act bug,
   a NaN-aware O(n) dedupe refactor). Total: 11 tasks.
2. **Raised promotion gate** — `evolution.minImprovement` 0.0 → 0.005 so tie-level
   deltas cannot promote.
3. **Island-model swarm** — three concurrent evolution runs (seeds 7, 13, 99), each
   with an isolated config copy (`.claude-flow/optimization/island-*.config.json`),
   4 generations × 4-task batches, sharing only the append-only eval log.

## 2. Does the suite discriminate now? Yes.

Evidence from the two round-2 probe runs (all 11 tasks):

| Run | Routing | Mean | Min | Max |
|---|---|---|---|---|
| `round2-cheap-probe` (everything forced to cheap tier) | 11 cheap | **0.941** | **0.722** | 1.000 |
| `round2-baseline` (genome routing, judge on) | 3 cheap / 7 mid / 1 frontier | **0.993** | 0.970 | 1.000 |

The cheap tier now measurably fails hard tasks (`bugfix-off-by-one` 0.722,
`bugfix-concurrent-withdraw` 0.815, `fann-cascade` 0.815), while genome routing
recovers to 0.993 by sending those tasks to mid/frontier. Baseline metered cost:
$0.00633 for the full 11-task judged run.

This is the key structural fact of round 2: **the initial genome's routing already
places each task on the cheapest tier that solves it** — which is precisely the
router objective. The headroom evolution could exploit was mostly gone before it
started.

## 3. Island evolution results: 12 candidates, 0 promotions

| Island | Gen 0 elite | Gen 1 | Gen 2 | Gen 3 | Gen 4 | Promotions |
|---|---|---|---|---|---|---|
| seed 7 | 0.970 | 0.970 ✗ | 0.824 ✗ | 0.970 ✗ | 0.970 ✗ | 0 |
| seed 13 | 0.970 | 0.824 ✗ | 0.971 ✗ | 0.971 ✗ | 0.972 ✗ | 0 |
| seed 99 | 0.971 | 0.969 ✗ | 0.971 ✗ | 0.971 ✗ | 0.971 ✗ | 0 |

(Fitness = meanScore − 0.3 × per-task cost × 100. All three islands ended with the
genome byte-identical to the initial config; the shared config was never touched.)

### Functional verdicts on every tried mutation

Each island classified its mutations as *functionally exercised* (changed actual
API calls/routing) vs *inert* (parameter never reached execution):

| Island | Mutation | Verdict | Outcome |
|---|---|---|---|
| 7 | `frontier.temperature 0.4→0.46` | exercised | no score/cost effect |
| 7 | `frontierThreshold 0.7→0.695` | **inert** (no task in window) | rejection was eval noise, not causal |
| 7 | `maxRetries 3→4` | inert (zero retries all run) | — |
| 7 | `frontier.maxTokens 1024→1280` | inert (cap never hit) | — |
| 13 | `frontierThreshold 0.7→0.631` | **exercised — genuinely harmful** | rerouted backprop task mid→frontier: score fell 1.0→0.5 at ~11x cost; rejected on merit |
| 13 | `cheap.temperature 0.2→0.077` | exercised | no score effect |
| 13 | `backoffBaseMs 500→625` | inert (no retries) | — |
| 13 | `escalation.scoreThreshold 0.55→0.517` + `mid.maxTokens 768→576` | inert + exercised | +0.002 (sub-gate), correctly rejected |
| 99 | `escalation.enabled true→false` | inert (escalation never fired) | −0.002 noise, rejected |
| 99 | `escalation.scoreThreshold 0.55→0.47` | inert | tie, rejected |
| 99 | `cheap.temperature 0.2→0.169` + `escalation.scoreThreshold →0.529` | exercised + inert | tie, rejected |
| 99 | `frontierThreshold 0.7→0.736` | inert (no routing change) | tie, rejected |

Two findings deserve emphasis:

- **The gate caught what round 1 missed.** Seed 99's gen-1 candidate was the exact
  mutation round 1 wrongly promoted (`escalation.enabled → false`). Under the raised
  gate and discriminating suite it was rejected.
- **The one genuinely harmful mutation was rejected on merit, not noise.** Seed 13's
  `frontierThreshold 0.631` actually rerouted a task and demonstrably hurt both
  quality and cost — the first proof that the fitness function penalizes a real
  regression. (Side observation: `anthropic/claude-sonnet-4.5` scored 0.5 on the
  strict backprop heuristic where `qwen/qwen3-coder` scored 1.0 — frontier ≠ better
  on narrowly-scored formats, which is itself an argument for learned routing.)

## 4. Statistical honesty: the noise floor

Island 7 quantified the core limitation: with a 4-task batch, **one flipped task
score moves fitness by ~0.125–0.146**, while observed cost jitter is ~0.001–0.002.
The `minImprovement` 0.005 gate therefore sits ~30x below the score-noise floor —
it reliably blocks cost-jitter promotions (as designed after round 1), but a real
score improvement needs to be enormous to clear one batch, and a lucky/unlucky
judge flip can swamp everything (seed 7 and 13 each saw one 0.824 outlier from a
single mid-tier miss + failed escalation).

Consequences for future runs (recommended, not yet applied):
- Evaluate candidates on the **full suite** (or repeated batches) instead of batch-4.
- Or average k≥3 evaluations per candidate before the accept/reject decision.
- Meter judge calls (currently unmetered) so cost-based fitness is complete.

## 5. Verdict

- **Improvement: none — proven null result, twice replicated.** Three independent
  islands, 12 candidates, 0 promotions; the initial genome is locally optimal on
  this suite within the noise floor. This confirms rather than contradicts round 1,
  now with a suite that demonstrably discriminates (cheap-probe spread 0.722–1.0).
- **Harness: proven honest.** The evolution machinery now (a) rejects cost-noise
  ties (12/12), (b) rejects a functionally harmful mutation on measured merit,
  (c) never touches the shared config without a promoted candidate, and (d) the
  round-1 failure mode is reproducibly blocked.
- **Routing thesis validated at micro-scale:** cheap-tier-only loses 5.2 points of
  quality; genome routing recovers it for +$0.005 per suite run — the
  cheapest-adequate-tier objective the metaharness router formalizes.

## 6. Convergence declaration and spend

Further evolution rounds on this suite/batch size would spend API budget sampling
noise around a saturated optimum. Declared **converged** pending a materially
harder suite or the batch-size fixes above.

| Item | Metered cost |
|---|---|
| Round 1 (baseline + 3 gens + post) | $0.00726 |
| Round 2 probes + judged baseline | ~$0.0088 |
| Island 7 / 13 / 99 | $0.0230 / $0.0223 / $0.0195 |
| **Session total (metered)** | **~$0.081** (+ ~$0.005–0.01 unmetered judge calls) |

Zero API failures across all rounds.

## 7. Raw data

- `.claude-flow/optimization/eval-log.jsonl` — 109 eval records (tags: `smoke`,
  `baseline`, `post-evolution`, `round2-cheap-probe`, `round2-baseline`,
  `darwin-gen0-elite`…`darwin-gen4`; island records interleaved, islands 7/13/99
  filterable by `"seed"` in `generations.jsonl`)
- `.claude-flow/optimization/generations.jsonl` — full mutation/acceptance history
- `.claude-flow/optimization/island-{7,13,99}.config.json` — island configs (genomes
  verified identical to `config/optimization/darwin.config.json`)

These paths are gitignored run artifacts; this report is the durable record.
