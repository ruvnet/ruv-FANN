# Darwin/Flywheel Live Optimization Run — Results

**Date:** 2026-08-08 (UTC) · **Repo state:** `36206a1` · **Runner:** optimization-runner agent

This documents one complete live (API-calling) pass of the flywheel + Darwin harness:
baseline eval → 3 generations of genome evolution → post-evolution re-eval. All raw
records are in `.claude-flow/optimization/eval-log.jsonl` and
`.claude-flow/optimization/generations.jsonl` (gitignored).

## 1. Setup

- **Harness:** `scripts/optimization/flywheel.js` (run → measure → log) and
  `scripts/optimization/darwin-evolve.js` (bounded mutation + elitist selection),
  config `config/optimization/darwin.config.json`.
- **Provider:** OpenRouter. Tiers: cheap = `qwen/qwen3-8b`, mid = `qwen/qwen3-coder`,
  frontier = `anthropic/claude-sonnet-4.5` (frontier was never routed to in this run).
- **Suite:** the built-in 7 tasks (2 code transforms, 2 bug-fix reasoning, 3 FANN/forecasting domain QA), complexity 0.1–0.6.
- **Scoring:** per-task heuristics blended with an LLM judge (`tier:mid`, weight 0.3) for
  the baseline and post-evolution runs (`--judge`). Evolution batches used heuristics only
  (judge disabled in the genome).
- **Commands:**
  1. `node scripts/optimization/flywheel.js --judge --tag baseline`
  2. `node scripts/optimization/darwin-evolve.js --generations 3 --batch 3 --seed 42`
  3. `node scripts/optimization/flywheel.js --judge --tag post-evolution`
- **Failures:** none. Every API call succeeded on the first attempt (`attempts=1`,
  0 escalations everywhere); no retries or backoff were exercised.

## 2. Baseline (`--judge --tag baseline`)

**Mean score 0.987 · total metered cost $0.002714 · 0 escalations**

| Task | Tier | Model | Score | Heuristic | Judge | Cost (USD) | Latency (ms) |
|---|---|---|---|---|---|---|---|
| transform-var-const | cheap | qwen/qwen3-8b | 1.000 | 1.0 | 1.00 | 0.000073 | 6,863 |
| transform-callback-async | cheap | qwen/qwen3-8b | 1.000 | 1.0 | 1.00 | 0.000439 | 43,711 |
| bugfix-off-by-one | mid | qwen/qwen3-coder | 1.000 | 1.0 | 1.00 | 0.000159 | 6,886 |
| bugfix-float-compare | mid | qwen/qwen3-coder | 0.970 | 1.0 | 0.90 | 0.000658 | 10,200 |
| fann-activation | cheap | qwen/qwen3-8b | 1.000 | 1.0 | 1.00 | 0.000111 | 12,067 |
| fann-cascade | mid | qwen/qwen3-coder | 0.985 | 1.0 | 0.95 | 0.001024 | 10,738 |
| forecast-lookback | mid | qwen/qwen3-coder | 0.955 | 1.0 | 0.85 | 0.000250 | 3,885 |

Routing with the initial genome (mid ≥ 0.35, frontier ≥ 0.7): 3 tasks → cheap,
4 tasks → mid, 0 → frontier. Heuristics were saturated (1.0 on every task); all
score variation came from the judge.

## 3. Evolution (`--generations 3 --batch 3 --seed 42`)

Batch = first 3 tasks (transform-var-const, transform-callback-async, bugfix-off-by-one).
Fitness = meanScore − 0.3 × (per-task cost × 100). Gen 0 evaluates the current elite.

| Gen | Mutation(s) | Mean score | Batch cost | Fitness | Accepted? |
|---|---|---|---|---|---|
| 0 (elite) | — | 1.000 | $0.000636 | 0.994 | (baseline elite) |
| 1 | retry.maxRetries 3→4; escalation.scoreThreshold 0.55→0.498 | 1.000 | $0.000663 | 0.993 | rejected |
| 2 | routing.midThreshold 0.35→0.37 | 1.000 | $0.000495 | 0.995 | **promoted** |
| 3 | retry.maxRetries 3→2; escalation.enabled true→false | 1.000 | $0.000418 | 0.996 | **promoted** |

**Fitness trajectory:** 0.994 → 0.993 (rej) → 0.995 → 0.996.

**Critical observation:** every candidate scored a perfect 1.000 on the 3-task batch, so
all fitness differences are pure cost differences — and the metered cost of an identical
genome varies run-to-run with sampled token counts (the same 3 tasks cost $0.000636 at
gen 0 and $0.000418 at gen 3 under mutations that could not have affected those calls).
The two promotions were driven by cost sampling noise, not by measured quality or true
cost improvement:

- **Gen 2** (midThreshold 0.35→0.37) changes the routing of *no task in the entire
  7-task suite* (no task has complexity in [0.35, 0.37)). It is functionally a no-op.
- **Gen 3** (maxRetries 3→2, escalation disabled) had no measurable effect in-run
  (no retries happened; no score fell below the 0.55 escalation threshold). Disabling
  escalation does remove the quality safety net for future harder tasks, so this is
  arguably a slight *regression* in robustness that the fitness function cannot see
  on a batch where everything scores 1.0.

## 4. Genome diff (initial vs. evolved)

| Genome key | Initial | Evolved | Functional effect on this suite |
|---|---|---|---|
| routing.midThreshold | 0.35 | 0.37 | none (no task in [0.35, 0.37)) |
| routing.frontierThreshold | 0.7 | 0.7 | — |
| tierParams (all tiers) | temp 0.2/0.3/0.4, maxTokens 512/768/1024 | unchanged | — |
| retry.maxRetries | 3 | 2 | none observed (no retries occurred) |
| retry.backoffBaseMs | 500 | 500 | — |
| escalation.enabled | true | **false** | none observed here; removes safety net for low-scoring answers |
| escalation.scoreThreshold / maxEscalations | 0.55 / 1 | unchanged | — |
| judge.* | enabled=false, tier:mid, weight 0.3 | unchanged | — |

The evolved genome **was written back** to `config/optimization/darwin.config.json` and
left in place. Note the git diff of that file is much larger than the table above:
the writeback re-serializes the whole file with `JSON.stringify(…, null, 2)`, so most
of the diff is formatting-only (arrays exploded to multi-line, `3.0`→`3`, `0.0`→`0`).
The only semantic changes are the three bolded/first rows above: midThreshold,
maxRetries, escalation.enabled.

## 5. Post-evolution check (`--judge --tag post-evolution`)

**Mean score 0.994 · total metered cost $0.002330 · 0 escalations**

| Task | Tier | Score | Judge | Cost (USD) |
|---|---|---|---|---|
| transform-var-const | cheap | 1.000 | 1.00 | 0.000074 |
| transform-callback-async | cheap | 1.000 | 1.00 | 0.000474 |
| bugfix-off-by-one | mid | 1.000 | 1.00 | 0.000180 |
| bugfix-float-compare | mid | 1.000 | 1.00 | 0.000320 |
| fann-activation | cheap | 1.000 | 1.00 | 0.000120 |
| fann-cascade | mid | 0.985 | 0.95 | 0.000927 |
| forecast-lookback | mid | 0.970 | 0.90 | 0.000235 |

### Baseline vs. post-evolution

| Metric | Baseline | Post-evolution | Delta |
|---|---|---|---|
| Mean score | 0.987 | 0.994 | +0.007 |
| Total metered cost | $0.002714 | $0.002330 | −$0.000384 (−14%) |
| Tier mix | 3 cheap / 4 mid | 3 cheap / 4 mid | identical |
| Escalations | 0 | 0 | — |

**Honest read: this is a null result.** The evolved genome routes every task to exactly
the same tier as the initial genome, so both deltas are run-to-run noise, not caused by
evolution. The +0.007 score delta is entirely two judge scores moving (float-compare
0.90→1.00, forecast-lookback 0.85→0.90) — within the judge's observed per-run variance.
The −14% cost delta is token-count sampling variance of the same models on the same
prompts (the same magnitude of variance appeared *between generations with identical
effective genomes* during evolution).

## 6. Cost of the whole run

| Step | API calls (answering) | Metered cost |
|---|---|---|
| Baseline suite | 7 | $0.002714 |
| Evolution (gen 0 + 3 gens × 3 tasks) | 12 | $0.002212 |
| Post-evolution suite | 7 | $0.002330 |
| **Metered total** | **26** | **$0.007256** |

Additionally, 14 LLM-judge calls (7 per judged suite, `qwen/qwen3-coder`) are **not**
included in the metered `costUsd` (the harness only meters the answering model's usage).
At mid-tier pricing and observed prompt sizes these add an estimated ~$0.003–0.006.
**Estimated total spend for the entire workflow: roughly $0.01–0.013** — comfortably
under any budget concern, but note the judge-cost blind spot in the harness accounting.

## 7. Caveats (read before citing any number above)

1. **Null result on quality.** Evolution produced no demonstrable quality or cost
   improvement. The two "promotions" were selected on cost noise while every candidate
   scored 1.000; the surviving mutations are functionally inert on this suite (and
   `escalation.enabled=false` is plausibly worse on harder future tasks).
2. **Saturated, tiny batch.** The 3-task evolution batch contains only easy tasks
   (complexity ≤ 0.5) that every tier answers perfectly, so the fitness landscape was
   flat in the score dimension. Evolution cannot learn anything when the metric is
   saturated; a discriminating batch (harder tasks, judge enabled in-loop) is a
   prerequisite for meaningful selection.
3. **Judge noise.** Judge scores on the same task/model pair moved by ±0.05–0.10 between
   runs at temperature 0 (judge sees different sampled answers). The 0.987 vs 0.994 mean
   difference is inside this noise band. n=1 run per condition; no significance claimed.
4. **Single seed, single run.** Seed 42, one evolution run, 3 generations. No repeats,
   no confidence intervals.
5. **Elitist selection without re-evaluation of the elite** means a lucky-cheap candidate
   run beats an unlucky-expensive elite run of an equivalent genome — exactly what
   happened. This is the anti-Goodhart failure mode the metaharness promotion gates
   (holdout batches, frozen conjunctive rules, minImprovement > noise floor) are
   designed to prevent; `minImprovement` is currently 0 and should be raised above the
   observed cost-noise magnitude (≈0.002 fitness points) at minimum.
6. **Cost accounting is partial.** Judge-call and smoke-test usage are unmetered;
   prices in the config are static estimates, not billed amounts.
7. **Escalation/retry paths untested.** No call ever failed or scored below the
   escalation threshold, so mutations to those knobs were selected blind.

**Bottom line:** the harness pipeline works end-to-end (run → mutate → evaluate →
promote → writeback, fully logged, zero API failures, ~1 cent total), but this run is
evidence about the *machinery*, not evidence that evolution improved the genome. Treat
the evolved config as equivalent-with-drift, and consider reverting
`escalation.enabled` to `true` or raising `minImprovement` before longer runs.
