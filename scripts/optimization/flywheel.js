#!/usr/bin/env node
'use strict';

/**
 * Flywheel eval runner for ruv-FANN (run -> measure -> log).
 *
 * Calls OpenRouter chat completions across configurable model tiers, scores
 * responses with per-task heuristics (plus an optional LLM judge), and appends
 * JSONL eval records to .claude-flow/optimization/eval-log.jsonl.
 *
 * No external dependencies (Node 18+ built-ins + global fetch only).
 * The API key is read from process.env.OPENROUTER_API_KEY and never logged.
 *
 * Usage:
 *   node scripts/optimization/flywheel.js [--config path] [--limit N]
 *        [--tier cheap|mid|frontier] [--judge] [--smoke] [--tag label]
 */

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const DEFAULT_CONFIG = path.join(ROOT, 'config', 'optimization', 'darwin.config.json');
const LOG_DIR = path.join(ROOT, '.claude-flow', 'optimization');
const EVAL_LOG = path.join(LOG_DIR, 'eval-log.jsonl');
const API_URL = 'https://openrouter.ai/api/v1/chat/completions';
const REQUEST_TIMEOUT_MS = 90_000;
const BACKOFF_CAP_MS = 8_000;

// ---------------------------------------------------------------- utilities

function getApiKey() {
  const key = process.env.OPENROUTER_API_KEY;
  if (!key || !key.trim()) {
    console.error(
      'ERROR: OPENROUTER_API_KEY is not set.\n' +
        'Export it before running, e.g.  export OPENROUTER_API_KEY=...  (never commit it).'
    );
    process.exit(1);
  }
  return key;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function loadConfig(configPath = DEFAULT_CONFIG) {
  const raw = fs.readFileSync(configPath, 'utf8');
  const config = JSON.parse(raw);
  validateConfig(config);
  return config;
}

function validateConfig(config) {
  const fail = (msg) => {
    throw new Error(`Invalid config: ${msg}`);
  };
  if (!config || typeof config !== 'object') fail('not an object');
  if (!config.tiers || typeof config.tiers !== 'object') fail('missing tiers');
  if (!Array.isArray(config.tierOrder) || config.tierOrder.length === 0) fail('missing tierOrder');
  for (const name of config.tierOrder) {
    const tier = config.tiers[name];
    if (!tier || typeof tier.model !== 'string') fail(`tier "${name}" missing model`);
    if (!/^[\w.:/-]+$/.test(tier.model)) fail(`tier "${name}" model id has invalid characters`);
  }
  if (!Array.isArray(config.tasks) || config.tasks.length === 0) fail('missing tasks');
  for (const task of config.tasks) {
    if (typeof task.id !== 'string' || typeof task.prompt !== 'string') fail('task missing id/prompt');
    if (typeof task.complexity !== 'number' || task.complexity < 0 || task.complexity > 1) {
      fail(`task "${task.id}" complexity must be in [0,1]`);
    }
    if (!task.heuristic || typeof task.heuristic !== 'object') fail(`task "${task.id}" missing heuristic`);
  }
  if (!config.genome || typeof config.genome !== 'object') fail('missing genome');
}

function appendLog(record, logPath = EVAL_LOG) {
  fs.mkdirSync(path.dirname(logPath), { recursive: true });
  fs.appendFileSync(logPath, JSON.stringify(record) + '\n', 'utf8');
}

// ------------------------------------------------------------- OpenRouter

/**
 * Chat-completion call with bounded exponential backoff on 429/5xx/network.
 * Returns { text, usage, model, attempts }.
 */
async function callOpenRouter(model, messages, params = {}, retry = {}) {
  const apiKey = getApiKey();
  const maxRetries = clampInt(retry.maxRetries ?? 3, 0, 6);
  const baseMs = clampInt(retry.backoffBaseMs ?? 500, 100, 5_000);

  const body = JSON.stringify({
    model,
    messages,
    temperature: clampNum(params.temperature ?? 0.2, 0, 2),
    max_tokens: clampInt(params.maxTokens ?? 512, 16, 8192),
  });

  let lastError = null;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    if (attempt > 0) {
      const backoff = Math.min(BACKOFF_CAP_MS, baseMs * 2 ** (attempt - 1));
      await sleep(backoff + Math.floor(Math.random() * 250));
    }
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
          'HTTP-Referer': 'https://github.com/ruvnet/ruv-FANN',
          'X-Title': 'ruv-FANN darwin flywheel',
        },
        body,
        signal: controller.signal,
      });
      if (res.status === 429 || res.status >= 500) {
        lastError = new Error(`HTTP ${res.status} from OpenRouter (retryable)`);
        continue;
      }
      if (!res.ok) {
        const detail = (await res.text()).slice(0, 300);
        throw new Error(`OpenRouter request failed: HTTP ${res.status}: ${detail}`);
      }
      const json = await res.json();
      const text = json?.choices?.[0]?.message?.content;
      if (typeof text !== 'string') throw new Error('OpenRouter response missing message content');
      return { text, usage: json.usage ?? {}, model: json.model ?? model, attempts: attempt + 1 };
    } catch (err) {
      if (err?.message?.startsWith('OpenRouter request failed')) throw err; // non-retryable 4xx
      lastError = err; // network error / timeout / retryable status
    } finally {
      clearTimeout(timer);
    }
  }
  throw new Error(`OpenRouter call gave up after ${maxRetries + 1} attempts: ${lastError?.message}`);
}

function clampNum(v, lo, hi) {
  return Math.min(hi, Math.max(lo, Number(v)));
}
function clampInt(v, lo, hi) {
  return Math.round(clampNum(v, lo, hi));
}

// ---------------------------------------------------------------- scoring

/**
 * Heuristic score in [0,1] from weighted components:
 * required keywords, anyOf keywords, forbidden regex absence, minimum length.
 */
function scoreHeuristic(heuristic, text) {
  const lower = String(text).toLowerCase();
  const parts = [];
  if (Array.isArray(heuristic.required) && heuristic.required.length > 0) {
    const hits = heuristic.required.filter((k) => lower.includes(String(k).toLowerCase())).length;
    parts.push({ weight: 0.5, value: hits / heuristic.required.length });
  }
  if (Array.isArray(heuristic.anyOf) && heuristic.anyOf.length > 0) {
    const hit = heuristic.anyOf.some((k) => lower.includes(String(k).toLowerCase()));
    parts.push({ weight: 0.3, value: hit ? 1 : 0 });
  }
  if (typeof heuristic.forbiddenRegex === 'string' && heuristic.forbiddenRegex) {
    let ok = 1;
    try {
      ok = new RegExp(heuristic.forbiddenRegex, 'i').test(text) ? 0 : 1;
    } catch {
      ok = 1; // bad pattern in config: don't penalize the model for it
    }
    parts.push({ weight: 0.1, value: ok });
  }
  if (typeof heuristic.minLength === 'number') {
    parts.push({ weight: 0.1, value: text.length >= heuristic.minLength ? 1 : 0 });
  }
  if (parts.length === 0) return 0;
  const totalWeight = parts.reduce((s, p) => s + p.weight, 0);
  return parts.reduce((s, p) => s + p.weight * p.value, 0) / totalWeight;
}

/** Optional LLM judge: returns a score in [0,1] or null on failure. */
async function judgeScore(config, genome, task, text) {
  const judgeCfg = genome.judge ?? {};
  let model = judgeCfg.model ?? 'tier:mid';
  if (model.startsWith('tier:')) {
    const tierName = model.slice(5);
    model = config.tiers[tierName]?.model;
  }
  if (!model) return null;
  const messages = [
    {
      role: 'system',
      content:
        'You are a strict evaluator. Reply with ONLY a JSON object {"score": <0..1>} grading how well the answer satisfies the task.',
    },
    { role: 'user', content: `TASK:\n${task.prompt}\n\nANSWER:\n${text}\n\nJSON only.` },
  ];
  try {
    const res = await callOpenRouter(model, messages, { temperature: 0, maxTokens: 64 }, genome.retry);
    const match = res.text.match(/\{[^}]*\}/);
    if (!match) return null;
    const score = Number(JSON.parse(match[0]).score);
    return Number.isFinite(score) ? clampNum(score, 0, 1) : null;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------- routing

function resolveTier(config, genome, task, forcedTier = null) {
  if (forcedTier) return forcedTier;
  const { midThreshold = 0.35, frontierThreshold = 0.7 } = genome.routing ?? {};
  if (task.complexity < midThreshold) return config.tierOrder[0];
  if (task.complexity < frontierThreshold) return config.tierOrder[Math.min(1, config.tierOrder.length - 1)];
  return config.tierOrder[config.tierOrder.length - 1];
}

function nextTier(config, tierName) {
  const i = config.tierOrder.indexOf(tierName);
  return i >= 0 && i < config.tierOrder.length - 1 ? config.tierOrder[i + 1] : null;
}

function estimateCostUsd(tier, usage) {
  const p = (usage?.prompt_tokens ?? 0) * (tier.promptCostPerMTok ?? 0);
  const c = (usage?.completion_tokens ?? 0) * (tier.completionCostPerMTok ?? 0);
  return (p + c) / 1_000_000;
}

// ----------------------------------------------------------------- runner

/** Run one task through routing + optional escalation. Returns an eval record. */
async function runTask(config, genome, task, opts = {}) {
  const escalation = genome.escalation ?? {};
  let tierName = resolveTier(config, genome, task, opts.forcedTier);
  let escalations = 0;
  let cumulativeCostUsd = 0;
  let record = null;

  for (;;) {
    const tier = config.tiers[tierName];
    const params = genome.tierParams?.[tierName] ?? {};
    const started = Date.now();
    const res = await callOpenRouter(
      tier.model,
      [{ role: 'user', content: task.prompt }],
      params,
      genome.retry
    );
    const heuristic = scoreHeuristic(task.heuristic, res.text);
    let judge = null;
    if (opts.judge || genome.judge?.enabled) judge = await judgeScore(config, genome, task, res.text);
    const weight = clampNum(genome.judge?.weight ?? 0.3, 0, 1);
    const score = judge === null ? heuristic : (1 - weight) * heuristic + weight * judge;
    cumulativeCostUsd += estimateCostUsd(tier, res.usage);
    record = {
      ts: new Date().toISOString(),
      tag: opts.tag ?? 'flywheel',
      taskId: task.id,
      taskType: task.type,
      tier: tierName,
      model: res.model,
      score: round3(score),
      heuristicScore: round3(heuristic),
      judgeScore: judge === null ? null : round3(judge),
      escalations,
      attempts: res.attempts,
      latencyMs: Date.now() - started,
      usage: res.usage,
      costUsd: round6(cumulativeCostUsd),
      responsePreview: res.text.slice(0, 200),
    };
    const upper = nextTier(config, tierName);
    const canEscalate =
      escalation.enabled && upper && escalations < (escalation.maxEscalations ?? 1) &&
      score < (escalation.scoreThreshold ?? 0.55);
    if (!canEscalate) break;
    escalations += 1;
    tierName = upper;
  }
  return record;
}

/**
 * Run the task suite. Returns { meanScore, totalCostUsd, records }.
 * Every record is appended to the JSONL eval log.
 */
async function runSuite(config, genome = config.genome, opts = {}) {
  const tasks = config.tasks.slice(0, opts.taskLimit ?? config.tasks.length);
  const records = [];
  for (const task of tasks) {
    try {
      const record = await runTask(config, genome, task, opts);
      appendLog(record);
      records.push(record);
      if (!opts.quiet) {
        console.log(
          `  [${record.taskId}] tier=${record.tier} score=${record.score} ` +
            `cost=$${record.costUsd} escalations=${record.escalations}`
        );
      }
    } catch (err) {
      const record = {
        ts: new Date().toISOString(),
        tag: opts.tag ?? 'flywheel',
        taskId: task.id,
        error: String(err.message).slice(0, 300),
        score: 0,
        costUsd: 0,
      };
      appendLog(record);
      records.push(record);
      if (!opts.quiet) console.error(`  [${task.id}] FAILED: ${record.error}`);
    }
  }
  const meanScore = records.reduce((s, r) => s + (r.score ?? 0), 0) / Math.max(1, records.length);
  const totalCostUsd = records.reduce((s, r) => s + (r.costUsd ?? 0), 0);
  return { meanScore: round3(meanScore), totalCostUsd: round6(totalCostUsd), records };
}

async function smokeTest(config) {
  const tierName = config.tierOrder[0];
  const tier = config.tiers[tierName];
  console.log(`Smoke test: one trivial call to ${tier.model} (tier "${tierName}")...`);
  const res = await callOpenRouter(
    tier.model,
    [{ role: 'user', content: 'Reply with exactly: OK' }],
    { temperature: 0, maxTokens: 8 },
    { maxRetries: 2, backoffBaseMs: 500 }
  );
  console.log(`Smoke test OK: model=${res.model} reply=${JSON.stringify(res.text.trim().slice(0, 40))}`);
  appendLog({ ts: new Date().toISOString(), tag: 'smoke', tier: tierName, model: res.model, ok: true });
}

function round3(n) {
  return Math.round(n * 1000) / 1000;
}
function round6(n) {
  return Math.round(n * 1e6) / 1e6;
}

// -------------------------------------------------------------------- CLI

function parseArgs(argv) {
  const args = { config: DEFAULT_CONFIG, limit: null, tier: null, judge: false, smoke: false, tag: 'flywheel' };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--config') args.config = argv[++i];
    else if (a === '--limit') args.limit = clampInt(argv[++i], 1, 1000);
    else if (a === '--tier') args.tier = argv[++i];
    else if (a === '--judge') args.judge = true;
    else if (a === '--smoke') args.smoke = true;
    else if (a === '--tag') args.tag = String(argv[++i]).slice(0, 64);
    else if (a === '--help' || a === '-h') {
      console.log(
        'Usage: node scripts/optimization/flywheel.js ' +
          '[--config path] [--limit N] [--tier cheap|mid|frontier] [--judge] [--smoke] [--tag label]'
      );
      process.exit(0);
    } else {
      console.error(`Unknown argument: ${a}`);
      process.exit(1);
    }
  }
  return args;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  getApiKey(); // fail fast, never printed
  const config = loadConfig(args.config);
  if (args.tier && !config.tiers[args.tier]) {
    console.error(`Unknown tier "${args.tier}". Available: ${config.tierOrder.join(', ')}`);
    process.exit(1);
  }
  if (args.smoke) {
    await smokeTest(config);
    return;
  }
  console.log(`Running flywheel suite (${args.limit ?? config.tasks.length} task(s))...`);
  const summary = await runSuite(config, config.genome, {
    taskLimit: args.limit ?? undefined,
    forcedTier: args.tier ?? null,
    judge: args.judge,
    tag: args.tag,
  });
  console.log(`Done. meanScore=${summary.meanScore} totalCost=$${summary.totalCostUsd}`);
  console.log(`Log: ${EVAL_LOG}`);
}

if (require.main === module) {
  main().catch((err) => {
    console.error(`Flywheel failed: ${err.message}`);
    process.exit(1);
  });
}

module.exports = {
  loadConfig,
  validateConfig,
  runSuite,
  runTask,
  callOpenRouter,
  scoreHeuristic,
  resolveTier,
  estimateCostUsd,
  appendLog,
  EVAL_LOG,
  LOG_DIR,
  DEFAULT_CONFIG,
};
