#!/usr/bin/env node
'use strict';

/**
 * Darwin: gradient-free, elitist evolution of the flywheel genome.
 *
 * Loads the genome from config/optimization/darwin.config.json, applies
 * bounded mutations (tier params, routing thresholds, retry/escalation
 * policy, judge weight), evaluates each candidate via the flywheel on a
 * small task batch, and keeps a candidate only if its fitness beats the
 * current elite (fitness = meanScore - costWeight * scaled cost).
 *
 * Generation history goes to .claude-flow/optimization/generations.jsonl;
 * the winning genome is written back to the config.
 *
 * Safety gate: the genome is data-only. It is validated against a strict
 * key whitelist + numeric bounds; model ids must match a safe pattern.
 * This script never imports child_process and never executes shell
 * commands or evaluates strings — a genome can only tune numbers and
 * pick models from the configured tiers.
 *
 * Usage:
 *   node scripts/optimization/darwin-evolve.js [--generations N] [--batch N]
 *        [--dry-run] [--seed N] [--config path]
 */

const fs = require('fs');
const path = require('path');
const flywheel = require('./flywheel.js');

const ROOT = path.resolve(__dirname, '..', '..');
const GEN_LOG = path.join(ROOT, '.claude-flow', 'optimization', 'generations.jsonl');

// ------------------------------------------------------------ safety gate

/** Exhaustive whitelist of genome paths. Anything else is rejected. */
const GENOME_SCHEMA = {
  'routing.midThreshold': 'number',
  'routing.frontierThreshold': 'number',
  'tierParams.*.temperature': 'number',
  'tierParams.*.maxTokens': 'number',
  'retry.maxRetries': 'number',
  'retry.backoffBaseMs': 'number',
  'escalation.enabled': 'boolean',
  'escalation.scoreThreshold': 'number',
  'escalation.maxEscalations': 'number',
  'judge.enabled': 'boolean',
  'judge.model': 'model-string',
  'judge.weight': 'number',
};
const MODEL_ID_RE = /^[\w.:/-]+$/; // no spaces, quotes, or shell metacharacters

function flatten(obj, prefix = '') {
  const out = [];
  for (const [key, value] of Object.entries(obj)) {
    const p = prefix ? `${prefix}.${key}` : key;
    if (value && typeof value === 'object' && !Array.isArray(value)) out.push(...flatten(value, p));
    else out.push([p, value]);
  }
  return out;
}

function schemaTypeFor(flatPath) {
  if (GENOME_SCHEMA[flatPath]) return GENOME_SCHEMA[flatPath];
  const wild = flatPath.replace(/(^|\.)(cheap|mid|frontier|[\w-]+)(?=\.(temperature|maxTokens)$)/, '$1*');
  return GENOME_SCHEMA[wild] ?? null;
}

/** Throws if the genome contains unknown keys, wrong types, or unsafe strings. */
function assertSafeGenome(genome, config) {
  for (const [flatPath, value] of flatten(genome)) {
    const type = schemaTypeFor(flatPath);
    if (!type) throw new Error(`Safety gate: unknown genome key "${flatPath}" rejected`);
    if (type === 'number') {
      if (typeof value !== 'number' || !Number.isFinite(value)) {
        throw new Error(`Safety gate: "${flatPath}" must be a finite number`);
      }
    } else if (type === 'boolean') {
      if (typeof value !== 'boolean') throw new Error(`Safety gate: "${flatPath}" must be boolean`);
    } else if (type === 'model-string') {
      if (typeof value !== 'string' || !MODEL_ID_RE.test(value.replace(/^tier:/, ''))) {
        throw new Error(`Safety gate: "${flatPath}" is not a safe model reference`);
      }
      const ref = value.startsWith('tier:') ? value.slice(5) : null;
      if (ref && !config.tiers[ref]) throw new Error(`Safety gate: judge tier "${ref}" not in config`);
    }
  }
  // tierParams keys must be actual configured tiers
  for (const tierName of Object.keys(genome.tierParams ?? {})) {
    if (!config.tiers[tierName]) throw new Error(`Safety gate: unknown tier "${tierName}" in tierParams`);
  }
}

// -------------------------------------------------------------- mutation

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp(v, [lo, hi]) {
  return Math.min(hi, Math.max(lo, v));
}

function bounds(config, name, fallback) {
  const b = config.mutationBounds?.[name];
  return Array.isArray(b) && b.length === 2 ? b : fallback;
}

/** Mutation operators: each returns a {path, from, to} description. */
function buildOperators(config, rng) {
  const tierNames = config.tierOrder;
  const pickTier = () => tierNames[Math.floor(rng() * tierNames.length)];
  const jitter = (v, step) => v + (rng() * 2 - 1) * step;
  return [
    (g) => {
      const t = pickTier();
      const from = g.tierParams[t].temperature;
      g.tierParams[t].temperature = round3(clamp(jitter(from, 0.15), bounds(config, 'temperature', [0, 1.2])));
      return { path: `tierParams.${t}.temperature`, from, to: g.tierParams[t].temperature };
    },
    (g) => {
      const t = pickTier();
      const from = g.tierParams[t].maxTokens;
      const factor = rng() < 0.5 ? 0.75 : 1.25;
      g.tierParams[t].maxTokens = Math.round(clamp(from * factor, bounds(config, 'maxTokens', [128, 2048])));
      return { path: `tierParams.${t}.maxTokens`, from, to: g.tierParams[t].maxTokens };
    },
    (g) => {
      const from = g.routing.midThreshold;
      g.routing.midThreshold = round3(clamp(jitter(from, 0.08), bounds(config, 'midThreshold', [0.05, 0.6])));
      return { path: 'routing.midThreshold', from, to: g.routing.midThreshold };
    },
    (g) => {
      const from = g.routing.frontierThreshold;
      g.routing.frontierThreshold = round3(
        clamp(jitter(from, 0.08), bounds(config, 'frontierThreshold', [0.4, 0.95]))
      );
      if (g.routing.frontierThreshold < g.routing.midThreshold + 0.05) {
        g.routing.frontierThreshold = round3(Math.min(0.95, g.routing.midThreshold + 0.05));
      }
      return { path: 'routing.frontierThreshold', from, to: g.routing.frontierThreshold };
    },
    (g) => {
      const from = g.retry.maxRetries;
      const delta = rng() < 0.5 ? -1 : 1;
      g.retry.maxRetries = Math.round(clamp(from + delta, bounds(config, 'maxRetries', [0, 4])));
      return { path: 'retry.maxRetries', from, to: g.retry.maxRetries };
    },
    (g) => {
      const from = g.retry.backoffBaseMs;
      const factor = rng() < 0.5 ? 0.8 : 1.25;
      g.retry.backoffBaseMs = Math.round(clamp(from * factor, bounds(config, 'backoffBaseMs', [250, 2000])));
      return { path: 'retry.backoffBaseMs', from, to: g.retry.backoffBaseMs };
    },
    (g) => {
      const from = g.escalation.scoreThreshold;
      g.escalation.scoreThreshold = round3(
        clamp(jitter(from, 0.08), bounds(config, 'escalationScoreThreshold', [0.3, 0.9]))
      );
      return { path: 'escalation.scoreThreshold', from, to: g.escalation.scoreThreshold };
    },
    (g) => {
      const from = g.escalation.enabled;
      g.escalation.enabled = !from;
      return { path: 'escalation.enabled', from, to: g.escalation.enabled };
    },
    (g) => {
      const from = g.judge.weight;
      g.judge.weight = round3(clamp(jitter(from, 0.1), bounds(config, 'judgeWeight', [0, 0.5])));
      return { path: 'judge.weight', from, to: g.judge.weight };
    },
  ];
}

function mutate(config, genome, rng) {
  const candidate = structuredClone(genome);
  const operators = buildOperators(config, rng);
  const count = rng() < 0.6 ? 1 : 2;
  const mutations = [];
  for (let i = 0; i < count; i++) {
    const op = operators[Math.floor(rng() * operators.length)];
    mutations.push(op(candidate));
  }
  return { candidate, mutations };
}

function round3(n) {
  return Math.round(n * 1000) / 1000;
}

// --------------------------------------------------------------- fitness

function fitness(summary, config) {
  const costWeight = config.evolution?.costWeight ?? 0.3;
  const perTaskCost = summary.totalCostUsd / Math.max(1, summary.records.length);
  // Scale so ~1 cent/task at costWeight 0.3 costs ~0.3 fitness points.
  return round3(summary.meanScore - costWeight * perTaskCost * 100);
}

function appendGenLog(record) {
  fs.mkdirSync(path.dirname(GEN_LOG), { recursive: true });
  fs.appendFileSync(GEN_LOG, JSON.stringify(record) + '\n', 'utf8');
}

// ------------------------------------------------------------------ main

function parseArgs(argv) {
  const args = {
    config: flywheel.DEFAULT_CONFIG,
    generations: null,
    batch: null,
    dryRun: false,
    seed: Date.now() % 2147483647,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--config') args.config = argv[++i];
    else if (a === '--generations') args.generations = Math.max(1, Math.min(100, parseInt(argv[++i], 10) || 1));
    else if (a === '--batch') args.batch = Math.max(1, Math.min(50, parseInt(argv[++i], 10) || 1));
    else if (a === '--dry-run') args.dryRun = true;
    else if (a === '--seed') args.seed = parseInt(argv[++i], 10) >>> 0;
    else if (a === '--help' || a === '-h') {
      console.log(
        'Usage: node scripts/optimization/darwin-evolve.js ' +
          '[--generations N] [--batch N] [--dry-run] [--seed N] [--config path]'
      );
      process.exit(0);
    } else {
      console.error(`Unknown argument: ${a}`);
      process.exit(1);
    }
  }
  return args;
}

async function evaluate(config, genome, tag, batch, dryRun) {
  if (dryRun) return { meanScore: null, totalCostUsd: 0, records: [] };
  const summary = await flywheel.runSuite(config, genome, { taskLimit: batch, tag, quiet: false });
  return summary;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.dryRun && !process.env.OPENROUTER_API_KEY) {
    console.error('ERROR: OPENROUTER_API_KEY is not set. Export it first, or use --dry-run.');
    process.exit(1);
  }
  const config = flywheel.loadConfig(args.config);
  const generations = args.generations ?? config.evolution?.generations ?? 5;
  const batch = args.batch ?? config.evolution?.taskBatch ?? 4;
  const minImprovement = config.evolution?.minImprovement ?? 0;
  const rng = mulberry32(args.seed);

  let elite = structuredClone(config.genome);
  assertSafeGenome(elite, config);
  console.log(
    `Darwin: ${generations} generation(s), batch=${batch} task(s), seed=${args.seed}` +
      (args.dryRun ? ' [dry-run: no API calls, config not modified]' : '')
  );

  console.log('Evaluating elite (gen 0)...');
  const eliteSummary = await evaluate(config, elite, 'darwin-gen0-elite', batch, args.dryRun);
  let eliteFitness = args.dryRun ? null : fitness(eliteSummary, config);
  console.log(`  elite fitness=${eliteFitness ?? 'n/a (dry-run)'}`);
  let improved = false;

  for (let gen = 1; gen <= generations; gen++) {
    const { candidate, mutations } = mutate(config, elite, rng);
    assertSafeGenome(candidate, config); // safety gate on every candidate
    const mutDesc = mutations.map((m) => `${m.path}: ${m.from} -> ${m.to}`).join(', ');
    console.log(`Gen ${gen}: mutating ${mutDesc}`);

    const summary = await evaluate(config, candidate, `darwin-gen${gen}`, batch, args.dryRun);
    const candFitness = args.dryRun ? null : fitness(summary, config);
    const promoted = !args.dryRun && candFitness > eliteFitness + minImprovement;

    appendGenLog({
      ts: new Date().toISOString(),
      seed: args.seed,
      gen,
      dryRun: args.dryRun,
      mutations,
      candidateFitness: candFitness,
      candidateMeanScore: summary.meanScore,
      candidateCostUsd: summary.totalCostUsd,
      eliteFitness,
      promoted,
    });

    if (promoted) {
      elite = candidate;
      eliteFitness = candFitness;
      improved = true;
      console.log(`  PROMOTED: fitness=${candFitness} (new elite)`);
    } else {
      console.log(`  rejected: fitness=${candFitness ?? 'n/a'} (elite=${eliteFitness ?? 'n/a'})`);
    }
  }

  if (!args.dryRun && improved) {
    // Re-read the file and patch only the genome, preserving everything else.
    const onDisk = JSON.parse(fs.readFileSync(args.config, 'utf8'));
    onDisk.genome = elite;
    fs.writeFileSync(args.config, JSON.stringify(onDisk, null, 2) + '\n', 'utf8');
    console.log(`Best genome written back to ${args.config}`);
  } else if (!args.dryRun) {
    console.log('No candidate beat the elite; config unchanged.');
  }
  console.log(`Generation history: ${GEN_LOG}`);
}

if (require.main === module) {
  main().catch((err) => {
    console.error(`Darwin failed: ${err.message}`);
    process.exit(1);
  });
}

module.exports = { assertSafeGenome, mutate, fitness, GEN_LOG };
