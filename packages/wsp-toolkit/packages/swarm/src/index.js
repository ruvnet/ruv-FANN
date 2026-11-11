import { Worker } from 'node:worker_threads';
import { performance } from 'node:perf_hooks';
import { SWARM_LIMITS, record, integer, snapshotJobs, prepareJobs, patchInfo } from './data.js';
export { SWARM_LIMITS } from './data.js';

/** Pure preflight: snapshots/validates the same data as execution and reserves full search work. */
export function estimateSwarmWork(jobs) {
  return prepareJobs(snapshotJobs(jobs)).reservedWork;
}

function configuration(value = {}) {
  const options = record(value, ['concurrency', 'timeoutMs', 'signal', 'maxTotalWork'], [], 'swarm options');
  const concurrency = options.concurrency === undefined ? 2 : integer(options.concurrency, 1, SWARM_LIMITS.maxConcurrency, 'concurrency');
  const timeoutMs = options.timeoutMs === undefined ? 5000 : integer(options.timeoutMs, 1, SWARM_LIMITS.maxTimeoutMs, 'timeoutMs');
  const maxTotalWork = options.maxTotalWork === undefined ? SWARM_LIMITS.maxTotalWork :
    integer(options.maxTotalWork, 0, SWARM_LIMITS.maxTotalWork, 'maxTotalWork');
  if (options.signal !== undefined && !(options.signal instanceof AbortSignal)) throw new TypeError('signal must be an AbortSignal');
  return { concurrency, timeoutMs, maxTotalWork, signal: options.signal };
}
function errorText(value) { return typeof value?.message === 'string' ? value.message.slice(0, 512) : 'proposal worker failed'; }

function execute(job, options) {
  const startedAtMs = performance.now();
  let worker;
  try {
    worker = new Worker(new URL('./worker.js', import.meta.url), { workerData: job, execArgv: [],
      resourceLimits: { maxOldGenerationSizeMb: 64, maxYoungGenerationSizeMb: 16, stackSizeMb: 4 } });
  } catch (error) {
    return Promise.resolve({ id: job.id, status: 'error', error: errorText(error), startedAtMs, finishedAtMs: performance.now() });
  }
  const workerThreadId = worker.threadId;
  return new Promise((resolve, reject) => {
    let settled = false;
    let timer;
    const finish = result => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      options.signal?.removeEventListener('abort', onAbort);
      // Keep listeners until termination resolves so an error during cleanup cannot become unhandled.
      worker.terminate().then(() => {
        worker.removeListener('message', onMessage);
        worker.removeListener('error', onError);
        worker.removeListener('exit', onExit);
        resolve({ id: job.id, ...result, workerThreadId, startedAtMs, finishedAtMs: performance.now() });
      }, reject);
    };
    const onAbort = () => finish({ status: 'cancelled', error: 'swarm aborted' });
    const onError = error => finish({ status: 'error', error: errorText(error) });
    const onExit = code => finish({ status: 'error', error: `proposal worker exited before a result (code ${code})` });
    const onMessage = message => {
      try {
        if (!message || message.workerThreadId !== workerThreadId || !['ok', 'error'].includes(message.status))
          throw new Error('invalid proposal worker result');
        if (message.status === 'error') { finish({ status: 'error', error: String(message.error || 'proposal execution failed').slice(0, 512) }); return; }
        const patch = message.patch === null ? null : patchInfo(message.patch);
        if (job.kind === 'search') {
          const summary = message.search;
          if (!summary || !['wasm', 'wasm-simd'].includes(summary.backend) ||
              !Number.isSafeInteger(summary.evaluations) || summary.evaluations < 0 || summary.evaluations > 256 ||
              !Number.isSafeInteger(summary.inferencePasses) || summary.inferencePasses < 1 || summary.inferencePasses > 259 ||
              !Number.isSafeInteger(summary.reservedWork) || summary.reservedWork < 1 || summary.reservedWork > 10_000_000)
            throw new Error('invalid proposal search result');
          finish({ status: 'ok', patch, search: { evaluations: summary.evaluations, inferencePasses: summary.inferencePasses,
            reservedWork: summary.reservedWork, backend: summary.backend } });
        } else finish({ status: 'ok', patch });
      } catch (error) { finish({ status: 'error', error: errorText(error) }); }
    };
    worker.on('message', onMessage);
    worker.on('error', onError);
    worker.on('exit', onExit);
    options.signal?.addEventListener('abort', onAbort, { once: true });
    timer = setTimeout(() => finish({ status: 'timeout', error: 'proposal worker deadline exceeded' }), options.timeoutMs);
    // Also handle an abort that occurred during Worker construction.
    if (options.signal?.aborted) onAbort();
  });
}

/** Fixed trusted operations only. Worker proposals are untrusted data and cannot promote checkpoints. */
export async function runProposalSwarm(jobs, options) {
  // All caller-owned job/configuration data are captured before the first await.
  const prepared = prepareJobs(snapshotJobs(jobs)), config = configuration(options);
  if (prepared.reservedWork > config.maxTotalWork) throw new RangeError('swarm exceeds configured aggregate work budget');
  const results = new Array(prepared.jobs.length);
  let next = 0;
  async function runner() {
    while (next < prepared.jobs.length) {
      const index = next++, job = prepared.jobs[index];
      if (config.signal?.aborted) { results[index] = { id: job.id, status: 'cancelled', error: 'swarm aborted before job start' }; continue; }
      results[index] = await execute(job, config);
    }
  }
  const settled = await Promise.allSettled(Array.from({ length: Math.min(config.concurrency, prepared.jobs.length) }, runner));
  const failed = settled.find(item => item.status === 'rejected');
  if (failed) throw failed.reason; // Other started workers have already finished cleanup.
  return results;
}
