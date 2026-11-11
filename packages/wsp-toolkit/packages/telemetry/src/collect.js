import { LIMITS, record, integer, catalogSnapshot, range, isoDay, npmName } from './data.js';
class SourceError extends Error {}
function abortable(promise, signal) {
  if (signal.aborted) return Promise.reject(signal.reason);
  return new Promise((resolve, reject) => {
    const abort = () => { cleanup(); reject(signal.reason); };
    const cleanup = () => signal.removeEventListener('abort', abort);
    signal.addEventListener('abort', abort, { once: true });
    Promise.resolve(promise).then(value => { cleanup(); resolve(value); }, error => { cleanup(); reject(error); });
  });
}
async function jsonFrom(response, signal) {
  if (!response || response.status !== 200 || response.redirected || response.type === 'opaqueredirect') {
    response?.body?.cancel().catch(() => {});
    throw new SourceError(`public source returned HTTP ${Number.isInteger(response?.status) ? response.status : 'unknown'} or a redirect`);
  }
  const declared = response.headers?.get('content-length');
  if (declared !== null && declared !== undefined && (!/^\d+$/.test(declared) || Number(declared) > LIMITS.maxBytes)) {
    response.body?.cancel().catch(() => {}); throw new SourceError('public source exceeds declared byte limit');
  }
  if (!response.body?.getReader) throw new SourceError('public source has no readable body');
  const reader = response.body.getReader(); let bytes = new Uint8Array(8192), size = 0;
  try {
    for (;;) {
      const { value, done } = await abortable(reader.read(), signal);
      if (done) break;
      if (!(value instanceof Uint8Array) || value.byteLength > LIMITS.maxBytes - size) throw new SourceError('public source exceeds streamed byte limit');
      if (size + value.byteLength > bytes.length) {
        const grown = new Uint8Array(Math.min(LIMITS.maxBytes, Math.max(size + value.byteLength, bytes.length * 2)));
        grown.set(bytes); bytes = grown;
      }
      bytes.set(value, size); size += value.byteLength;
    }
    try { return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes.subarray(0, size))); }
    catch { throw new SourceError('public source is not valid UTF-8 JSON'); }
  } catch (error) { reader.cancel().catch(() => {}); throw error; }
  finally { try { reader.releaseLock(); } catch { /* Abort can leave an uncooperative trusted stream pending. */ } }
}
function dependencies(value) {
  if (value === undefined) return [];
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new SourceError('invalid package dependency metadata');
  const keys = Object.keys(value); if (keys.length > 4096) throw new SourceError('package dependency metadata exceeds limit');
  return keys.map(npmName).sort();
}
function daysFrom(value, name, start, end) {
  if (value?.package !== name || value.start !== start || value.end !== end || !Array.isArray(value.downloads) || value.downloads.length > LIMITS.maxDays)
    throw new SourceError('public download identity or date range mismatch');
  const seen = new Set();
  return value.downloads.map(entry => {
    const day = isoDay(entry?.day);
    if (day < start || day > end || seen.has(day)) throw new SourceError('public source contains duplicate or out-of-range dates');
    seen.add(day);
    return { day, downloads: integer(entry.downloads, 0, Number.MAX_SAFE_INTEGER, 'daily downloads') };
  }).sort((a, b) => a.day.localeCompare(b.day));
}
function errorText(error, signal) { return signal.aborted ? 'collection deadline exceeded' : error instanceof SourceError ? error.message.slice(0, 200) : 'public source request or validation failed'; }

/** Public read-only collection. No credentials, local scanning or background collection. */
export async function collectPublicMetrics(input, configuration) {
  const catalog = catalogSnapshot(input);
  const options = record(configuration, ['start', 'end', 'fetch', 'timeoutMs', 'concurrency'], ['start', 'end'], 'collection options');
  const start = options.start, end = options.end; range(start, end);
  if (end >= new Date().toISOString().slice(0, 10)) throw new RangeError('collect only completed UTC days');
  const fetcher = options.fetch === undefined ? globalThis.fetch : options.fetch;
  if (typeof fetcher !== 'function') throw new TypeError('fetch must be a trusted function');
  const timeoutMs = options.timeoutMs === undefined ? 20000 : integer(options.timeoutMs, 1, LIMITS.maxTimeoutMs, 'collection timeout');
  const concurrency = options.concurrency === undefined ? 4 : integer(options.concurrency, 1, LIMITS.maxConcurrency, 'collection concurrency');
  const controller = new AbortController(), observedAt = new Date().toISOString();
  const timer = setTimeout(() => controller.abort(new Error('collection deadline exceeded')), timeoutMs);
  const request = async url => {
    controller.signal.throwIfAborted();
    const parsed = new URL(url);
    if (parsed.protocol !== 'https:' || !['registry.npmjs.org', 'api.npmjs.org', 'api.github.com'].includes(parsed.hostname) || parsed.username || parsed.password)
      throw new SourceError('unsupported public source URL');
    const response = await abortable(fetcher(url, { redirect: 'error', credentials: 'omit', referrerPolicy: 'no-referrer', signal: controller.signal,
      headers: { Accept: 'application/json' } }), controller.signal);
    if (response.url && new URL(response.url).origin !== parsed.origin) throw new SourceError('public source changed origin');
    return jsonFrom(response, controller.signal);
  };
  const packages = catalog.packages.map(p => ({ name: p.name, status: p.status === 'candidate' ? 'candidate' : 'unavailable', days: [], repository: p.repository }));
  const repositoryNames = [...new Set(catalog.packages.filter(p => p.status === 'published').map(p => p.repository))].sort();
  const repositories = repositoryNames.map(repository => ({ repository, status: 'unavailable' }));
  const tasks = [];
  for (let index = 0; index < catalog.packages.length; index++) {
    const p = catalog.packages[index]; if (p.status === 'candidate') continue;
    tasks.push(async () => {
      const result = packages[index];
      const downloadsURL = `https://api.npmjs.org/downloads/range/${start}:${end}/${encodeURIComponent(p.name)}`;
      const metadataURL = `https://registry.npmjs.org/${encodeURIComponent(p.name)}/latest`;
      result.sources = { downloads: downloadsURL, metadata: metadataURL };
      try { result.days = daysFrom(await request(downloadsURL), p.name, start, end); result.status = 'ok'; }
      catch (error) { result.error = errorText(error, controller.signal); }
      try {
        const data = await request(metadataURL);
        if (data?.name !== p.name || typeof data.version !== 'string' || data.version.length < 1 || data.version.length > 128)
          throw new SourceError('public package metadata identity mismatch');
        result.version = data.version; result.dependencies = dependencies(data.dependencies);
        result.peerDependencies = dependencies(data.peerDependencies); result.optionalDependencies = dependencies(data.optionalDependencies);
        result.metadataStatus = 'ok';
      } catch (error) { result.metadataStatus = 'unavailable'; result.metadataError = errorText(error, controller.signal); }
    });
  }
  for (let index = 0; index < repositoryNames.length; index++) tasks.push(async () => {
    const repository = repositoryNames[index], result = repositories[index];
    result.source = `https://api.github.com/repos/${repository}`;
    try {
      const data = await request(result.source);
      if (typeof data?.full_name !== 'string' || data.full_name.toLowerCase() !== repository.toLowerCase()) throw new SourceError('public repository identity mismatch');
      result.stars = integer(data.stargazers_count, 0, Number.MAX_SAFE_INTEGER, 'repository stars');
      result.openIssuesAndPRs = integer(data.open_issues_count, 0, Number.MAX_SAFE_INTEGER, 'open issues and PRs');
      result.status = 'ok';
    } catch (error) { result.error = errorText(error, controller.signal); }
  });
  let next = 0;
  try {
    await Promise.all(Array.from({ length: Math.min(concurrency, tasks.length) }, async () => {
      while (next < tasks.length) await tasks[next++]();
    }));
    return { schema: 'ruvnet.constellation-observations/1', observedAt, start, end, packages, repositories };
  } finally { clearTimeout(timer); }
}
