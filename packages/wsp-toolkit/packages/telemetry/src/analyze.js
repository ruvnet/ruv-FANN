import { LIMITS, record, integer, text, array, snapshot, catalogSnapshot, isoDay, isoStamp, addDays, range, names, safeSum } from './data.js';

function observationsSnapshot(value, catalog) {
  const raw = record(snapshot(value), ['schema', 'observedAt', 'start', 'end', 'packages', 'repositories'],
    ['schema', 'observedAt', 'start', 'end', 'packages', 'repositories'], 'observations');
  if (raw.schema !== 'ruvnet.constellation-observations/1') throw new TypeError('unsupported observations schema');
  isoStamp(raw.observedAt); range(raw.start, raw.end);
  if (raw.end >= raw.observedAt.slice(0, 10)) throw new RangeError('observations must end before their UTC collection date');
  const catalogByName = new Map(catalog.packages.map(p => [p.name, p])), seen = new Set();
  const packages = array(raw.packages, LIMITS.maxPackages, 'observed packages').map(value => {
    const p = record(value, ['name', 'status', 'days', 'version', 'dependencies', 'peerDependencies', 'optionalDependencies', 'metadataStatus', 'metadataError', 'error', 'repository', 'sources'],
      ['name', 'status', 'days'], 'package observation');
    const declared = catalogByName.get(p.name);
    if (!declared || seen.has(p.name)) throw new TypeError('duplicate or unknown observed package'); seen.add(p.name);
    if (!['ok', 'unavailable', 'candidate'].includes(p.status) || (declared.status === 'candidate') !== (p.status === 'candidate')) throw new TypeError('observation status contradicts catalog');
    const daySeen = new Set();
    const days = array(p.days, LIMITS.maxDays, 'observed days').map(value => {
      const day = record(value, ['day', 'downloads'], ['day', 'downloads'], 'day observation'); isoDay(day.day);
      if (day.day < raw.start || day.day > raw.end || daySeen.has(day.day)) throw new TypeError('duplicate or out-of-range observed day');
      daySeen.add(day.day); return { day: day.day, downloads: integer(day.downloads, 0, Number.MAX_SAFE_INTEGER, 'daily downloads') };
    }).sort((a, b) => a.day.localeCompare(b.day));
    if (p.status !== 'ok' && days.length) throw new TypeError('unavailable or candidate observations cannot carry measured downloads');
    if (p.version !== undefined) text(p.version, 128, 'observed version');
    if (p.metadataStatus !== undefined && !['ok', 'unavailable'].includes(p.metadataStatus)) throw new TypeError('invalid metadata status');
    return { ...p, days,
      dependencies: p.dependencies === undefined ? [] : names(p.dependencies),
      peerDependencies: p.peerDependencies === undefined ? [] : names(p.peerDependencies),
      optionalDependencies: p.optionalDependencies === undefined ? [] : names(p.optionalDependencies) };
  });
  const repositories = [], repoSeen = new Set(), allowedRepos = new Set(catalog.packages.map(p => p.repository));
  for (const value of array(raw.repositories, LIMITS.maxPackages, 'repositories')) {
    const item = record(value, ['repository', 'status', 'stars', 'openIssuesAndPRs', 'error', 'source'], ['repository', 'status'], 'repository observation');
    if (!allowedRepos.has(item.repository) || repoSeen.has(item.repository) || !['ok', 'unavailable'].includes(item.status)) throw new TypeError('invalid repository observation');
    repoSeen.add(item.repository);
    if (item.status === 'ok') {
      integer(item.stars, 0, Number.MAX_SAFE_INTEGER, 'stars'); integer(item.openIssuesAndPRs, 0, Number.MAX_SAFE_INTEGER, 'issues and PRs');
    } else if (item.stars !== undefined || item.openIssuesAndPRs !== undefined) throw new TypeError('unavailable repository cannot carry counts');
    repositories.push({ repository: item.repository, status: item.status,
      stars: item.status === 'ok' ? item.stars : null, openIssuesAndPRs: item.status === 'ok' ? item.openIssuesAndPRs : null,
      source: `https://api.github.com/repos/${item.repository}`, ...(item.error ? { error: text(item.error, 1000, 'repository error') } : {}) });
  }
  return { ...raw, packages, repositories };
}
function completeWindow(days, end, length) {
  const values = [];
  for (let offset = 0; offset < length; offset++) {
    const value = days.get(addDays(end, -offset)); if (value === undefined) return null; values.push(value);
  }
  return safeSum(values);
}
function attachMetrics(value, catalog, start, end) {
  const primitives = catalog.packages.filter(p => p.role === 'primitive').map(p => p.name);
  const unavailable = { status: 'unavailable', unit: 'consenting-project-snapshots', denominator: 0, numerator: 0, percentage: null,
    projects: 0, perPrimitive: primitives.map(name => ({ name, numerator: 0, denominator: 0, percentage: null })),
    source: { provided: false, samplesReceived: 0, snapshotsInWindow: 0, duplicateSnapshotsRemoved: 0 },
    note: 'Attach requires explicit consenting project snapshots. npm download counts do not identify users or joint installations.' };
  if (value === undefined) return unavailable;
  const raw = record(value, ['schema', 'consent', 'samples'], ['schema', 'consent', 'samples'], 'adoption');
  if (raw.schema !== 'ruvnet.constellation-adoption/1' || raw.consent !== true) throw new TypeError('adoption requires the supported schema and explicit consent:true');
  const samples = array(raw.samples, LIMITS.maxSamples, 'adoption samples'), latest = new Map(), seenSnapshots = new Map(); let inWindow = 0;
  for (const value of samples) {
    const sample = record(value, ['sampleId', 'observedAt', 'packages'], ['sampleId', 'observedAt', 'packages'], 'adoption sample');
    if (typeof sample.sampleId !== 'string' || !/^[a-f0-9]{64}$/.test(sample.sampleId)) throw new TypeError('sampleId must be an anonymized lowercase SHA-256 identifier');
    isoStamp(sample.observedAt); const packages = names(sample.packages, LIMITS.maxSamplePackages);
    const stamp = Date.parse(sample.observedAt), key = `${sample.sampleId}:${stamp}`, content = JSON.stringify(packages);
    if (seenSnapshots.has(key) && seenSnapshots.get(key) !== content) throw new TypeError('conflicting duplicate project snapshot');
    // Check every declared instant, including older snapshots, before selecting a latest in-window observation.
    seenSnapshots.set(key, content);
    const day = sample.observedAt.slice(0, 10); if (day < start || day > end) continue;
    inWindow++; const previous = latest.get(sample.sampleId);
    if (!previous || stamp > previous.stamp) latest.set(sample.sampleId, { stamp, packages });
  }
  const nuclei = new Set(catalog.packages.filter(p => p.role === 'nucleus').map(p => p.name));
  const projects = [...latest.values()], eligible = projects.filter(p => p.packages.some(name => nuclei.has(name)));
  const denominator = eligible.length, numerator = eligible.filter(p => p.packages.some(name => primitives.includes(name))).length;
  return { status: denominator ? 'available' : 'unavailable', unit: 'consenting-project-snapshots', denominator, numerator,
    percentage: denominator ? 100 * numerator / denominator : null, projects: projects.length,
    perPrimitive: primitives.map(name => { const numerator = eligible.filter(p => p.packages.includes(name)).length;
      return { name, numerator, denominator, percentage: denominator ? 100 * numerator / denominator : null }; }),
    source: { provided: true, samplesReceived: samples.length, snapshotsInWindow: inWindow, duplicateSnapshotsRemoved: inWindow - projects.length },
    note: 'Latest consenting snapshot per project in this window. Denominator: sampled projects containing any catalog nucleus. This is not a percentage of developers or npm users.' };
}
function correlation(left, right) {
  const r = new Map(right.days.map(d => [d.day, d.downloads]));
  const pairs = left.days.filter(d => r.has(d.day)).map(d => [d.downloads, r.get(d.day)]);
  if (pairs.length < 7) return null;
  const mx = pairs.reduce((sum, p) => sum + p[0], 0) / pairs.length, my = pairs.reduce((sum, p) => sum + p[1], 0) / pairs.length;
  let cross = 0, xx = 0, yy = 0;
  for (const [x, y] of pairs) { cross += (x - mx) * (y - my); xx += (x - mx) ** 2; yy += (y - my) ** 2; }
  if (xx === 0 || yy === 0) return null;
  return { left: left.name, right: right.name, r: Math.max(-1, Math.min(1, cross / Math.sqrt(xx * yy))), samples: pairs.length };
}

/** Download correlation is observational; attach is computed only from an explicit local adoption sample. */
export function analyzeConstellation(inputCatalog, inputObservations, configuration = {}) {
  const catalog = catalogSnapshot(inputCatalog), observations = observationsSnapshot(inputObservations, catalog);
  const options = record(configuration, ['adoption'], [], 'analysis options');
  const adoption = options.adoption === undefined ? undefined : snapshot(options.adoption);
  const observed = new Map(observations.packages.map(p => [p.name, p])), known = new Set(catalog.packages.map(p => p.name));
  const packages = catalog.packages.map(p => {
    const data = observed.get(p.name) ?? { name: p.name, status: p.status === 'candidate' ? 'candidate' : 'unavailable', days: [], dependencies: [], peerDependencies: [], optionalDependencies: [] };
    const days = new Map(data.days.map(d => [d.day, d.downloads]));
    const baselineValues = [7, 14, 21, 28].map(offset => days.get(addDays(observations.end, -offset)));
    const weekdayBaseline = baselineValues.every(v => v !== undefined) ? safeSum(baselineValues) / 4 : null;
    const latestDayDownloads = days.get(observations.end) ?? null;
    const weekdayDeltaPercent = latestDayDownloads !== null && weekdayBaseline !== null && weekdayBaseline > 0 ? 100 * (latestDayDownloads - weekdayBaseline) / weekdayBaseline : null;
    const edges = [];
    for (const [key, kind] of [['dependencies', 'runtime'], ['peerDependencies', 'peer'], ['optionalDependencies', 'optional']])
      for (const target of data.metadataStatus === 'ok' ? data[key] ?? [] : []) if (known.has(target)) edges.push({ source: p.name, target, kind });
    return { name: p.name, role: p.role, family: p.family, catalogStatus: p.status, status: data.status,
      catalogVersion: p.version, version: data.version ?? null, description: p.description, repository: p.repository,
      npmURL: p.status === 'published' ? `https://www.npmjs.com/package/${p.name}` : null,
      sourceURL: `https://github.com/${p.repository}${p.directory ? `/tree/main/${p.directory}` : ''}`,
      latestDayDownloads, last7Downloads: completeWindow(days, observations.end, 7), last30Downloads: completeWindow(days, observations.end, 30),
      weekdayBaseline, weekdayDeltaPercent, weekdayBaselineDays: baselineValues.filter(v => v !== undefined).length,
      days: data.days, coverage: { observedDays: days.size, expectedDays: range(observations.start, observations.end).length },
      metadataStatus: data.metadataStatus ?? 'unavailable', dependencies: data.dependencies ?? [], peerDependencies: data.peerDependencies ?? [], optionalDependencies: data.optionalDependencies ?? [],
      runtimeDependencyCount: data.metadataStatus === 'ok' ? (data.dependencies ?? []).length : null,
      peerDependencyCount: data.metadataStatus === 'ok' ? (data.peerDependencies ?? []).length : null,
      optionalDependencyCount: data.metadataStatus === 'ok' ? (data.optionalDependencies ?? []).length : null,
      declaredLinks: edges, ...(data.error ? { error: text(data.error, 1000, 'package error') } : {}) };
  });
  const published = packages.filter(p => p.catalogStatus === 'published'), candidates = packages.length - published.length;
  const total = key => published.length && published.every(p => p[key] !== null) ? safeSum(published.map(p => p[key])) : null;
  const coverage = { catalogPackages: packages.length, publishedPackages: published.length, candidatePackages: candidates,
    packagesWithLatestDay: published.filter(p => p.latestDayDownloads !== null).length,
    packagesWithLast7: published.filter(p => p.last7Downloads !== null).length, packagesWithLast30: published.filter(p => p.last30Downloads !== null).length,
    unavailablePackages: published.filter(p => p.status !== 'ok').length };
  const dayMaps = published.map(p => new Map(p.days.map(d => [d.day, d.downloads])));
  const dailySeries = range(observations.start, observations.end).map(day => {
    const values = dayMaps.map(days => days.get(day)).filter(v => v !== undefined);
    return { day, downloads: published.length && values.length === published.length ? safeSum(values) : null,
      observedDownloads: safeSum(values), observedPackages: values.length, expectedPackages: published.length };
  });
  const eligible = published.filter(p => p.latestDayDownloads !== null && p.weekdayBaseline !== null), families = [...new Set(eligible.map(p => p.family))].sort();
  const netChange = eligible.reduce((sum, p) => sum + p.latestDayDownloads - p.weekdayBaseline, 0);
  const weekdaySummary = { eligiblePackages: eligible.length, aboveBaseline: eligible.filter(p => p.latestDayDownloads > p.weekdayBaseline).length, netChange,
    families: families.map(family => { const change = eligible.filter(p => p.family === family).reduce((sum, p) => sum + p.latestDayDownloads - p.weekdayBaseline, 0);
      return { family, netChange: change, shareOfNetChangePercent: netChange > 0 ? 100 * change / netChange : null }; }),
    note: 'Previous four matching weekdays, compared with the final complete UTC day. Synchronized downloads do not prove co-installation or causal cross-pollination.' };
  const considered = published.filter(p => p.status === 'ok').sort((a, b) => (b.last30Downloads ?? -1) - (a.last30Downloads ?? -1) || a.name.localeCompare(b.name)).slice(0, 32);
  const pairs = [];
  for (let i = 0; i < considered.length; i++) for (let j = i + 1; j < considered.length; j++) { const value = correlation(considered[i], considered[j]); if (value) pairs.push(value); }
  pairs.sort((a, b) => Math.abs(b.r) - Math.abs(a.r) || a.left.localeCompare(b.left) || a.right.localeCompare(b.right));
  return { schema: 'ruvnet.constellation-report/1', observedAt: observations.observedAt, start: observations.start, end: observations.end,
    packages, totals: { latestDayDownloads: total('latestDayDownloads'), last7Downloads: total('last7Downloads'), last30Downloads: total('last30Downloads'),
      observedLast7Downloads: safeSum(published.filter(p => p.last7Downloads !== null).map(p => p.last7Downloads)),
      observedLast30Downloads: safeSum(published.filter(p => p.last30Downloads !== null).map(p => p.last30Downloads)), coverage },
    dailySeries, repositories: observations.repositories, weekdaySummary,
    correlations: { kind: 'daily-download-correlation', consideredPackages: considered.length, eligiblePairs: pairs.length, pairs: pairs.slice(0, 10),
      note: 'Pearson correlation on shared observed days, at least seven points; limited to the 32 highest-volume measured packages. Correlation is not attach rate.' },
    declaredLinks: packages.flatMap(p => p.declaredLinks),
    declaredLinksCoverage: { observedManifests: published.filter(p => p.metadataStatus === 'ok').length, expectedManifests: published.length }, attach: attachMetrics(adoption, catalog, observations.start, observations.end),
    definitions: { downloads: 'npm registry download events, including CI and transitive installs; not distinct developers.',
      totals: 'Only complete published-package windows are totaled. Candidate and unavailable observations stay unknown; overlapping cohorts are not unique users.',
      weekday: 'Mean of the previous four matching weekdays; percentage unknown when coverage is incomplete or baseline is zero.',
      repositories: 'GitHub open_issues_count includes open pull requests as well as issues.',
      adoption: 'Explicit local consenting project snapshots only; no automatic collection or upload.' } };
}
