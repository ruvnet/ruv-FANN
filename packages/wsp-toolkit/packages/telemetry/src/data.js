export const LIMITS = Object.freeze({ maxPackages: 1000, maxDays: 366, maxBytes: 2 * 1024 * 1024,
  maxInputBytes: 24 * 1024 * 1024, maxConcurrency: 8, maxTimeoutMs: 60000, maxSamples: 10000, maxSamplePackages: 4096 });
const npmPattern = /^(?:@[a-z0-9][a-z0-9._-]*\/)?[a-z0-9][a-z0-9._-]*$/;
export function npmName(value) {
  if (typeof value !== 'string' || value.length > 214 || !npmPattern.test(value)) throw new TypeError('invalid npm package name');
  return value;
}
export function repoName(value) {
  if (typeof value !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}\/[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$/.test(value))
    throw new TypeError('invalid GitHub repository name');
  return value;
}
export function record(value, allowed, required, name) {
  if (!value || typeof value !== 'object' || Array.isArray(value) || ![Object.prototype, null].includes(Object.getPrototypeOf(value)))
    throw new TypeError(`${name} must be a plain record`);
  const out = Object.create(null);
  for (const key of Reflect.ownKeys(value)) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (typeof key !== 'string' || !allowed.includes(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw new TypeError(`unknown field or accessor in ${name}`);
    out[key] = descriptor.value;
  }
  if (required.some(key => !Object.hasOwn(out, key))) throw new TypeError(`missing ${name} field`);
  return out;
}
export function integer(value, min, max, name) {
  if (!Number.isSafeInteger(value) || value < min || value > max) throw new RangeError(`${name} must be an integer in ${min}..${max}`);
  return value;
}
export function text(value, max, name) {
  if (typeof value !== 'string' || value.length < 1 || value.length > max) throw new TypeError(`invalid ${name}`);
  return value;
}
export function array(value, maximum, name) {
  if (!Array.isArray(value) || value.length > maximum) throw new RangeError(`invalid ${name} array`);
  return value;
}
export function snapshot(value) {
  const ancestors = new Set(); let nodes = 0, chars = 0;
  function copy(item, depth) {
    if (++nodes > 2_000_000 || depth > 20) throw new RangeError('telemetry input exceeds structural limit');
    if (item === null || typeof item === 'boolean') return item;
    if (typeof item === 'number') {
      if (!Number.isFinite(item)) throw new TypeError('telemetry numbers must be finite');
      return item;
    }
    if (typeof item === 'string') {
      chars += item.length;
      if (chars > LIMITS.maxInputBytes) throw new RangeError('telemetry input exceeds text limit');
      return item;
    }
    if (!item || typeof item !== 'object' || ancestors.has(item)) throw new TypeError('telemetry input must be acyclic JSON data');
    const isArray = Array.isArray(item);
    if (!(isArray ? Object.getPrototypeOf(item) === Array.prototype : [Object.prototype, null].includes(Object.getPrototypeOf(item))))
      throw new TypeError('telemetry input must contain plain data');
    if (isArray && item.length > 1_000_000) throw new RangeError('telemetry array exceeds limit');
    ancestors.add(item); const out = isArray ? [] : Object.create(null); let count = 0;
    for (const key of Reflect.ownKeys(item)) {
      if (isArray && key === 'length') continue;
      if (typeof key !== 'string' || ['__proto__', 'constructor', 'prototype'].includes(key) ||
        (isArray && (!/^(0|[1-9][0-9]*)$/.test(key) || Number(key) >= item.length))) throw new TypeError('unsafe telemetry property');
      chars += key.length;
      if (chars > LIMITS.maxInputBytes) throw new RangeError('telemetry input exceeds text limit');
      const descriptor = Object.getOwnPropertyDescriptor(item, key);
      if (!descriptor || !Object.hasOwn(descriptor, 'value')) throw new TypeError('telemetry input accessors are forbidden');
      out[key] = copy(descriptor.value, depth + 1); count++;
    }
    if (isArray && (out.length !== item.length || count !== item.length)) throw new TypeError('telemetry arrays must be dense');
    ancestors.delete(item); return out;
  }
  const out = copy(value, 0);
  if (new TextEncoder().encode(JSON.stringify(out)).length > LIMITS.maxInputBytes) throw new RangeError('telemetry input exceeds byte limit');
  return out;
}
export function isoDay(value) {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value)) throw new TypeError('dates must use YYYY-MM-DD');
  const stamp = Date.parse(`${value}T00:00:00.000Z`);
  if (!Number.isFinite(stamp) || new Date(stamp).toISOString().slice(0, 10) !== value) throw new TypeError('invalid calendar date');
  return value;
}
export function isoStamp(value) {
  const parts = typeof value === 'string' && value.match(/^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,3}))?Z$/);
  if (!parts || !Number.isFinite(Date.parse(value)) ||
      new Date(Date.parse(value)).toISOString() !== `${parts[1]}.${(parts[2] || '').padEnd(3, '0')}Z`)
    throw new TypeError('observedAt must be a valid UTC ISO timestamp');
  return value;
}
export function addDays(day, amount) { return new Date(Date.parse(`${day}T00:00:00Z`) + amount * 86400000).toISOString().slice(0, 10); }
export function range(start, end) {
  isoDay(start); isoDay(end);
  const count = (Date.parse(`${end}T00:00:00Z`) - Date.parse(`${start}T00:00:00Z`)) / 86400000 + 1;
  integer(count, 1, LIMITS.maxDays, 'date range length');
  return Array.from({ length: count }, (_, i) => addDays(start, i));
}
export function catalogSnapshot(input) {
  const catalog = snapshot(input);
  const raw = record(catalog, ['schema', 'repository', 'packages', 'profiles'], ['schema', 'repository', 'packages', 'profiles'], 'catalog');
  if (raw.schema !== 'ruvnet.constellation-catalog/1') throw new TypeError('unsupported catalog schema');
  repoName(raw.repository); const seen = new Set();
  const packages = array(raw.packages, LIMITS.maxPackages, 'catalog packages').map(value => {
    const p = record(value, ['name', 'role', 'family', 'status', 'version', 'repository', 'directory', 'description', 'contracts', 'links', 'proposedScope'],
      ['name', 'role', 'family', 'status', 'version', 'repository', 'description'], 'package');
    npmName(p.name); if (seen.has(p.name)) throw new TypeError('duplicate catalog package'); seen.add(p.name);
    if (!['nucleus', 'primitive', 'satellite'].includes(p.role) || !['published', 'candidate'].includes(p.status)) throw new TypeError('invalid catalog role or status');
    text(p.family, 64, 'package family'); text(p.version, 128, 'package version'); text(p.description, 2000, 'package description'); repoName(p.repository);
    if (p.directory !== undefined && (typeof p.directory !== 'string' || p.directory.length > 512 || p.directory.split('/').some(v => !/^[A-Za-z0-9_.@-]+$/.test(v) || ['.', '..'].includes(v))))
      throw new TypeError('unsafe package directory');
    return p;
  });
  if (!packages.length) throw new RangeError('catalog requires packages');
  for (const p of array(raw.profiles, 100, 'profiles')) {
    const profile = record(p, ['id', 'description', 'packages'], ['id', 'description', 'packages'], 'profile');
    text(profile.id, 64, 'profile id'); text(profile.description, 2000, 'profile description');
    for (const name of array(profile.packages, LIMITS.maxPackages, 'profile packages')) if (!seen.has(npmName(name))) throw new TypeError('profile names unknown package');
  }
  return { schema: raw.schema, repository: raw.repository, packages, profiles: raw.profiles };
}
export function names(value, maximum = 4096) {
  const values = array(value, maximum, 'package names');
  if (new Set(values).size !== values.length) throw new TypeError('duplicate package names');
  return values.map(npmName).sort();
}
export function safeSum(values) {
  let sum = 0;
  for (const value of values) sum = integer(sum + value, 0, Number.MAX_SAFE_INTEGER, 'download total');
  return sum;
}
