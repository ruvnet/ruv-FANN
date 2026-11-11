/** Identifiers describe shared formats; numerical WSP validation remains authoritative. */
export const CONTRACTS = Object.freeze({
  model: 'ruv-fann.dense-layers/1', patch: 'ruv-fann.weight-patch/1',
  evaluation: 'ruv-fann.weight-evaluation/1', dataset: 'ruv-fann.weight-dataset/1',
  policy: 'ruv-fann.weight-policy/1', invariants: 'ruv-fann.invariants/1',
  invariantCheck: 'ruv-fann.invariant-check/1', loop: 'ruv-fann.evolution-loop/1',
  module: 'ruvnet.constellation-module/1', catalog: 'ruvnet.constellation-catalog/1'
});
export const CONTRACT_LIMITS = Object.freeze({ maxPackages: 1000, maxProfiles: 64,
  maxContracts: 32, maxIntegrations: 64, maxBytes: 2 * 1024 * 1024 });
const own = Object.getOwnPropertyDescriptor;
const npmPart = '[a-z0-9][a-z0-9._-]*';
const npmPattern = new RegExp(`^(?:@${npmPart}/)?${npmPart}$`);
const roles = ['nucleus', 'primitive', 'satellite'];

function record(value, allowed, required, name) {
  if (!value || typeof value !== 'object' || Array.isArray(value) ||
      ![Object.prototype, null].includes(Object.getPrototypeOf(value))) throw new TypeError(`${name} must be a plain data record`);
  const keys = Reflect.ownKeys(value), out = {};
  if (keys.length > allowed.length) throw new TypeError(`unknown ${name} fields`);
  for (const key of keys) {
    const property = own(value, key);
    if (typeof key !== 'string' || !allowed.includes(key) || !property || !Object.hasOwn(property, 'value'))
      throw new TypeError(`unknown ${name} field or accessor`);
    out[key] = property.value;
  }
  if (required.some(key => !Object.hasOwn(out, key))) throw new TypeError(`missing ${name} field`);
  return out;
}
function text(value, maximum, name, budget) {
  if (typeof value !== 'string' || !value.length || value.length > maximum || /[\x00-\x1f\x7f]/.test(value))
    throw new TypeError(`${name} must be bounded nonempty text without control characters`);
  budget?.(new TextEncoder().encode(value).length + 4);
  return value;
}
function list(value, maximum, name, transform, minimum = 0) {
  if (!Array.isArray(value) || Object.getPrototypeOf(value) !== Array.prototype) throw new TypeError(`${name} must be a plain array`);
  const length = own(value, 'length')?.value;
  if (!Number.isSafeInteger(length) || length < minimum || length > maximum) throw new RangeError(`${name} count exceeds limits`);
  const keys = Reflect.ownKeys(value);
  if (keys.length !== length + 1 || keys.some(key => key !== 'length' &&
      (typeof key !== 'string' || !/^(0|[1-9][0-9]*)$/.test(key) || Number(key) >= length))) throw new TypeError(`${name} must contain only dense data elements`);
  const result = new Array(length);
  for (let index = 0; index < length; index++) {
    const property = own(value, String(index));
    if (!property || !Object.hasOwn(property, 'value')) throw new TypeError(`${name} accessors and holes are forbidden`);
    result[index] = transform(property.value, index);
  }
  return result;
}
function unique(values, name) {
  if (new Set(values).size !== values.length) throw new TypeError(`duplicate ${name}`);
  return values;
}
function name(value, label, budget) {
  const valueText = text(value, 214, label, budget);
  if (!npmPattern.test(valueText) || ['node_modules', 'favicon.ico'].includes(valueText)) throw new TypeError(`${label} must be a safe lowercase npm name`);
  return valueText;
}
function version(value, budget) {
  const result = text(value, 64, 'version', budget);
  const match = /^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*))?(?:\+([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*))?$/.exec(result);
  if (!match || match[4]?.split('.').some(part => /^[0-9]+$/.test(part) && part.length > 1 && part[0] === '0')) throw new TypeError('version must be exact SemVer');
  return result;
}
function repository(value, budget) {
  const result = text(value, 201, 'repository', budget);
  if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}\/[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$/.test(result) || result.split('/').some(part => part === '.' || part === '..'))
    throw new TypeError('repository must be owner/name');
  return result;
}
function directory(value, budget) {
  const result = text(value, 512, 'directory', budget);
  if (result.split('/').some(part => !/^[A-Za-z0-9_.-]+$/.test(part) || part === '.' || part === '..')) throw new TypeError('directory must be a relative repository path without traversal');
  return result;
}
function contract(value, budget) {
  const result = text(value, 128, 'contract identifier', budget);
  if (!/^[a-z][a-z0-9.-]*(?:\/[a-z][a-z0-9.-]*)*\/[1-9][0-9]*$/.test(result)) throw new TypeError('contract identifier must be an explicit versioned schema ID');
  return result;
}
function links(value, budget) {
  const raw = record(value, ['npm', 'source'], [], 'links'), result = {};
  if (!Object.keys(raw).length) throw new TypeError('links must declare npm or source');
  for (const key of ['npm', 'source']) if (Object.hasOwn(raw, key)) {
    const item = text(raw[key], 2048, `${key} URL`, budget);
    let url;
    try { url = new URL(item); } catch { throw new TypeError('invalid link URL'); }
    const hosts = key === 'npm' ? ['www.npmjs.com', 'npmjs.com'] : ['github.com'];
    if (url.protocol !== 'https:' || !hosts.includes(url.hostname) || url.username || url.password || url.port || url.search)
      throw new TypeError('catalog links must use approved HTTPS hosts without credentials or queries');
    result[key] = url.href;
  }
  return result;
}
function byteBudget() {
  let estimated = 256;
  return size => { estimated += size; if (estimated > CONTRACT_LIMITS.maxBytes) throw new RangeError('contract data exceeds byte budget'); };
}
function finish(value) {
  if (new TextEncoder().encode(JSON.stringify(value)).length > CONTRACT_LIMITS.maxBytes) throw new RangeError('contract data exceeds canonical byte budget');
  return value;
}

/** Validate and own catalog data. Published status is a declaration, not proof of namespace ownership. */
export function validateCatalog(value) {
  const raw = record(value, ['schema', 'repository', 'packages', 'profiles'], ['schema', 'repository', 'packages', 'profiles'], 'catalog');
  if (raw.schema !== CONTRACTS.catalog) throw new TypeError('unsupported constellation catalog schema');
  const budget = byteBudget(), packages = list(raw.packages, CONTRACT_LIMITS.maxPackages, 'packages', value => {
    const entry = record(value, ['name', 'role', 'family', 'status', 'version', 'repository', 'directory', 'description', 'contracts', 'links', 'proposedScope'],
      ['name', 'role', 'family', 'status', 'version', 'repository', 'description'], 'package');
    if (!roles.includes(entry.role) || !['published', 'candidate'].includes(entry.status)) throw new TypeError('invalid package role or status');
    const family = text(entry.family, 48, 'family', budget);
    if (!/^[a-z][a-z0-9-]{0,47}$/.test(family)) throw new TypeError('family must be a bounded lowercase identifier');
    const result = { name: name(entry.name, 'package name', budget), role: entry.role, family, status: entry.status,
      version: version(entry.version, budget), repository: repository(entry.repository, budget), description: text(entry.description, 2000, 'description', budget) };
    if (Object.hasOwn(entry, 'directory')) result.directory = directory(entry.directory, budget);
    if (Object.hasOwn(entry, 'contracts')) result.contracts = unique(list(entry.contracts, CONTRACT_LIMITS.maxContracts, 'contracts', item => contract(item, budget)), 'contract identifiers');
    if (Object.hasOwn(entry, 'links')) result.links = links(entry.links, budget);
    if (Object.hasOwn(entry, 'proposedScope')) {
      const scope = text(entry.proposedScope, 100, 'proposed scope', budget);
      if (!new RegExp(`^@${npmPart}$`).test(scope)) throw new TypeError('proposedScope must be an npm scope, not a published-name claim');
      result.proposedScope = scope;
    }
    budget(128);
    return result;
  }, 1);
  unique(packages.map(entry => entry.name), 'package names');
  const known = new Set(packages.map(entry => entry.name));
  const profiles = list(raw.profiles, CONTRACT_LIMITS.maxProfiles, 'profiles', value => {
    const entry = record(value, ['id', 'description', 'packages'], ['id', 'description', 'packages'], 'profile');
    const id = text(entry.id, 48, 'profile id', budget);
    if (!/^[a-z][a-z0-9-]{0,47}$/.test(id)) throw new TypeError('invalid profile id');
    const members = unique(list(entry.packages, CONTRACT_LIMITS.maxPackages, 'profile packages', value => {
      const member = name(value, 'profile package name', budget);
      if (!known.has(member)) throw new TypeError('profile refers to an unknown package');
      return member;
    }, 1), 'profile members');
    return { id, description: text(entry.description, 2000, 'profile description', budget), packages: members };
  });
  unique(profiles.map(profile => profile.id), 'profile ids');
  return finish({ schema: CONTRACTS.catalog, repository: repository(raw.repository, budget), packages, profiles });
}

/** Metadata contract only: does not evaluate a model, authorize promotion or prove compatibility. */
export function validateModuleManifest(value) {
  const raw = record(value, ['schema', 'name', 'version', 'role', 'contracts', 'integrations'], ['schema', 'name', 'version', 'role', 'contracts', 'integrations'], 'module manifest');
  if (raw.schema !== CONTRACTS.module || !roles.includes(raw.role)) throw new TypeError('unsupported module schema or role');
  const budget = byteBudget();
  return finish({ schema: CONTRACTS.module, name: name(raw.name, 'module name', budget), version: version(raw.version, budget), role: raw.role,
    contracts: unique(list(raw.contracts, CONTRACT_LIMITS.maxContracts, 'contracts', item => contract(item, budget), 1), 'contract identifiers'),
    integrations: unique(list(raw.integrations, CONTRACT_LIMITS.maxIntegrations, 'integrations', item => name(item, 'integration name', budget)), 'integrations') });
}
