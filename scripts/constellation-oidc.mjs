/** Local configuration guards for npm trusted publishing; npm verifies the actual OIDC identity. */
export const WSP_OIDC_POLICY = Object.freeze({
  repository: 'ruvnet/ruv-FANN',
  workflowFile: 'protected-wsp-release.yml',
  workflowName: 'protected-release',
  job: 'publish',
  eventName: 'workflow_dispatch',
  ref: 'refs/heads/main'
});

const own = Object.getOwnPropertyDescriptor;
function data(object, key, label) {
  const descriptor = own(object, key);
  if (!descriptor) return undefined;
  if (!Object.hasOwn(descriptor, 'value')) throw new TypeError(`${label} must contain data fields only`);
  return descriptor.value;
}
function plain(value, allowed, required, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value) ||
      ![Object.prototype, null].includes(Object.getPrototypeOf(value))) throw new TypeError(`${label} must be a plain data object`);
  const out = Object.create(null);
  for (const key of Reflect.ownKeys(value)) {
    if (typeof key !== 'string' || !allowed.includes(key)) throw new TypeError(`unknown ${label} field`);
    out[key] = data(value, key, label);
  }
  if (required.some(key => !Object.hasOwn(out, key))) throw new TypeError(`missing ${label} field`);
  return out;
}
function version(value, label) {
  if (typeof value !== 'string' || value.length > 40 || !/^v?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(value))
    throw new TypeError(`${label} must be a stable three-part version`);
  const numbers = value.replace(/^v/, '').split('.').map(Number);
  if (numbers.some(value => !Number.isSafeInteger(value))) throw new TypeError(`invalid ${label} version`);
  return numbers;
}
function atLeast(actual, minimum) {
  for (let index = 0; index < minimum.length; index++) {
    if (actual[index] > minimum[index]) return true;
    if (actual[index] < minimum[index]) return false;
  }
  return true;
}
function fallbackKey(key) {
  const normalized = key.toLowerCase();
  if (normalized === 'npm_token' || normalized === 'node_auth_token') return true;
  return normalized.startsWith('npm_config_') && /(?:^|[_/:])(?:auth|authtoken|password|username)(?:$|[_:])/.test(normalized.slice('npm_config_'.length));
}
function envValue(env, key) {
  const value = data(env, key, 'publish environment');
  if (value !== undefined && typeof value !== 'string') throw new TypeError('publish environment values must be strings');
  return value;
}
function string(value, label, maximum = 4096) {
  if (typeof value !== 'string' || value.length < 1 || value.length > maximum || /[\x00-\x1f\x7f]/.test(value))
    throw new TypeError(`${label} must be a bounded nonempty string`);
  return value;
}
function policySnapshot(value) {
  const out = plain(value, ['repository', 'workflowFile', 'workflowName', 'job', 'eventName', 'ref'],
    ['repository', 'workflowFile', 'workflowName', 'job', 'eventName', 'ref'], 'OIDC policy');
  if (typeof out.repository !== 'string' || !/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(out.repository)) throw new TypeError('invalid OIDC repository policy');
  if (typeof out.workflowFile !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\.ya?ml$/.test(out.workflowFile)) throw new TypeError('invalid OIDC workflow policy');
  for (const key of ['workflowName', 'job', 'eventName', 'ref']) string(out[key], `OIDC ${key} policy`, 256);
  if (!/^refs\/heads\/[A-Za-z0-9_.\/-]+$/.test(out.ref)) throw new TypeError('OIDC policy must select a branch');
  return out;
}

/**
 * Configuration guard only. Environment strings are not an attestation; npm must
 * exchange the real GitHub token and match the package's trusted-publisher registration.
 */
export function validateOIDCPublishContext(configuration, policy = WSP_OIDC_POLICY) {
  const input = plain(configuration, ['env', 'sourceSha', 'checkoutSha', 'nodeVersion', 'npmVersion'],
    ['env', 'sourceSha', 'checkoutSha', 'nodeVersion', 'npmVersion'], 'OIDC publish context');
  const expected = policySnapshot(policy), env = input.env;
  if (!env || typeof env !== 'object' || Array.isArray(env)) throw new TypeError('publish environment is required');
  for (const key of Reflect.ownKeys(env)) {
    if (typeof key !== 'string') throw new TypeError('publish environment contains invalid fields');
    const value = envValue(env, key);
    if (fallbackKey(key) && value !== undefined && value !== '')
      throw new Error('Tokenless publishing refuses npm credential fallback; remove npm auth environment values');
  }
  if (!atLeast(version(input.nodeVersion, 'Node'), [22, 14, 0])) throw new Error('npm OIDC publishing requires Node >=22.14.0');
  if (!atLeast(version(input.npmVersion, 'npm'), [11, 5, 1])) throw new Error('npm OIDC publishing requires npm >=11.5.1');
  if (typeof input.sourceSha !== 'string' || !/^[a-f0-9]{40}$/.test(input.sourceSha) || input.checkoutSha !== input.sourceSha)
    throw new Error('OIDC publishing requires the exact reviewed checkout SHA');
  const workflowRef = `${expected.repository}/.github/workflows/${expected.workflowFile}@${expected.ref}`;
  const checks = { GITHUB_ACTIONS: 'true', GITHUB_REPOSITORY: expected.repository, GITHUB_REF: expected.ref,
    GITHUB_WORKFLOW_REF: workflowRef, GITHUB_WORKFLOW: expected.workflowName, GITHUB_JOB: expected.job,
    GITHUB_EVENT_NAME: expected.eventName, GITHUB_SHA: input.sourceSha };
  for (const [key, value] of Object.entries(checks)) if (envValue(env, key) !== value)
    throw new Error('OIDC publishing requires the exact protected repository, workflow, job, event, branch and source');
  const urlValue = string(envValue(env, 'ACTIONS_ID_TOKEN_REQUEST_URL'), 'GitHub OIDC request URL', 8192);
  let url;
  try { url = new URL(urlValue); } catch { throw new Error('invalid GitHub OIDC request endpoint'); }
  if (url.protocol !== 'https:' || url.username || url.password || url.hash || url.port ||
      !(url.hostname === 'actions.githubusercontent.com' || url.hostname.endsWith('.actions.githubusercontent.com')))
    throw new Error('GitHub OIDC request endpoint must use HTTPS on the GitHub Actions token host');
  string(envValue(env, 'ACTIONS_ID_TOKEN_REQUEST_TOKEN'), 'GitHub OIDC request capability', 32768);
  return Object.freeze({ schema: 'ruvnet.oidc-publish-context/1', provider: 'github-actions', mode: 'oidc-only',
    evidence: 'configuration-validated', repository: expected.repository, workflowRef, job: expected.job,
    eventName: expected.eventName, sourceSha: input.sourceSha,
    nodeVersion: input.nodeVersion, npmVersion: input.npmVersion });
}

/**
 * Build a child environment for an isolated auth-free npm working directory.
 * Never serialize this object: it preserves the short-lived Actions OIDC capability.
 */
export function createTokenlessPublishEnvironment(environment, configuration) {
  if (!environment || typeof environment !== 'object' || Array.isArray(environment)) throw new TypeError('publish environment is required');
  const settings = plain(configuration, ['userConfig', 'globalConfig', 'cache'], ['userConfig', 'globalConfig', 'cache'], 'npm isolation');
  for (const key of Object.keys(settings)) string(settings[key], `npm ${key} path`);
  const owned = Object.create(null);
  for (const key of Reflect.ownKeys(environment)) {
    if (typeof key !== 'string') throw new TypeError('publish environment contains invalid fields');
    const value = envValue(environment, key);
    if (fallbackKey(key) || ['gh_token', 'github_token', 'npm_config_userconfig', 'npm_config_globalconfig', 'npm_config_cache', 'npm_config_registry', 'npm_config_provenance'].includes(key.toLowerCase())) continue;
    if (value !== undefined) owned[key] = value;
  }
  owned.NPM_CONFIG_USERCONFIG = settings.userConfig;
  owned.NPM_CONFIG_GLOBALCONFIG = settings.globalConfig;
  owned.NPM_CONFIG_CACHE = settings.cache;
  owned.NPM_CONFIG_REGISTRY = 'https://registry.npmjs.org';
  owned.NPM_CONFIG_PROVENANCE = 'true';
  return owned;
}
