import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile, mkdtemp, writeFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { spawnSync, execFileSync } from 'node:child_process';
import { validateOIDCPublishContext, createTokenlessPublishEnvironment, WSP_OIDC_POLICY } from './constellation-oidc.mjs';

const sourceSha = 'a'.repeat(40), oidcCapability = 'test-only-ephemeral-capability';
function context() {
  return { sourceSha, checkoutSha: sourceSha, nodeVersion: 'v24.22.1', npmVersion: '11.9.0', env: {
    GITHUB_ACTIONS: 'true', GITHUB_REPOSITORY: 'ruvnet/ruv-FANN', GITHUB_REF: 'refs/heads/main',
    GITHUB_WORKFLOW_REF: 'ruvnet/ruv-FANN/.github/workflows/protected-wsp-release.yml@refs/heads/main',
    GITHUB_WORKFLOW: 'protected-release', GITHUB_JOB: 'publish', GITHUB_EVENT_NAME: 'workflow_dispatch', GITHUB_SHA: sourceSha,
    ACTIONS_ID_TOKEN_REQUEST_URL: 'https://run-actions-1.actions.githubusercontent.com/example/_apis/distributedtask/hubs/build/plans/plan/jobs/job/idtoken?api-version=2.0',
    ACTIONS_ID_TOKEN_REQUEST_TOKEN: oidcCapability,
    NODE_AUTH_TOKEN: '', NPM_TOKEN: ''
  } };
}

test('protected stable OIDC context accepts minimum tool versions without claiming token attestation', () => {
  const input = context();input.nodeVersion = 'v22.14.0';input.npmVersion = '11.5.1';
  const proof = validateOIDCPublishContext(input);
  assert.equal(proof.mode, 'oidc-only');assert.equal(proof.evidence, 'configuration-validated');
  assert.equal(proof.sourceSha, sourceSha);assert.equal(proof.repository, WSP_OIDC_POLICY.repository);
  assert.equal(Object.isFrozen(proof), true);
  const text = JSON.stringify(proof);
  assert.equal(text.includes(oidcCapability), false);assert.equal(text.includes('idtoken?'), false);
  assert.equal('ACTIONS_ID_TOKEN_REQUEST_TOKEN' in proof, false);
  input.env.GITHUB_REPOSITORY = 'changed/repo';assert.equal(proof.repository, 'ruvnet/ruv-FANN');
});

test('runtime floor rejects unsupported, prerelease and malformed versions', () => {
  for (const value of ['v20.20.2', 'v22.13.0', 'v22.14.0-rc.1', 'v022.14.0', 'v24', null]) {
    const input=context();input.nodeVersion=value;assert.throws(() => validateOIDCPublishContext(input));
  }
  for (const value of ['10.9.0', '11.5.0', '11.5.1-beta.1', '11.09.0', 'latest', null]) {
    const input=context();input.npmVersion=value;assert.throws(() => validateOIDCPublishContext(input));
  }
  const input=context();input.nodeVersion='v26.1.0';input.npmVersion='12.0.0';validateOIDCPublishContext(input);
});

test('local, fork, PR, wrong job/workflow/event, unreviewed source and stale checkout all fail closed', () => {
  const changes = { GITHUB_ACTIONS: 'false', GITHUB_REPOSITORY: 'fork/ruv-FANN', GITHUB_REF: 'refs/pull/17/merge',
    GITHUB_WORKFLOW_REF: 'ruvnet/ruv-FANN/.github/workflows/other.yml@refs/heads/main', GITHUB_WORKFLOW: 'qualification',
    GITHUB_JOB: 'qualify', GITHUB_EVENT_NAME: 'pull_request_target', GITHUB_SHA: 'b'.repeat(40) };
  for (const [key,value] of Object.entries(changes)) {
    const input=context();input.env[key]=value;assert.throws(() => validateOIDCPublishContext(input));
  }
  const wrongCheckout=context();wrongCheckout.checkoutSha='b'.repeat(40);assert.throws(() => validateOIDCPublishContext(wrongCheckout));
  const invalidSha=context();invalidSha.sourceSha='not-reviewed';assert.throws(() => validateOIDCPublishContext(invalidSha));
  const absent=context();delete absent.env.GITHUB_EVENT_NAME;assert.throws(() => validateOIDCPublishContext(absent));
});

test('OIDC endpoint and capability must exist and cannot select an external or credentialed host', () => {
  for (const url of ['http://run-actions.actions.githubusercontent.com/token', 'https://actions.githubusercontent.com.evil.test/token',
    'https://user:secret@run-actions.actions.githubusercontent.com/token', 'https://run-actions.actions.githubusercontent.com:8443/token',
    'https://run-actions.actions.githubusercontent.com/token#fragment', 'not a URL', '', null]) {
    const input=context();input.env.ACTIONS_ID_TOKEN_REQUEST_URL=url;assert.throws(() => validateOIDCPublishContext(input));
  }
  for (const value of ['', undefined, null, 'line\nbreak']) {
    const input=context();input.env.ACTIONS_ID_TOKEN_REQUEST_TOKEN=value;assert.throws(() => validateOIDCPublishContext(input));
  }
});

test('credential fallback is refused in all supported environment spellings without disclosing values', () => {
  for (const key of ['NPM_TOKEN', 'NODE_AUTH_TOKEN', 'npm_token', 'npm_config__auth', 'NPM_CONFIG__AUTHTOKEN',
    'npm_config_//registry.npmjs.org/:_authToken', 'npm_config_//registry.npmjs.org/:_password', 'npm_config_username']) {
    const input=context();input.env[key]='credential-value-must-not-appear';
    assert.throws(() => validateOIDCPublishContext(input), error => {
      assert.equal(error.message.includes('credential-value-must-not-appear'), false);
      return /refuses npm credential fallback/.test(error.message);
    });
  }
});

test('data-only configuration never invokes getters and refuses unknown fields', () => {
  let calls=0;
  const input=context();Object.defineProperty(input.env,'NODE_AUTH_TOKEN',{get(){calls++;return 'hidden';}});
  assert.throws(() => validateOIDCPublishContext(input), /data fields/);assert.equal(calls,0);
  const top=context();Object.defineProperty(top,'sourceSha',{get(){calls++;return sourceSha;}});
  assert.throws(() => validateOIDCPublishContext(top));assert.equal(calls,0);
  assert.throws(() => validateOIDCPublishContext({...context(),allowTokenFallback:true}));
  const wrongType=context();wrongType.env.UNRELATED={toString(){calls++;return 'side effect';}};
  assert.throws(() => validateOIDCPublishContext(wrongType));assert.equal(calls,0);
});

test('isolated publish environment removes token/config fallback, retains proxy/CA and OIDC, and owns its copy', () => {
  const input={...context().env, NPM_TOKEN:'npm-secret', GH_TOKEN:'github-secret', GITHUB_TOKEN:'github-secret',
    npm_config__auth:'basic-secret', 'npm_config_//registry.npmjs.org/:_authToken':'other-secret',
    npm_config_userconfig:'/ambient/secret.npmrc', NPM_CONFIG_GLOBALCONFIG:'/ambient/global.npmrc',
    npm_config_registry:'https://untrusted.invalid', npm_config_cache:'/ambient/cache',
    HTTPS_PROXY:'http://approved-proxy:8080', NODE_EXTRA_CA_CERTS:'/approved/ca.pem'};
  const env=createTokenlessPublishEnvironment(input,{userConfig:'/isolated/user.npmrc',globalConfig:'/isolated/global.npmrc',cache:'/isolated/cache'});
  for(const key of ['NPM_TOKEN','NODE_AUTH_TOKEN','GH_TOKEN','GITHUB_TOKEN','npm_config__auth','npm_config_//registry.npmjs.org/:_authToken','npm_config_userconfig','npm_config_registry','npm_config_cache'])
    assert.equal(Object.hasOwn(env,key),false,key);
  assert.equal(env.NPM_CONFIG_USERCONFIG,'/isolated/user.npmrc');assert.equal(env.NPM_CONFIG_GLOBALCONFIG,'/isolated/global.npmrc');
  assert.equal(env.NPM_CONFIG_REGISTRY,'https://registry.npmjs.org');assert.equal(env.NPM_CONFIG_PROVENANCE,'true');
  assert.equal(env.NPM_CONFIG_CACHE,'/isolated/cache');assert.equal(env.HTTPS_PROXY,input.HTTPS_PROXY);assert.equal(env.NODE_EXTRA_CA_CERTS,input.NODE_EXTRA_CA_CERTS);
  assert.equal(env.ACTIONS_ID_TOKEN_REQUEST_TOKEN,oidcCapability);assert.equal(input.NPM_TOKEN,'npm-secret');
  env.PATH='changed';assert.equal(input.PATH,undefined);
});

test('actual publication entry point rejects a foreign context before registry or gh operations', async () => {
  const directory=await mkdtemp(join(tmpdir(),'wsp-oidc-entry-'));
  try {
    const checkoutSha=execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8',cwd:new URL('..',import.meta.url)}).trim();
    const file=join(directory,'seal.json');await writeFile(file,JSON.stringify({sourceSha:checkoutSha}));
    const env={...process.env,...context().env,GITHUB_REPOSITORY:'foreign/forbidden',GITHUB_SHA:checkoutSha};
    const result=spawnSync(process.execPath,[new URL('./wsp-release-lane.mjs',import.meta.url).pathname,'publish','--seal',file,'--output',join(directory,'publication.json')],
      {encoding:'utf8',env,timeout:10000});
    assert.notEqual(result.status,0);assert.match(result.stderr,/OIDC publishing requires (?:the exact protected|Node >=22\.14\.0)/);
    assert.equal(result.stderr.includes(oidcCapability),false);
    await assert.rejects(readFile(join(directory,'publication.json')),error=>error.code==='ENOENT');
  } finally {await rm(directory,{recursive:true,force:true});}
});

test('protected workflow pins tools, grants OIDC and preserves exact-archive qualification without npm secrets', async () => {
  const workflow=await readFile(new URL('../.github/workflows/protected-wsp-release.yml',import.meta.url),'utf8');
  const lane=await readFile(new URL('./wsp-release-lane.mjs',import.meta.url),'utf8');
  assert.match(workflow,/id-token: write/);assert.match(workflow,/node-version: '24\.22\.1'/);assert.match(workflow,/npm@11\.9\.0/);
  assert.doesNotMatch(workflow,/secrets\.NPM_TOKEN/);assert.match(workflow,/NODE_AUTH_TOKEN: ''/);assert.match(workflow,/NPM_TOKEN: ''/);
  assert.match(workflow,/wsp-candidate-download\.mjs --expected-sha/);
  assert.match(lane,/validateOIDCPublishContext\(/);assert.match(lane,/await verifySeal\(seal, dirname\(sealPath\)\)/);
  assert.match(lane,/'--ignore-scripts', '--access', 'public', '--provenance'/);
  assert.match(lane,/createTokenlessPublishEnvironment\(process\.env/);
  assert.doesNotMatch(lane,/Missing NPM_TOKEN/);
});
