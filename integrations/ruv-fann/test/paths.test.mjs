import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { repo, externalEvidencePath } from './e2e/fixture.mjs';

test('evidence containment rejects dot-prefixed children, repo ancestors and symlink redirection', async () => {
  await assert.rejects(externalEvidencePath(join(repo, '..pack')), /external/);
  await assert.rejects(externalEvidencePath(repo), /external/);
  await assert.rejects(externalEvidencePath(join(repo, '..')), /external/);
  const directory = await mkdtemp(join(tmpdir(), 'fann-package-path-'));
  try {
    assert.equal(await externalEvidencePath(join(directory, 'new')), join(directory, 'new'));
    await symlink(repo, join(directory, 'redirect'), 'dir');
    await assert.rejects(externalEvidencePath(join(directory, 'redirect', 'new')), /symlink/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
