/** Proves focused bounds and lifecycle tests fail against known broken SDK mutants. */
import { execFileSync, spawnSync } from 'node:child_process';
import { cp, mkdtemp, mkdir, readFile, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('../', import.meta.url));
const temporary = await mkdtemp(join(tmpdir(), 'fann-sdk-mutants-'));
try {
  await cp(join(root, 'packages/core/src'), join(temporary, 'src'), { recursive: true });
  await cp(join(root, 'packages/core/test'), join(temporary, 'test'), { recursive: true });
  await cp(join(root, 'packages/core/package.json'), join(temporary, 'package.json'));
  await mkdir(join(temporary, 'node_modules'));
  await symlink(join(root, 'packages/wasm'), join(temporary, 'node_modules/ruv-fann-wasm'));
  execFileSync(process.execPath, ['--test', '--test-reporter=tap', 'test/runtime.test.mjs'], { cwd: temporary, stdio: 'pipe' });
  const mutations = [
    { name: 'remove native byte preflight', file: 'src/validation.js',
      from: 'if (!Number.isSafeInteger(estimated) || estimated > LIMITS.maxBytes)', to: 'if (false)',
      test: 'storage preflight rejects' },
    { name: 'allow epoch 1001', file: 'src/validation.js', from: 'maxEpochs: 1000', to: 'maxEpochs: 1001',
      test: 'public model/options validation' },
    { name: 'remove total training compute cap', file: 'src/index.js',
      from: 'if (backwardWork * (epochs + 1) + updateWork * epochs > LIMITS.maxTrainingWork)', to: 'if (false)',
      test: 'public model/options validation' },
    { name: 'remove disposed model guard', file: 'src/index.js',
      from: "if (disposed) throw new Error('prepared model is disposed');", to: '',
      test: 'prepared batches equal rows' }
  ];
  for (const mutation of mutations) {
    const file = join(temporary, mutation.file), original = await readFile(file, 'utf8');
    if (!original.includes(mutation.from)) throw new Error(`Mutation source missing: ${mutation.name}`);
    await writeFile(file, original.replace(mutation.from, mutation.to));
    const result = spawnSync(process.execPath, ['--test', '--test-reporter=tap', `--test-name-pattern=${mutation.test}`, 'test/runtime.test.mjs'],
      { cwd: temporary, encoding: 'utf8' });
    await writeFile(file, original);
    if (result.error || result.status !== 1 || !result.stdout.includes('not ok')) {
      throw new Error(`Mutant was not killed by assertion: ${mutation.name}\n${result.stdout}\n${result.stderr}`);
    }
    console.log(`PASS: killed mutant '${mutation.name}' with targeted regression assertions.`);
  }
} finally { await rm(temporary, { recursive: true, force: true }); }
