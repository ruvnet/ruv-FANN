#!/usr/bin/env node
/** Build an actual archive, update the private fixture lock, and install it cleanly. */
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const root = fileURLToPath(new URL('../', import.meta.url));
const fixture = path.join(root, 'test-npm-install-v106');
const temporary = await fs.mkdtemp(path.join(os.tmpdir(), 'ruv-install-'));
const npm = process.env.npm_execpath;
if (!npm) {
  throw new Error('Run this check through npm run test:installation');
}
function run(args, cwd) {
  execFileSync(process.execPath, [npm, ...args], { cwd, stdio: 'inherit',
    env: { ...process.env, npm_config_jobs: '1' } });
}
try {
  run(['pack', '--pack-destination', temporary], root);
  const fixtureManifest = JSON.parse(await fs.readFile(path.join(fixture, 'package.json'), 'utf8'));
  const project = path.join(temporary, 'fixture');
  await fs.mkdir(project);
  await fs.writeFile(path.join(project, 'package.json'), `${JSON.stringify(fixtureManifest, null, 2)}\n`);
  await fs.copyFile(path.join(fixture, 'smoke.mjs'), path.join(project, 'smoke.mjs'));
  run(['install', '--package-lock-only', '--ignore-scripts', '--no-fund'], project);
  await fs.copyFile(path.join(project, 'package-lock.json'), path.join(fixture, 'package-lock.json'));
  run(['ci', '--foreground-scripts', '--no-fund'], project);
  run(['audit', '--audit-level=low'], project);
  run(['test'], project);
} finally {
  await fs.rm(temporary, { recursive: true, force: true });
}
