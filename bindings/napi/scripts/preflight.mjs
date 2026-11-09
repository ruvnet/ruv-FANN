/** Fail closed on ambient compiler/target policies before invoking Cargo. */
import { lstat, readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { homedir } from 'node:os';

export function rejectOverrides(environment = process.env) {
  for (const [key, value] of Object.entries(environment)) {
    if (!value) continue;
    if (['RUSTC', 'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER', 'RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS',
      'CARGO_BUILD_TARGET', 'CARGO_BUILD_RUSTC', 'CARGO_BUILD_RUSTC_WRAPPER',
      'CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER', 'CARGO_BUILD_RUSTFLAGS'].includes(key) ||
      /^CARGO_TARGET_.*_(?:LINKER|RUSTFLAGS|RUNNER)$/.test(key))
      throw new Error(`Native build refuses ambient compiler/target/wrapper override ${key}`);
  }
}

export function lintOnlyConfig(text) {
  const clean = text.replace(/#[^\n\r]*/g, '').trim();
  const match = /^\[build\]\s*\[target\.'cfg\(all\(\)\)'\]\s*rustflags\s*=\s*\[([\s\S]*)\]\s*$/.exec(clean);
  if (!match) return false;
  const strings = match[1].match(/"(?:[^"\\]|\\.)*"/g) ?? [];
  const residue = match[1].replace(/"(?:[^"\\]|\\.)*"/g, '');
  if (!/^[\s,]*$/.test(residue) || strings.length % 2 !== 0) return false;
  let flags;
  try { flags = strings.map(value => JSON.parse(value)); } catch { return false; }
  for (let index = 0; index < flags.length; index += 2)
    if (!['-A', '-W', '-D', '-F'].includes(flags[index]) || !/^[A-Za-z_][A-Za-z0-9_:]*$/.test(flags[index + 1])) return false;
  return true;
}

export async function configPreflight(directory, repository, environment = process.env) {
  rejectOverrides(environment);
  const candidates = new Set();
  for (let current = resolve(directory);;) {
    for (const name of ['config', 'config.toml']) candidates.add(join(current, '.cargo', name));
    const parent = dirname(current); if (parent === current) break; current = parent;
  }
  const cargoHome = resolve(environment.CARGO_HOME || join(homedir(), '.cargo'));
  for (const name of ['config', 'config.toml']) candidates.add(join(cargoHome, name));
  const checked = {};
  for (const path of candidates) {
    let info;
    try { info = await lstat(path); } catch (error) { if (error.code === 'ENOENT') continue; throw error; }
    if (!info.isFile() || info.isSymbolicLink() || info.size > 65536) throw new Error(`Unsafe Cargo config file: ${path}`);
    const text = await readFile(path, 'utf8');
    if (path !== join(repository, '.cargo/config.toml') || !lintOnlyConfig(text))
      throw new Error(`Native build refuses ambient/non-lint Cargo config: ${path}`);
    checked[path] = text;
  }
  return checked;
}
