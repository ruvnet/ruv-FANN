#!/usr/bin/env node
/** Validate actual npm entry points; a manifest alone is not a working package. */
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { pathToFileURL } = require('node:url');

function validatePackage(basePath = path.resolve(__dirname, '..')) {
  const pkg = JSON.parse(fs.readFileSync(path.join(basePath, 'package.json'), 'utf8'));
  const errors = [];
  const paths = new Set();
  const invalid = new Set();
  const collect = value => {
    if (typeof value === 'string') paths.add(value);
    else if (value && typeof value === 'object') Object.values(value).forEach(collect);
  };
  for (const field of ['main', 'module', 'browser', 'types', 'bin', 'exports']) collect(pkg[field]);
  if (!pkg.main) errors.push('Missing main entry point');
  for (const item of paths) {
    const absolute = path.resolve(basePath, item);
    const relative = path.relative(basePath, absolute);
    if (relative.startsWith('..' + path.sep) || relative === '..' || path.isAbsolute(relative)) {
      invalid.add(item);
      errors.push(`Entry point escapes the package: ${item}`);
      continue;
    }
    if (!fs.existsSync(absolute) || !fs.statSync(absolute).isFile()) {
      invalid.add(item);
      errors.push(`Missing entry point: ${item}`);
      continue;
    }
    const real = path.relative(fs.realpathSync(basePath), fs.realpathSync(absolute));
    if (real === '..' || real.startsWith('..' + path.sep) || path.isAbsolute(real)) {
      invalid.add(item);
      errors.push(`Entry point symlink escapes the package: ${item}`);
      continue;
    }
    if (item.endsWith('.wasm')) {
      const fd = fs.openSync(absolute, 'r');
      const header = Buffer.alloc(8);
      try {
        const size = fs.readSync(fd, header, 0, 8, 0);
        if (size !== 8 || !header.equals(Buffer.from([0, 97, 115, 109, 1, 0, 0, 0]))) {
          errors.push(`Invalid WebAssembly header: ${item}`);
        }
      } finally { fs.closeSync(fd); }
    }
  }
  if (pkg.main && !invalid.has(pkg.main) && fs.existsSync(path.resolve(basePath, pkg.main))) {
    const result = spawnSync(process.execPath, ['-e', 'require(process.argv[1])', path.resolve(basePath, pkg.main)], {
      encoding: 'utf8', timeout: 10_000, maxBuffer: 64 * 1024, cwd: basePath,
    });
    if (result.status !== 0 || result.error) {
      errors.push(`Main entry point cannot load: ${(result.error?.message || result.stderr || 'child process failed').slice(0, 1000)}`);
    }
  }
  if (pkg.name === 'cuda-wasm') {
    const functions = ['transpileCuda', 'transpileFile', 'getCompilerVersion', 'getVersion', 'CudaWasmError'];
    for (const [kind, entry] of [['cjs', pkg.main], ['esm', pkg.module]]) {
      if (!entry || invalid.has(entry) || !fs.existsSync(path.resolve(basePath, entry))) continue;
      const expected = JSON.stringify(functions);
      const code = kind === 'esm'
        ? 'const api = await import(process.argv[1]); for (const name of JSON.parse(process.argv[2])) { if (typeof api[name] !== "function") process.exit(1); } if (api.getVersion() !== process.argv[3]) process.exit(1);'
        : 'const api = require(process.argv[1]); for (const name of JSON.parse(process.argv[2])) { if (typeof api[name] !== "function") process.exit(1); } if (api.getVersion() !== process.argv[3]) process.exit(1);';
      const command = kind === 'esm' ? ['--input-type=module', '-e', code] : ['-e', code];
      command.push(kind === 'esm' ? pathToFileURL(path.resolve(basePath, entry)).href : path.resolve(basePath, entry), expected, pkg.version);
      const result = spawnSync(process.execPath, command, { encoding: 'utf8', timeout: 10000, maxBuffer: 65536, cwd: basePath });
      if (result.status !== 0 || result.error) errors.push(`${kind} source API is not loadable or complete`);
    }
    const cli = pkg.bin?.['cuda-wasm'];
    if (cli && !invalid.has(cli) && fs.existsSync(path.resolve(basePath, cli))) {
      const result = spawnSync(process.execPath, [path.resolve(basePath, cli), '--version'], { encoding: 'utf8', timeout: 10000, maxBuffer: 65536, cwd: basePath });
      if (result.status !== 0 || result.stdout.trim() !== pkg.version) errors.push('CLI version entry point is not functional');
    }
  }
  return errors;
}

if (require.main === module) {
  try {
    const errors = validatePackage();
    if (errors.length) {
      console.error('Package artifacts are incomplete:\n' + errors.map(error => `- ${error}`).join('\n'));
      process.exitCode = 1;
    } else console.log('All declared entry points exist and the main module loads.');
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
module.exports = { validatePackage };
