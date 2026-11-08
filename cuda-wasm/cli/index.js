#!/usr/bin/env node
'use strict';
const { parseArgs } = require('node:util');
const fs = require('node:fs/promises');
const path = require('node:path');
const { transpileFile, getVersion } = require('../dist/index.js');

const help = `cuda-wasm ${getVersion()} — generate Rust or WGSL source with cuda2rust

Usage: cuda-wasm transpile <input.cu> [options]

  -t, --target <rust|wgsl>  Output language (default: rust)
  -o, --output <path>      Write a new file instead of stdout
  -b, --binary <path>      Trusted cuda2rust executable (or CUDA2RUST_BINARY)
      --timeout <ms>       Per-process timeout, up to 120000 (default: 30000)
      --force              Allow overwriting an output file
  -v, --version            Print this npm version
  -h, --help               Show this help

Requires cuda2rust >=0.2.0. Emits source text only; does not compile or run kernels.
`;

async function main() {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      target: { type: 'string', short: 't' }, output: { type: 'string', short: 'o' },
      binary: { type: 'string', short: 'b' }, timeout: { type: 'string' },
      force: { type: 'boolean' }, version: { type: 'boolean', short: 'v' },
      help: { type: 'boolean', short: 'h' },
    },
  });
  if (values.help || positionals.length === 0 && !values.version) {
    process.stdout.write(help);
    return;
  }
  if (values.version) { process.stdout.write(`${getVersion()}\n`); return; }
  if (positionals.length !== 2 || positionals[0] !== 'transpile') {
    throw new Error('Expected: cuda-wasm transpile <input.cu>. Use --help for options.');
  }
  const input = positionals[1];
  if (values.output && path.resolve(values.output) === path.resolve(input)) {
    throw new Error('Output must differ from the input file.');
  }
  if (values.force && values.output) {
    const inputStat = await fs.stat(input);
    let outputStat;
    try { outputStat = await fs.stat(values.output); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    if (outputStat && inputStat.dev === outputStat.dev && inputStat.ino === outputStat.ino) {
      throw new Error('Output must not alias the input file.');
    }
  }
  const result = await transpileFile(input, {
    target: values.target ?? 'rust', binary: values.binary,
    timeoutMs: values.timeout === undefined ? 30_000 : Number(values.timeout),
  });
  if (values.output) {
    // Exclusive creation protects an existing file unless the caller opts in.
    await fs.writeFile(values.output, result.code, { flag: values.force ? 'w' : 'wx' });
  } else process.stdout.write(result.code);
}

main().catch(error => {
  process.stderr.write(`${error.code || 'E_CLI'}: ${error.message}\n`);
  if (error.compilerStderr) process.stderr.write(`${error.compilerStderr}\n`);
  process.exitCode = 1;
});
