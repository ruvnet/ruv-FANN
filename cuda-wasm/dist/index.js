'use strict';

const { execFile } = require('node:child_process');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { version } = require('../package.json');

const MAX_INPUT_BYTES = 1024 * 1024;
const MAX_OUTPUT_BYTES = 16 * 1024 * 1024;
const MAX_TIMEOUT_MS = 120_000;

class CudaWasmError extends Error {
  constructor(code, message, details = {}) {
    super(message);
    this.name = 'CudaWasmError';
    this.code = code;
    Object.assign(this, details);
  }
}

function optionsFor(options) {
  if (!options || typeof options !== 'object' || Array.isArray(options)) {
    throw new CudaWasmError('E_OPTIONS', 'Options must be an object.');
  }
  for (const key of Object.keys(options)) {
    if (!['target', 'binary', 'timeoutMs', 'maxOutputBytes'].includes(key)) {
      throw new CudaWasmError('E_OPTIONS', `Unknown option: ${key}`);
    }
  }
  const target = options.target ?? 'rust';
  if (target !== 'rust' && target !== 'wgsl') {
    throw new CudaWasmError('E_TARGET', 'Target must be rust or wgsl; this package emits source text.');
  }
  const binary = options.binary ?? process.env.CUDA2RUST_BINARY ?? 'cuda2rust';
  if (typeof binary !== 'string' || !binary.trim() || binary.includes('\0') || binary.length > 4096) {
    throw new CudaWasmError('E_OPTIONS', 'binary must name a trusted cuda2rust executable.');
  }
  // Batch/shell scripts are not executables on Windows. Never enable a shell.
  if (process.platform === 'win32' && /\.(cmd|bat)$/i.test(binary)) {
    throw new CudaWasmError('E_OPTIONS', 'Use the cuda2rust.exe binary, not a shell script.');
  }
  const timeoutMs = options.timeoutMs ?? 30_000;
  const maxOutputBytes = options.maxOutputBytes ?? 4 * 1024 * 1024;
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > MAX_TIMEOUT_MS) {
    throw new CudaWasmError('E_OPTIONS', `timeoutMs must be an integer from 1 to ${MAX_TIMEOUT_MS}.`);
  }
  if (!Number.isSafeInteger(maxOutputBytes) || maxOutputBytes < 1 || maxOutputBytes > MAX_OUTPUT_BYTES) {
    throw new CudaWasmError('E_OPTIONS', `maxOutputBytes must be an integer from 1 to ${MAX_OUTPUT_BYTES}.`);
  }
  return { binary, target, timeoutMs, maxOutputBytes };
}

function cleanStderr(stderr) {
  return String(stderr || '').slice(0, 8192).replace(/[\u0000-\u0008\u000b-\u001f\u007f]/g, '');
}

function execute(binary, args, options) {
  return new Promise((resolve, reject) => {
    execFile(binary, args, {
      shell: false, windowsHide: true, encoding: 'utf8',
      timeout: options.timeoutMs, killSignal: 'SIGKILL', maxBuffer: options.maxOutputBytes,
    }, (error, stdout, stderr) => {
      if (!error) return resolve(stdout);
      const details = { exitCode: typeof error.code === 'number' ? error.code : undefined,
        compilerStderr: cleanStderr(stderr) };
      if (error.code === 'ENOENT') {
        return reject(new CudaWasmError('E_COMPILER_NOT_FOUND',
          'cuda2rust was not found. Install cuda-rust-wasm 0.2.0 or newer, or set CUDA2RUST_BINARY to a built executable.', details));
      }
      if (error.code === 'ERR_CHILD_PROCESS_STDIO_MAXBUFFER') {
        return reject(new CudaWasmError('E_OUTPUT_LIMIT', 'Compiler output exceeded maxOutputBytes.', details));
      }
      if (error.killed) {
        return reject(new CudaWasmError('E_TIMEOUT', `Compiler exceeded timeoutMs (${options.timeoutMs}).`, details));
      }
      reject(new CudaWasmError('E_COMPILER', 'cuda2rust failed; inspect compilerStderr for details.', details));
    });
  });
}

async function compilerVersion(options) {
  const result = await execute(options.binary, ['--version'], { ...options, maxOutputBytes: 4096 });
  const match = /^cuda2rust (\d+)\.(\d+)\.(\d+)\s*$/.exec(result);
  if (!match || (Number(match[1]) === 0 && Number(match[2]) < 2)) {
    throw new CudaWasmError('E_COMPILER_VERSION', 'Expected cuda2rust 0.2.0 or newer with the source-generation CLI.');
  }
  return `${match[1]}.${match[2]}.${match[3]}`;
}

async function getCompilerVersion(options = {}) {
  return compilerVersion(optionsFor(options));
}

async function transpileCuda(source, options = {}) {
  const settings = optionsFor(options);
  if (typeof source !== 'string' || !source.trim() || !source.isWellFormed()) {
    throw new CudaWasmError('E_INPUT', 'Source must be a nonempty, well-formed UTF-8 string.');
  }
  if (Buffer.byteLength(source, 'utf8') > MAX_INPUT_BYTES) {
    throw new CudaWasmError('E_INPUT_LIMIT', 'CUDA source exceeds the 1 MiB parser limit.');
  }
  const compiler = await compilerVersion(settings);
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'cuda-wasm-'));
  try {
    const input = path.join(directory, 'input.cu');
    await fs.writeFile(input, source, { mode: 0o600, flag: 'wx' });
    const command = settings.target === 'rust' ? 'transpile' : 'generate-wgsl';
    const code = await execute(settings.binary, [command, '--input', input], settings);
    if (!code.trim()) throw new CudaWasmError('E_COMPILER', 'cuda2rust returned empty source.');
    return { code, target: settings.target, compilerVersion: compiler };
  } finally {
    await fs.rm(directory, { recursive: true, force: true });
  }
}

async function transpileFile(inputPath, options = {}) {
  // Validate options before opening a user path or allocating the bounded buffer.
  optionsFor(options);
  if (typeof inputPath !== 'string' || !inputPath || inputPath.includes('\0')) {
    throw new CudaWasmError('E_INPUT', 'inputPath must be a file path.');
  }
  let file;
  let source;
  try {
    const { constants } = require('node:fs');
    file = await fs.open(inputPath, constants.O_RDONLY | (constants.O_NONBLOCK || 0));
    const stat = await file.stat();
    if (!stat.isFile()) throw new CudaWasmError('E_INPUT', 'Input must be a regular file.');
    if (stat.size > MAX_INPUT_BYTES) throw new CudaWasmError('E_INPUT_LIMIT', 'CUDA source exceeds the 1 MiB parser limit.');
    const bytes = Buffer.alloc(MAX_INPUT_BYTES + 1);
    let used = 0;
    while (used < bytes.length) {
      const { bytesRead } = await file.read(bytes, used, bytes.length - used, used);
      if (bytesRead === 0) break;
      used += bytesRead;
    }
    if (used > MAX_INPUT_BYTES) throw new CudaWasmError('E_INPUT_LIMIT', 'CUDA source exceeds the 1 MiB parser limit.');
    try { source = new TextDecoder('utf-8', { fatal: true }).decode(bytes.subarray(0, used)); }
    catch { throw new CudaWasmError('E_INPUT', 'Input file must contain valid UTF-8.'); }
  } catch (error) {
    if (error instanceof CudaWasmError) throw error;
    throw new CudaWasmError('E_INPUT', `Could not read input file (${error.code || 'I/O error'}).`);
  } finally {
    if (file) await file.close();
  }
  return transpileCuda(source, options);
}

function getVersion() { return version; }

module.exports = { transpileCuda, transpileFile, getCompilerVersion, getVersion, CudaWasmError, MAX_INPUT_BYTES };
module.exports.default = module.exports;
