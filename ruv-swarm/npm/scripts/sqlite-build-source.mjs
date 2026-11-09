/** Bounded download and selective extraction; no shell/tar extraction or URLs from callers. */
import { createRequire } from 'node:module';
import { gunzipSync } from 'node:zlib';
import path from 'node:path';
import tls from 'node:tls';
import {
  NODE_GYP_VERSION, SOURCE_FILES, sha256, readBoundedFile,
} from './sqlite-build-contract.mjs';

export const MAX_ARCHIVE_BYTES = 32 * 1024 * 1024;
export const MAX_TAR_BYTES = 96 * 1024 * 1024;
const require = createRequire(import.meta.url);

export function nodeGypPackage() {
  const manifest = require.resolve('node-gyp/package.json');
  const pkg = JSON.parse(readBoundedFile(manifest, 65536));
  if (pkg.version !== NODE_GYP_VERSION) {
    throw new Error(`Install the pinned node-gyp ${ NODE_GYP_VERSION}`);
  }
  return { manifest, directory: path.dirname(manifest) };
}

export async function downloadPinnedSource(source) {
  const gyp = nodeGypPackage(), gypRequire = createRequire(gyp.manifest);
  const { fetch, EnvHttpProxyAgent } = gypRequire('undici');
  const options = {};
  const caFile = process.env.npm_config_cafile || process.env.NODE_EXTRA_CA_CERTS || process.env.SSL_CERT_FILE;
  const trusted = { rejectUnauthorized: true };
  if (caFile) {
    const pem = readBoundedFile(caFile, 1024 * 1024).toString('utf8');
    const certificates = pem.match(/-----BEGIN CERTIFICATE-----[\s\S]*?-----END CERTIFICATE-----/g);
    if (!certificates?.length) {
      throw new Error('Configured CA file contains no certificates');
    }
    trusted.ca = [...tls.rootCertificates, ...certificates];
  }
  options.connect = trusted;
  options.requestTls = trusted;
  options.proxyTls = trusted;
  // This reads only proxy/NO_PROXY settings. No npm registry credentials or
  // authorization headers are copied to the public SQLite source request.
  const dispatcher = new EnvHttpProxyAgent(options);
  try {
    const response = await fetch(source.archive.url, {
      dispatcher, redirect: 'error', signal: globalThis.AbortSignal.timeout(60000),
      headers: { Accept: 'application/octet-stream' },
    });
    if (response.status !== 200 || !response.body) {
      throw new Error('Pinned SQLite source request failed');
    }
    const length = response.headers.get('content-length');
    if (length !== null && (!/^\d+$/.test(length) || Number(length) > MAX_ARCHIVE_BYTES)) {
      await response.body.cancel(); throw new Error('SQLite source archive exceeds its byte limit');
    }
    const chunks = [];
    let bytes = 0;
    for await (const chunk of response.body) {
      bytes += chunk.length;
      if (bytes > MAX_ARCHIVE_BYTES) {
        throw new Error('SQLite source archive exceeds its byte limit');
      }
      chunks.push(chunk);
    }
    return checkedArchive(Buffer.concat(chunks, bytes), source);
  } finally {
    await dispatcher.close();
  }
}

export function checkedArchive(bytes, source) {
  if (bytes.length === 0 || bytes.length > MAX_ARCHIVE_BYTES || bytes.length !== source.archive.bytes ||
    sha256(bytes) !== source.archive.sha256) {
    throw new Error('Pinned SQLite archive size/checksum mismatch');
  }
  return bytes;
}

function text(header, start, length) {
  const field = header.subarray(start, start + length), end = field.indexOf(0);
  return field.subarray(0, end < 0 ? field.length : end).toString('utf8');
}
function octal(header, start, length) {
  const value = text(header, start, length).trim();
  if (!/^[0-7]+$/.test(value)) {
    throw new Error('Unsupported SQLite tar numeric field');
  }
  const number = Number.parseInt(value, 8);
  if (!Number.isSafeInteger(number)) {
    throw new Error('SQLite tar size overflow');
  }
  return number;
}

/** Return only the three pinned byte strings; never use archive paths as filesystem paths. */
export function selectedTarFiles(tar, source) {
  if (tar.length > MAX_TAR_BYTES) {
    throw new Error('SQLite tar exceeds its byte limit');
  }
  const found = new Map();
  let offset = 0, entries = 0, ended = false;
  while (offset + 512 <= tar.length) {
    const header = tar.subarray(offset, offset + 512);
    if (header.every(byte => byte === 0)) {
      ended = true; break;
    }
    if (++entries > 4096) {
      throw new Error('SQLite tar entry limit exceeded');
    }
    let checksum = 0;
    for (let i = 0; i < 512; i++) {
      checksum += i >= 148 && i < 156 ? 32 : header[i];
    }
    if (checksum !== octal(header, 148, 8)) {
      throw new Error('SQLite tar header checksum mismatch');
    }
    const prefix = text(header, 345, 155);
    const name = (prefix ? `${prefix }/` : '') + text(header, 0, 100);
    if (name.startsWith('/') || name.includes('\\') || name.split('/').some(part => part === '..')) {
      throw new Error('Unsafe SQLite tar path');
    }
    const size = octal(header, 124, 12), type = header[156];
    const start = offset + 512, next = start + Math.ceil(size / 512) * 512;
    if (next > tar.length) {
      throw new Error('Truncated SQLite tar entry');
    }
    if (![0, 48, 53].includes(type)) {
      throw new Error('SQLite source archive contains a nonregular entry');
    }
    for (const file of SOURCE_FILES) {
      if (name !== source.archive.prefix + file) {
        continue;
      }
      if (type !== 0 && type !== 48) {
        throw new Error('SQLite amalgamation must be a regular file');
      }
      if (found.has(file)) {
        throw new Error('Duplicate SQLite amalgamation entry');
      }
      const bytes = tar.subarray(start, start + size), pin = source.files[file];
      if (size !== pin.bytes || sha256(bytes) !== pin.sha256) {
        throw new Error(`SQLite source checksum mismatch: ${ file}`);
      }
      found.set(file, bytes);
    }
    offset = next;
  }
  if (!ended || found.size !== SOURCE_FILES.length) {
    throw new Error('Incomplete SQLite amalgamation archive');
  }
  return found;
}

export async function sourceFiles(source, filename) {
  const archive = filename
    ? checkedArchive(readBoundedFile(filename, MAX_ARCHIVE_BYTES), source)
    : await downloadPinnedSource(source);
  return selectedTarFiles(gunzipSync(archive, { maxOutputLength: MAX_TAR_BYTES }), source);
}
