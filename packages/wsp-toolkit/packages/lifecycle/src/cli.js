#!/usr/bin/env node
import { constants } from 'node:fs';
import { open } from 'node:fs/promises';
import { runRegressionGate, LIFECYCLE_LIMITS } from './index.js';
try {
  if (process.argv.length !== 3 || process.argv[2].startsWith('-')) throw new Error('usage: wsp-gate <bundle.json>');
  const file = await open(process.argv[2], constants.O_RDONLY | constants.O_NONBLOCK);
  let bytes;
  try {
    if (!(await file.stat()).isFile()) throw new Error('bundle must be a regular file');
    // Bounded read, including when the file grows after stat.
    const buffer = Buffer.alloc(LIFECYCLE_LIMITS.maxBytes + 1);
    let offset = 0;
    while (offset < buffer.length) {
      const { bytesRead } = await file.read(buffer, offset, buffer.length - offset, null);
      if (!bytesRead) break;
      offset += bytesRead;
    }
    if (offset > LIFECYCLE_LIMITS.maxBytes) throw new RangeError('bundle exceeds byte limit');
    bytes = buffer.subarray(0, offset);
  } finally { await file.close(); }
  const result = await runRegressionGate(JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes)));
  process.stdout.write(`${JSON.stringify(result)}\n`);
  process.exitCode = result.accepted ? 0 : 1;
} catch (error) {
  process.stderr.write(`wsp-gate: ${error.message}\n`);
  process.exitCode = 2;
}
