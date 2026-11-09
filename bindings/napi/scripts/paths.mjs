/** Keep build and proof outputs outside the checkout, including symlink aliases. */
import { lstat } from 'node:fs/promises';
import { dirname, isAbsolute, relative, resolve, sep } from 'node:path';

function contains(parent, child) {
  const part = relative(parent, child);
  return part === '' || (!isAbsolute(part) && part !== '..' && !part.startsWith('..' + sep));
}

export async function externalPath(value, repository, { existingDirectory = false } = {}) {
  const output = resolve(value), root = resolve(repository);
  if (contains(root, output) || contains(output, root))
    throw new Error('Output must be external to the checkout and cannot contain it');
  let depth = 0;
  for (let current = output;; current = dirname(current)) {
    if (++depth > 256) throw new Error('Output ancestry exceeds its path budget');
    let info;
    try { info = await lstat(current); } catch (error) { if (error.code !== 'ENOENT') throw error; }
    if (info) {
      if (info.isSymbolicLink()) throw new Error('Output rejects symlink ancestry');
      if (!info.isDirectory()) throw new Error('Output ancestry must contain ordinary directories');
      if (current === output && !existingDirectory) throw new Error('Output already exists');
    }
    if (dirname(current) === current) break;
  }
  return output;
}
