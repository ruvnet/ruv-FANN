/** Explicit Node asset loader. Browser callers supply bundled WASM bytes. */
import { constants } from 'node:fs';
import { open } from 'node:fs/promises';

export async function ruvectorWasmBytes() {
  const url = new URL('./ruvector_wasm_bg.wasm', import.meta.resolve('@ruvector/wasm'));
  const handle = await open(url, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
  try {
    const stat = await handle.stat();
    const limit = 1024 * 1024;
    if (!stat.isFile() || stat.size < 8 || stat.size > limit) throw new Error('Invalid bounded RuVector WASM asset');
    const bytes = new Uint8Array(stat.size + 1);
    let used = 0;
    while (used < bytes.length) {
      const read = await handle.read(bytes, used, bytes.length - used, null);
      if (read.bytesRead === 0) break;
      used += read.bytesRead;
    }
    if (used !== stat.size || (await handle.stat()).size !== stat.size) throw new Error('RuVector WASM asset changed while reading');
    if (bytes[0] !== 0 || bytes[1] !== 97 || bytes[2] !== 115 || bytes[3] !== 109) throw new Error('Invalid WASM header');
    return bytes.slice(0, used);
  } finally { await handle.close(); }
}
