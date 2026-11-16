import { build } from 'esbuild';
import { mkdir, readFile, writeFile, copyFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';

const wasmRoot = resolve(dirname(fileURLToPath(import.meta.resolve('ruv-fann-wasm'))), '..');
await mkdir('dist/runtime', { recursive: true });
await copyFile(join(wasmRoot, 'dist/ruv_swarm_wasm_bg.wasm'), 'dist/core.wasm');
await copyFile(join(wasmRoot, 'dist/ruv_swarm_simd.wasm'), 'dist/simd.wasm');
await copyFile('dist/core.wasm', 'dist/runtime/core.wasm');
await copyFile('dist/simd.wasm', 'dist/runtime/simd.wasm');

function runtimePlugin(edge) {
  return { name: 'static-wasm-runtime', setup(builder) {
    builder.onResolve({ filter: /^ruv-fann-wasm$/ }, () => ({ path: 'runtime', namespace: 'playground-runtime' }));
    builder.onLoad({ filter: /.*/, namespace: 'playground-runtime' }, async () => {
      const source = await readFile(join(wasmRoot, 'src/index.js'), 'utf8');
      const support = source.slice(source.indexOf('export function supportsSIMD'), source.indexOf('export function validateRuntimeOptions'));
      return { resolveDir: wasmRoot, contents: `
        import * as core from ${JSON.stringify(join(wasmRoot, 'dist/core.js'))};
        import * as simd from ${JSON.stringify(join(wasmRoot, 'dist/simd.js'))};
        ${edge ? "import coreModule from './core.wasm'; import simdModule from './simd.wasm';" : ''}
        ${support}
        const flights = new Map();
        export function validateRuntimeOptions(options = {}) {
          const v = options.simd ?? 'auto';
          if (![true,false,'auto'].includes(v)) throw new TypeError('Invalid SIMD option');
          return v;
        }
        export async function loadWasm(options = {}) {
          const v = validateRuntimeOptions(options), accelerated = v === 'auto' ? ${edge ? 'true' : 'supportsSIMD()'} : v;
          const key = accelerated ? 'simd' : 'core';
          if (!flights.has(key)) {
            const promise = (async () => {
              const bindings = accelerated ? simd : core;
              const wasm = ${edge ? 'bindings.initSync({module: accelerated ? simdModule : coreModule})' : "await bindings.default({module_or_path: fetch('/runtime/' + key + '.wasm')})"};
              if (!(wasm.memory instanceof WebAssembly.Memory)) throw new Error('Missing WASM memory');
              return Object.freeze({backend:accelerated?'wasm-simd':'wasm',bindings,memory:wasm.memory});
            })();
            flights.set(key,promise); promise.catch(() => { if (flights.get(key)===promise) flights.delete(key); });
          }
          return flights.get(key);
        }` };
    });
    if (edge) {
      builder.onResolve({ filter: /^\.\/(core|simd)\.wasm$/ }, args => ({ path: args.path, external: true }));
      builder.onResolve({ filter: /dist\/(site-assets\.js|.*-base64\.txt)$/ }, args => ({ path: './' + args.path.split('/').at(-1), external: true }));
    }
  } };
}
const common = { bundle: true, format: 'esm', platform: 'browser', target: 'es2022', minify: true,
  legalComments: 'none', define: { process: 'undefined' } };
await build({ ...common, entryPoints: ['src/app.js'], outfile: 'dist/app.js', plugins: [runtimePlugin(false)] });
await copyFile('src/index.html', 'dist/index.html');
await copyFile('src/style.css', 'dist/style.css');
const assets = {};
for (const [name, type] of [['index.html', 'text/html; charset=utf-8'], ['app.js', 'application/javascript; charset=utf-8'], ['style.css', 'text/css; charset=utf-8']]) {
  assets['/' + name] = { type, body: await readFile('dist/' + name, 'utf8') };
}
await writeFile('dist/site-assets.js', `export default ${JSON.stringify(assets)};\n`);
for (const key of ['core', 'simd']) await writeFile(`dist/${key}-base64.txt`, (await readFile(`dist/${key}.wasm`)).toString('base64'));
await build({ ...common, entryPoints: ['src/worker.js'], outfile: 'dist/worker.js', plugins: [runtimePlugin(true)] });
const modules = ['worker.js', 'site-assets.js', 'core-base64.txt', 'simd-base64.txt', 'core.wasm', 'simd.wasm'];
const manifest = {};
for (const name of modules) {
  const bytes = await readFile('dist/' + name);
  manifest[name] = { bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex'),
    contentType: name.endsWith('.wasm') ? 'application/wasm' : name.endsWith('.txt') ? 'text/plain' : 'application/javascript+module' };
}
await writeFile('dist/manifest.json', JSON.stringify(manifest, null, 2));
console.log(JSON.stringify(manifest, null, 2));
