// Real generated WASM, including public JS boundary and numeric/JSON copies.
// node examples/benchmark-throughput.mjs [--baseline /path/to/previous/wasm] [--only-calibration]
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import os from 'node:os';
import {inferenceSuite, calibrationSuite} from './throughput-suite.mjs';

const args = process.argv.slice(2);
const baselineIndex = args.indexOf('--baseline');
const baseline = baselineIndex >= 0 ? path.resolve(args[baselineIndex + 1]) : null;
const directoryIndex = args.indexOf('--wasm-dir');
const current = directoryIndex >= 0 ? path.resolve(args[directoryIndex + 1]) : fileURLToPath(new URL('../wasm', import.meta.url));
const onlyCalibration = args.includes('--only-calibration');
const baselineOnly = args.includes('--baseline-only');
if (baselineOnly && !baseline) throw new Error('--baseline-only requires --baseline');
const references = new Map();
const results = [];
let exactComparisons = 0;
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
for (const [revision, directory] of [...(baseline ? [['before', baseline]] : []), ...(baselineOnly ? [] : [['current', current]])]) {
  for (const variant of ['core', 'simd']) {
    const binary = path.join(directory, variant === 'core' ? 'ruv_swarm_wasm_bg.wasm' : 'ruv_swarm_simd.wasm');
    const moduleUrl = pathToFileURL(path.join(directory, 'ruv_swarm_wasm.js'));
    moduleUrl.searchParams.set('benchmark', `${revision}-${variant}`);
    const bindings = await import(moduleUrl.href);
    await bindings.default({module_or_path: fs.readFileSync(binary)});
    const features = JSON.parse(bindings.get_features());
    if (features.simd_support !== (variant === 'simd')) throw new Error('Incorrect binary variant');
    const calibration = calibrationSuite(bindings, {validate(key, groups, output) {
      const id = `${variant}/${key}`;
      if (revision === 'before') references.set(id, [groups, output]);
      else if (references.has(id)) {
        const expected = references.get(id);
        if (expected[0] !== groups || expected[1] !== output) throw new Error(`Changed calibration JSON bytes: ${id}`);
        exactComparisons++;
        return true;
      }
      return false;
    }});
    results.push({revision, variant, sha256: hash(binary), features,
      inference: onlyCalibration ? [] : inferenceSuite(bindings), calibration});
    process.stderr.write(`Completed ${revision}/${variant}\n`);
  }
}
console.log(JSON.stringify({schema_version: 1, date_utc: new Date().toISOString(), runtime: 'Node',
  node: process.version, v8: process.versions.v8, platform: process.platform, cpu: os.cpus()[0]?.model,
  cpu_quota: fs.existsSync('/sys/fs/cgroup/cpu.max') ? fs.readFileSync('/sys/fs/cgroup/cpu.max', 'utf8').trim() : null,
  precision: 'f64', clock: 'performance.now(), ns units derived from ms; no claim of ns resolution',
  method: 'Warm each exact API, adapt bounded iterations toward 3 ms, rotate call order, retain every paired sample. Fixture generation and output verification excluded from timings.',
  inference_scope: 'Whole row-major batches, including JS guards, copies and legacy result allocations; report ns/row plus batch latency.',
  calibration_scope: 'Raw public JSON fit+free and apply+returned string. Parsing/emission included; fixture construction and JS output parsing excluded.',
  setup_scope: 'One observed batch constructor per case, outside inference timings; no cold-start distribution.',
  exact_before_after_calibration_cases: exactComparisons,
  results}, null, 2));
