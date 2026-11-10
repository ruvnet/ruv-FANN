// Standalone functions also executed inside a real browser and installed consumers.
export async function runWorkbenchProbe() {
  const wsp = await import('ruv-fann-wsp');
  const { loadRuntime } = await import('ruv-fann');
  const check = (value, message) => { if (!value) throw new Error(message); };
  const near = (a, b) => Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b));
  const model = { schema: 'ruv-fann.dense-layers/1', layers: [
    { inputs: 1, outputs: 2, weights: [1, -1], bias: [0.5, -0.5], activation: 'Sigmoid', steepness: 0.7 },
    { inputs: 2, outputs: 1, weights: [2, 3], bias: [1], activation: 'Linear', steepness: 1.4 }
  ] };
  const batch = { inputs: [-1, 0, 1, 2], rows: 4 };
  const info = await wsp.inspectWeightModel(model);
  check(info.parameterCount === 7 && info.layers.length === 2, 'inspection dimensions');
  const trace = await wsp.traceWeightModel(model, batch, { simd: true });
  check(trace.backend === 'wasm-simd' && trace.baseHash === info.baseHash, 'trace backend and hash');
  check(trace.layers[0].preactivations.every((value, index) => near(value,
    index % 2 === 0 ? batch.inputs[Math.floor(index / 2)] + 0.5 : -batch.inputs[Math.floor(index / 2)] - 0.5)), 'raw affine trace ignores activation steepness');
  const runtime = await loadRuntime({ simd: true });
  const prepared = runtime.prepare(model, { maxRows: 4 });
  try {
    const output = prepared.predict(Float64Array.from(batch.inputs), batch.rows);
    check(output.every((value, index) => near(value, trace.layers[1].activations[index])), 'trace agrees with complete WASM inference');
  } finally { prepared.dispose(); }
  const ablation = await wsp.createNeuronAblationPatch(model, 0, 0);
  check(ablation.baseHash === info.baseHash && ablation.edits.length === 1 && ablation.edits[0].layer === 1 &&
    ablation.edits[0].index === 0 && ablation.edits[0].value === 0, 'ablation zeros outgoing connection');
  const changed = await wsp.applyWeightPatch(model, ablation);
  check(changed.layers[1].weights[0] === 0 && model.layers[1].weights[0] === 2, 'ablation ownership');
  const linear = { schema: 'ruv-fann.dense-layers/1', layers: [
    { inputs: 1, outputs: 1, weights: [0], bias: [0], activation: 'Linear', steepness: 1 }
  ] };
  const sensitivity = await wsp.analyzeWeightSensitivity(linear, { inputs: [0, 1, 2], targets: [1, 3, 5], rows: 3 }, [
    { layer: 0, parameter: 'weights', index: 0 }, { layer: 0, parameter: 'bias', index: 0 }
  ], { simd: false, epsilon: 0.0001 });
  check(sensitivity.method === 'central-finite-difference' && sensitivity.backend === 'wasm', 'honest numerical method');
  check(near(sensitivity.entries[0].derivative, -26 / 3) && near(sensitivity.entries[1].derivative, -6), 'independent analytic MSE sensitivity oracle');
  return { traceBackend: trace.backend, sensitivityMethod: sensitivity.method, ablationHashBound: true, analyticDerivative: true };
}

export async function runWorkbenchSecurityProbe() {
  const wsp = await import('ruv-fann-wsp');
  const check = (value, message) => { if (!value) throw new Error(message); };
  let checks = 0, getterCalls = 0;
  const reject = async (fn, name) => {
    let failed = false;
    try { await fn(); } catch { failed = true; }
    check(failed, `security rejection missing: ${name}`); checks++;
  };
  const make = () => ({ schema: 'ruv-fann.dense-layers/1', layers: [
    { inputs: 1, outputs: 1, weights: [1], bias: [0], activation: 'Sigmoid', steepness: 1 },
    { inputs: 1, outputs: 1, weights: [2], bias: [0], activation: 'Linear', steepness: 1 }
  ] });
  const batch = () => ({ inputs: [0, 1], rows: 2 });
  const dataset = () => ({ inputs: [0, 1], targets: [0, 1], rows: 2 });
  const location = { layer: 0, parameter: 'weights', index: 0 };
  await reject(() => wsp.traceWeightModel(make(), { ...batch(), accepted: true }), 'unknown batch field');
  await reject(() => wsp.traceWeightModel(make(), { inputs: [Infinity], rows: 1 }), 'nonfinite input');
  await reject(() => wsp.traceWeightModel(make(), { inputs: new Array(2), rows: 2 }), 'sparse array');
  const getter = { rows: 2, get inputs() { getterCalls++; return [0, 1]; } };
  await reject(() => wsp.traceWeightModel(make(), getter), 'batch accessor');
  const hostile = [0, 1]; Object.defineProperty(hostile, 0, { get() { getterCalls++; return 0; } });
  await reject(() => wsp.traceWeightModel(make(), { inputs: hostile, rows: 2 }), 'array accessor');
  const symbol = [0, 1]; symbol[Symbol.iterator] = function* () { getterCalls++; yield 0; yield 1; };
  await reject(() => wsp.traceWeightModel(make(), { inputs: symbol, rows: 2 }), 'custom iterator');
  const extra = [0, 1]; extra.accepted = true;
  await reject(() => wsp.traceWeightModel(make(), { inputs: extra, rows: 2 }), 'extra array field');
  await reject(() => wsp.traceWeightModel(make(), batch(), { simd: false, maxWork: Infinity }), 'caller cannot raise work limit');
  await reject(() => wsp.analyzeWeightSensitivity(make(), dataset(), [location, location]), 'duplicate probe');
  await reject(() => wsp.analyzeWeightSensitivity(make(), dataset(), [location], { epsilon: 0 }), 'zero epsilon');
  await reject(() => wsp.analyzeWeightSensitivity(make(), dataset(), [location], { epsilon: Infinity }), 'infinite epsilon');
  await reject(() => wsp.analyzeWeightSensitivity(make(), dataset(), Array.from({ length: 17 }, () => location)), 'probe amplification');
  await reject(() => wsp.createNeuronAblationPatch(make(), 1, 0), 'output layer ablation');
  await reject(() => wsp.createNeuronAblationPatch(make(), 0, 1), 'neuron out of range');
  const zero = make(); zero.layers[1].weights[0] = 0;
  await reject(() => wsp.createNeuronAblationPatch(zero, 0, 0), 'no-op ablation');
  const large = { schema: 'ruv-fann.dense-layers/1', layers: [
    { inputs: 1, outputs: 64, weights: Array(64).fill(1), bias: Array(64).fill(0), activation: 'Linear', steepness: 1 },
    { inputs: 64, outputs: 64, weights: Array(4096).fill(1), bias: Array(64).fill(0), activation: 'Linear', steepness: 1 }
  ] };
  await reject(() => wsp.traceWeightModel(large, { inputs: Array(1024).fill(0), rows: 1024 }), 'aggregate trace storage');
  await reject(() => wsp.analyzeWeightSensitivity(large, { inputs: Array(1024).fill(0), targets: Array(65536).fill(0), rows: 1024 }, [location]), 'aggregate probe work');
  check(getterCalls === 0, 'caller accessors/iterators executed'); checks++;
  const original = make(), values = batch();
  const pending = wsp.traceWeightModel(original, values, { simd: false });
  original.layers[0].weights[0] = 1000; values.inputs[0] = 1000;
  const result = await pending;
  check(result.layers[0].preactivations[0] === 0, 'pre-await snapshot isolation'); checks++;
  return { checks, failed: 0, status: 'passed', getterCalls };
}
