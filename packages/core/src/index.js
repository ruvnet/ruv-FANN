import { loadWasm } from 'ruv-fann-wasm';
import { LIMITS, integer, options, finite, array, inputs, modelDocument, storageBudget } from './validation.js';
export { LIMITS } from './validation.js';
const runtimes = new WeakMap();

/** Load one of two isolated actual WASM backends. No JS numerical fallback. */
export async function loadRuntime(configuration = {}) {
  const raw = await loadWasm(configuration);
  if (!runtimes.has(raw)) runtimes.set(raw, Object.freeze({
    backend: raw.backend,
    prepare(model, configuration = {}) { return prepare(raw, model, configuration); },
    train(model, configuration) { return train(raw, model, configuration); }
  }));
  return runtimes.get(raw);
}

function prepare(raw, model, configuration) {
  options(configuration, ['maxRows'], 'prepare options');
  const maxRows = integer(configuration.maxRows === undefined ? 128 : configuration.maxRows, 1, LIMITS.maxRows, 'maxRows');
  const document = modelDocument(model);
  storageBudget(document, maxRows, false);
  const { json } = document;
  const prepared = raw.bindings.WasmPreparedNetwork.from_dense_layers_json(json);
  let batch;
  try { batch = prepared.create_batch(maxRows); } finally { prepared.free(); }
  const inputSize = batch.input_size, outputSize = batch.output_size, estimatedBytes = batch.estimated_bytes;
  let disposed = false;
  function rowsFor(values, rows) {
    if (disposed) throw new Error('prepared model is disposed');
    return inputs(values, inputSize, rows, maxRows, 'inputs', raw.memory);
  }
  function predictInto(values, output, rows) {
    const count = rowsFor(values, rows);
    if (array(output, 'outputs', raw.memory) !== count * outputSize)
      throw new RangeError('outputs length does not match rows');
    batch.run_into(values, output, count);
  }
  return Object.freeze({ inputSize, outputSize, maxRows, estimatedBytes,
    predict(values, rows) {
      const count = rowsFor(values, rows);
      const output = new Float64Array(count * outputSize);
      batch.run_into(values, output, count);
      return output;
    },
    predictInto,
    dispose() { if (!disposed) { disposed = true; batch.free(); } }
  });
}

function train(raw, model, configuration) {
  options(configuration, ['inputs', 'targets', 'rows', 'epochs', 'learningRate', 'optimizer'], 'training options');
  const rows = integer(configuration.rows, 1, LIMITS.maxRows, 'rows');
  const epochs = integer(configuration.epochs, 1, LIMITS.maxEpochs, 'epochs');
  const learningRate = finite(configuration.learningRate, 'learningRate');
  if (learningRate <= 0) throw new RangeError('learningRate must be positive');
  const optimizer = configuration.optimizer === undefined ? 'sgd' : configuration.optimizer;
  if (optimizer !== 'sgd' && optimizer !== 'adamw') throw new TypeError('optimizer must be sgd or adamw');
  const document = modelDocument(model);
  storageBudget(document, rows, true);
  // Bound total synchronous compute before native training workspace allocation.
  const backwardWork = rows * (32 * document.parameterCount + 8 * (document.inputSize + document.outputSize) + 1);
  const updateWork = 64 * document.parameterCount;
  if (backwardWork * (epochs + 1) + updateWork * epochs > LIMITS.maxTrainingWork)
    throw new RangeError('training exceeds total work budget');
  const input = configuration.inputs, target = configuration.targets;
  inputs(input, document.inputSize, rows, LIMITS.maxRows, 'inputs', raw.memory);
  inputs(target, document.outputSize, rows, LIMITS.maxRows, 'targets', raw.memory);
  const network = raw.bindings.WasmTrainableNetwork.from_dense_layers_json(document.json, rows);
  try {
    if (optimizer === 'adamw') network.configure_adamw(learningRate, 0.9, 0.999, 1e-8, 0);
    const initialLoss = finite(network.mse_backward(input, target, rows), 'initial loss');
    let finalLoss = initialLoss;
    for (let epoch = 0; epoch < epochs; epoch++) {
      if (optimizer === 'sgd') network.sgd_step(learningRate);
      else network.adamw_step();
      finalLoss = finite(network.mse_backward(input, target, rows), 'final loss');
    }
    return { model: JSON.parse(network.to_dense_layers_json()), initialLoss, finalLoss,
      epochs, parameterCount: network.parameter_count };
  } finally { network.free(); }
}
