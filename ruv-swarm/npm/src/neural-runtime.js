export function numbers(value, name, maximum = 1000000) {
  if ((!Array.isArray(value) && !ArrayBuffer.isView(value)) || value.length === 0 || value.length > maximum ||
      !Array.from(value).every(item => typeof item === 'number' && Number.isFinite(item))) {
    throw new TypeError(`${name} must contain 1..${maximum} finite numbers`);
  }
  return Array.from(value);
}

export function infer(runtime, params) {
  const layers = numbers(params.layers, 'layers', 64);
  const input = numbers(params.input, 'input');
  const weights = numbers(params.weights, 'weights');
  const activation = params.activation || 'Linear';
  const network = runtime.createNeuralNetwork(layers, activation);
  try {
    network.set_weights(Float64Array.from(weights));
    return { kind: 'dense-inference', activation, output: Array.from(network.run(Float64Array.from(input))) };
  } finally {
    network.free();
  }
}

export function forecast(runtime, params) {
  const input = numbers(params.input, 'input');
  const model = runtime.createForecastingModel(params.model || 'linear');
  try {
    return { kind: 'statistical-forecast', model: params.model || 'linear', output: Array.from(model.predict(Float64Array.from(input))) };
  } finally {
    model.free();
  }
}

export function denseBenchmark(runtime, params = {}) {
  const iterations = params.iterations ?? 1000;
  if (!Number.isInteger(iterations) || iterations < 1 || iterations > 10000) {
    throw new RangeError('iterations must be an integer between 1 and 10000');
  }
  if (params.type && !['wasm', 'neural', 'dense-inference'].includes(params.type)) {
    throw new RangeError('Only the measured dense-inference benchmark is supported');
  }
  const network = runtime.createNeuralNetwork([2, 1], 'Linear');
  try {
    network.set_weights(Float64Array.of(2, -1));
    const input = Float64Array.of(3, 4);
    let checksum = 0;
    const started = process.hrtime.bigint();
    for (let index = 0; index < iterations; index++) {
      checksum += network.run(input)[0];
    }
    const elapsedNs = Number(process.hrtime.bigint() - started);
    if (checksum !== iterations * 2) {
      throw new Error('Inference verification failed during benchmark');
    }
    return { kind: 'dense-inference', iterations, elapsed_ms: elapsedNs / 1000000,
      operations_per_second: elapsedNs ? iterations * 1000000000 / elapsedNs : null, checksum,
      scope: 'Local WASM inference including JavaScript call overhead; no LLM or distributed workload' };
  } finally {
    network.free();
  }
}
