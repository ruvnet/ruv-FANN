/** Reusable real-native proof; linear finite differences and an optional actual Candle oracle file. */
import assert from 'node:assert/strict';
import { readFile, lstat } from 'node:fs/promises';
import { createHash } from 'node:crypto';

export const trainingJson = (parameters = [0.2, -0.3, 0.1]) => JSON.stringify({ schema: 'ruv-fann.dense-layers/1',
  layers: [{ inputs: 2, outputs: 1, weights: parameters.slice(0, 2), bias: [parameters[2]], activation: 'Linear', steepness: 1 }] });
const parameters = model => JSON.parse(model.to_dense_layers_json()).layers.flatMap(layer => [...layer.weights, ...(layer.bias ?? [])]);
export function close(a, b, tolerance = 1e-10) {
  assert.ok(typeof tolerance === 'number' && Number.isFinite(tolerance) && tolerance >= 0 && tolerance <= 1e-6, 'Comparison tolerance must be finite and bounded');
  assert.equal(a.length, b.length);
  for (let index = 0; index < a.length; index++) {
    assert.ok(typeof a[index] === 'number' && Number.isFinite(a[index]) && typeof b[index] === 'number' && Number.isFinite(b[index]), 'Actual and reference values must be finite numbers');
    assert.ok(Math.abs(a[index] - b[index]) <= tolerance * Math.max(1, Math.abs(a[index]), Math.abs(b[index])), `${index}: ${a[index]} vs ${b[index]}`);
  }
}

export async function trainingProof(NativeTrainableNetwork, fixturePath = process.env.RUV_FANN_CANDLE_FIXTURE) {
  const input = new Float64Array([2, -1, 1, 3]), targets = new Float64Array([0.7, -0.2]);
  const train = NativeTrainableNetwork.from_dense_layers_json(trainingJson(), 2);
  let firstLoss, lastLoss;
  try {
    firstLoss = train.mse_backward(input, targets, 2);
    assert.ok(Math.abs(firstLoss - 0.085) < 1e-12);
    close([...train.parameter_gradients()], [-0.2, -1.3, -0.3]);
    close([...train.input_gradients()], [0.02, -0.03, -0.08, 0.12]);
    close([...train.predictions()], [0.8, -0.6]);
    const analytic = [...train.parameter_gradients()], base = parameters(train), epsilon = 1e-6;
    for (let index = 0; index < base.length; index++) {
      const losses = [];
      for (const sign of [-1, 1]) {
        const values = base.slice(); values[index] += sign * epsilon;
        const varied = NativeTrainableNetwork.from_dense_layers_json(trainingJson(values), 2);
        try { losses.push(varied.mse_backward(input, targets, 2)); } finally { varied.free(); }
      }
      close([(losses[1] - losses[0]) / (2 * epsilon)], [analytic[index]], 1e-7);
    }
    const unchangedBeforeOverflow = train.to_dense_layers_json();
    assert.throws(() => train.sgd_step(Number.MAX_VALUE));
    assert.equal(train.to_dense_layers_json(), unchangedBeforeOverflow);
    const frozen = train.prepare(), frozenOutput = [...frozen.run(new Float64Array([2, -1]))];
    try {
      train.sgd_step(0.02);
      assert.throws(() => train.sgd_step(0.02)); // A successful step consumes the bound gradient revision.
      const unchanged = train.to_dense_layers_json();
      assert.throws(() => train.sgd_step(NaN)); assert.equal(train.to_dense_layers_json(), unchanged);
      for (let iteration = 0; iteration < 80; iteration++) {
        train.mse_backward(input, targets, 2); train.sgd_step(0.02);
      }
      lastLoss = train.mse_backward(input, targets, 2);
      assert.ok(lastLoss < firstLoss * 1e-3, `${lastLoss} did not improve ${firstLoss}`);
      close([...frozen.run(new Float64Array([2, -1]))], frozenOutput, 0);
    } finally { frozen.free(); }
  } finally { train.free(); }
  train.free(); assert.throws(() => train.parameter_gradients(), /freed/);

  const adam = NativeTrainableNetwork.from_dense_layers_json(trainingJson(), 2);
  let adamFirst, adamLast;
  try {
    adam.configure_adamw(0.02, 0.9, 0.99, 1e-8, 0.01);
    adamFirst = adam.mse_backward(input, targets, 2);
    const old = adam.to_dense_layers_json();
    for (const args of [[NaN, 0.9, 0.99, 1e-8, 0], [0.02, 1, 0.99, 1e-8, 0], [0.02, 0.9, 0.99, 0, 0]])
      assert.throws(() => adam.configure_adamw(...args));
    assert.equal(adam.to_dense_layers_json(), old);
    for (let iteration = 0; iteration < 100; iteration++) { adam.mse_backward(input, targets, 2); adam.adamw_step(); }
    assert.throws(() => adam.adamw_step());
    adamLast = adam.mse_backward(input, targets, 2);
    assert.ok(adamLast < adamFirst * 0.01, `${adamLast} did not improve ${adamFirst}`);
  } finally { adam.free(); }

  let candle = { verified: false, reason: 'RUV_FANN_CANDLE_FIXTURE was not supplied' };
  if (fixturePath) {
    const info = await lstat(fixturePath);
    assert.ok(info.isFile() && !info.isSymbolicLink() && info.size <= 4 * 1024 * 1024);
    const bytes = await readFile(fixturePath), fixture = JSON.parse(bytes);
    assert.match(fixture.oracle, /candle/i);
    assert.ok(fixture.dense_json && Array.isArray(fixture.parameter_gradients) && Array.isArray(fixture.input_gradients));
    assert.equal(fixture.sgd_steps?.length, 3); assert.equal(fixture.adamw_steps?.length, 3);
    for (const value of [fixture.absolute_tolerance, fixture.relative_tolerance])
      assert.ok(typeof value === 'number' && Number.isFinite(value) && value > 0 && value <= 1e-9, 'Candle tolerance must be finite, positive and at most 1e-9');
    const tolerance = Math.max(fixture.absolute_tolerance, fixture.relative_tolerance);
    const input = new Float64Array(fixture.input), target = new Float64Array(fixture.training_target);
    const model = NativeTrainableNetwork.from_dense_layers_json(fixture.dense_json, 1);
    let parameterGradientError = 0, inputGradientError = 0, sgdError = 0, adamwError = 0;
    try {
      const loss = model.mse_backward(input, target, 1);
      close([loss], [fixture.training_loss], tolerance);
      close([...model.parameter_gradients()], fixture.parameter_gradients, tolerance);
      close([...model.input_gradients()], fixture.input_gradients, tolerance);
      parameterGradientError = Math.max(...model.parameter_gradients().map((value, index) => Math.abs(value - fixture.parameter_gradients[index])));
      inputGradientError = Math.max(...model.input_gradients().map((value, index) => Math.abs(value - fixture.input_gradients[index])));
      model.sgd_step(fixture.sgd_learning_rate);
      close(parameters(model), fixture.sgd_expected_parameters, tolerance);
      sgdError = Math.max(...parameters(model).map((value, index) => Math.abs(value - fixture.sgd_expected_parameters[index])));
    } finally { model.free(); }
    const opt = NativeTrainableNetwork.from_dense_layers_json(fixture.dense_json, 1);
    try {
      const config = fixture.adamw_config;
      opt.configure_adamw(config.learning_rate, config.beta1, config.beta2, config.epsilon, config.weight_decay);
      opt.mse_backward(input, target, 1); opt.adamw_step();
      close(parameters(opt), fixture.adamw_expected_parameters, tolerance);
      adamwError = Math.max(...parameters(opt).map((value, index) => Math.abs(value - fixture.adamw_expected_parameters[index])));
    } finally { opt.free(); }
    for (const mode of ['sgd', 'adamw']) {
      const model = NativeTrainableNetwork.from_dense_layers_json(fixture.dense_json, 1);
      try {
        if (mode === 'adamw') {
          const c = fixture.adamw_config;
          model.configure_adamw(c.learning_rate, c.beta1, c.beta2, c.epsilon, c.weight_decay);
        }
        for (const reference of fixture[mode + '_steps']) {
          close([model.mse_backward(input, target, 1)], [reference.loss], tolerance);
          close([...model.predictions()], reference.predictions, tolerance);
          close([...model.parameter_gradients()], reference.parameter_gradients, tolerance);
          close([...model.input_gradients()], reference.input_gradients, tolerance);
          if (mode === 'sgd') model.sgd_step(fixture.sgd_learning_rate); else model.adamw_step();
          close(parameters(model), reference.parameters, tolerance);
        }
      } finally { model.free(); }
    }
    candle = { verified: true, oracle: fixture.oracle, fixtureSha256: createHash('sha256').update(bytes).digest('hex'),
      parameterGradientError, inputGradientError, sgdError, adamwError, tolerance, sgdStepCount: 3, adamwStepCount: 3 };
  }
  return { actualAutograd: true, finiteDifferences: true, sgdLearning: true, adamwLearning: true,
    atomicUpdates: true, immutableSnapshot: true, closedLifecycle: true, firstLoss, lastLoss, adamFirst, adamLast, candle };
}
