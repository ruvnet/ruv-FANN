import { NativePreparedNetwork, NativePreparedBatch, NativeTrainableNetwork } from '@ruv-fann/napi';

const prepared = NativePreparedNetwork.from_dense_layers_json('{}');
const input = new Float64Array(prepared.input_size);
const result: Float64Array = prepared.run(input);
prepared.run_into(input, new Float64Array(prepared.output_size));
const batch: NativePreparedBatch = prepared.create_batch(32);
batch.run_into(input, result, 1);
const training = NativeTrainableNetwork.from_dense_layers_json('{}', 32);
const loss: number = training.mse_backward(input, result, 1);
const gradients: Float64Array = training.parameter_gradients();
training.configure_adamw(0.005, 0.9, 0.99, 1e-8, 0.01);
training.adamw_step();
training.sgd_step(0.01);
const frozen: NativePreparedNetwork = training.prepare();
const checkpoint: string = training.to_dense_layers_json();
void [loss, gradients, frozen, checkpoint];
// @ts-expect-error Factories are the only public constructors.
new NativePreparedNetwork();
// @ts-expect-error Float32 input is a different public numeric type.
prepared.run(new Float32Array(2));
// @ts-expect-error Raw native scalars require primitive numbers.
training.sgd_step('0.01');
// @ts-expect-error Training requires an explicit bounded capacity.
NativeTrainableNetwork.from_dense_layers_json('{}');
prepared.free(); batch.free(); training.free(); frozen.free();
