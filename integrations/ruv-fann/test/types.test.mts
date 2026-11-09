import { createEncoder, createNeuralIndex, type PreparedNetwork, type Metadata } from '@ruv-fann/integrations';
import { createRuvectorStore } from '@ruv-fann/integrations/ruvector';
import { tuneBatchRows } from '@ruv-fann/integrations/metaharness';
import { ruvectorWasmBytes } from '@ruv-fann/integrations/node';
import type { WasmPreparedNetwork } from '../../../ruv-swarm/npm/wasm/ruv_swarm_wasm.js';

declare const prepared: PreparedNetwork;
declare const wasmBytes: Uint8Array;
declare const actualFann: WasmPreparedNetwork;
createEncoder(actualFann).close();
const encoder = createEncoder(prepared, { batchRows: 7 });
const features: Float64Array = encoder.encode(Float64Array.of(1, 2, 3));
const store = await createRuvectorStore({ wasmBytes, dimensions: prepared.output_size });
const index = createNeuralIndex({ prepared, store, modelId: 'my-fixed-model' });
const metadata: Metadata = { label: 'example', count: 2 };
index.add({ id: 'item', input: Float64Array.of(1, 2, 3), metadata });
const score: number | undefined = index.search(Float64Array.of(1, 2, 3), { k: 1 })[0]?.score;
const receipt = await tuneBatchRows({ prepared, inputs: features, modelId: 'my-fixed-model' });
const recommended: number = receipt.recommendation.batchRows;
// @ts-expect-error Inference requires Float64 input, not vector-store precision.
encoder.encode(Float32Array.of(1, 2, 3));
// @ts-expect-error Initializer requires explicit shipped WASM bytes.
await createRuvectorStore({ dimensions: 3 });
// @ts-expect-error A model label is required for the neural index.
createNeuralIndex({ prepared, store });
void [features, score, recommended];
const localBytes: Uint8Array = await ruvectorWasmBytes();
void localBytes;
index.close(); encoder.close();
