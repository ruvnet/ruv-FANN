import { init, type EMeansOptions } from '@ruv-fann/integrations/clustering';
import { trainProductQuantizer, encodeProductQuantizer, decodeProductQuantizer, type ProductQuantizer } from '@ruv-fann/integrations/clustering/quantization';
import { partition, postingList, type IVFPartition } from '@ruv-fann/integrations/clustering/ivf';
import { createClusteringEvaluator } from '@ruv-fann/integrations/clustering/metaharness';
import { createClusteringWorker, type WorkerOptions } from '@ruv-fann/integrations/clustering/cloudflare';

const data = new Float32Array([1, 2, 3, 4]);
const options: EMeansOptions = { clusters: 1 };
await init();
const model: ProductQuantizer = trainProductQuantizer(data, 2, 2, options);
const decoded: Float32Array = decodeProductQuantizer(model, encodeProductQuantizer(model, data));
const ivf: IVFPartition = partition(decoded, 2, options);
const list: Uint32Array = postingList(ivf, 0);
const evaluator: typeof createClusteringEvaluator = createClusteringEvaluator;
declare const workerOptions: WorkerOptions;
const worker = createClusteringWorker(workerOptions);
void [list, evaluator, worker];
// @ts-expect-error float32 vectors are required
trainProductQuantizer([1, 2], 2, 1, options);
