import { init, type EMeansOptions } from 'ruv-fann/clustering';
import { trainProductQuantizer, encodeProductQuantizer, decodeProductQuantizer, type ProductQuantizer } from 'ruv-fann/clustering/quantization';
import { partition, postingList, type IVFPartition } from 'ruv-fann/clustering/ivf';
import { createClusteringEvaluator, type ClusteringEvaluator } from 'ruv-fann/clustering/metaharness';
import { createClusteringWorker, type WorkerOptions } from 'ruv-fann/clustering/cloudflare';

const data = new Float32Array([1, 2, 3, 4]);
const options: EMeansOptions = { clusters: 1 };
await init();
const model: ProductQuantizer = trainProductQuantizer(data, 2, 2, options);
const reconstructed: Float32Array = decodeProductQuantizer(model, encodeProductQuantizer(model, data));
const ivf: IVFPartition = partition(reconstructed, 2, options);
const list: Uint32Array = postingList(ivf, 0);
const evaluator: typeof createClusteringEvaluator = createClusteringEvaluator;
declare const instance: ClusteringEvaluator;
declare const workerOptions: WorkerOptions;
const worker: { fetch(request: Request): Promise<Response> } = createClusteringWorker(workerOptions);
void [list, evaluator, instance, worker];
// @ts-expect-error byte codes are required
decodeProductQuantizer(model, new Uint32Array());
