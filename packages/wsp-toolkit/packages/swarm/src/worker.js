import { parentPort, workerData, threadId } from 'node:worker_threads';
import { searchWeightPatch } from 'ruv-fann-wsp-search';

try {
  if (!parentPort) throw new Error('proposal worker requires a parent port');
  if (workerData.kind === 'patch') {
    parentPort.postMessage({ status: 'ok', patch: workerData.patch, workerThreadId: threadId });
  } else if (workerData.kind === 'search') {
    const result = await searchWeightPatch(workerData.model, workerData.dataset, workerData.policy, workerData.searchOptions);
    parentPort.postMessage({ status: 'ok', patch: result.patch, workerThreadId: threadId,
      search: { evaluations: result.evaluations, inferencePasses: result.inferencePasses,
        reservedWork: result.reservedWork, backend: result.backend } });
  } else {
    throw new Error('unsupported proposal worker operation');
  }
} catch (error) {
  parentPort?.postMessage({ status: 'error', workerThreadId: threadId,
    error: typeof error?.message === 'string' ? error.message.slice(0, 512) : 'proposal execution failed' });
} finally { parentPort?.close(); }
