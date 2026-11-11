export interface WorkerBuild { script: string; coreWasm: string; simdWasm: string }
/** Node-only build helper. Bundles installed WSP and wasm-bindgen bindings, with static Wasm imports. */
export function buildWorker(outDirectory: string): Promise<Readonly<WorkerBuild>>;
