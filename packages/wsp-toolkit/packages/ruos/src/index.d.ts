export function createValidationCommand(directory: string): string;
/** Supply only an authentic host tool response. This is not a cryptographic attestation. */
export function assertRuOSCompletion(toolResult: {isError?: boolean; structuredContent?: Record<string, unknown>}, expectedMachineId: string): Readonly<{completionVersion: 1; completionVerified: true; exitCode: 0; machineId: string; assertionsIndependentlyVerified: false}>;
export function runWasmCanary(): Promise<{schema: 'ruv-fann.wasm-canary/1'; backend: 'wasm' | 'wasm-simd'; baselineHash: string; repairedHash: string; restoredHash: string; baselineMSE: number; repairedMSE: number; rejectedMSE: number; finalRevision: number; assertions: number}>;
