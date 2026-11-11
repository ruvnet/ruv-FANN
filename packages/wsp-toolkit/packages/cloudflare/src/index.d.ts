export interface WorkerHandlerOptions {
  /** 24..1024 printable non-space ASCII characters. Store as a Worker secret. */
  bearerToken?: string;
  /** Explicitly opts out of authentication; defaults to false. Use only for local tests. */
  allowUnauthenticated?: boolean;
  /** May tighten the hard 1 MiB limit. Defaults to 512 KiB. */
  maxRequestBytes?: number;
}
export interface WorkerHandler { fetch(request: Request): Promise<Response> }
export const MAX_REQUEST_BYTES: number;
/** POST /evaluate, /trace, /sensitivity, /ablation; GET /health. No durable state or promotion. */
export function createWorkerHandler(options?: WorkerHandlerOptions): Readonly<WorkerHandler>;
