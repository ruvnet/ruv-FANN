import { createLab } from './engine.js';
let inFlight = 0;
const json = (data, status = 200) => new Response(JSON.stringify(data), { status,
  headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' } });

// Deliberately stateless: the public evaluator cannot publish or promote a model.
// Cases and policy come from our source, never the request.
export async function evaluateRequest(request) {
  if (request.method !== 'POST') return json({ error: 'Use POST' }, 405);
  if (request.headers.get('content-type')?.split(';')[0].trim() !== 'application/json') return json({ error: 'Use application/json' }, 415);
  if (inFlight >= 2) return json({ error: 'Evaluator busy; try again shortly' }, 429);
  const declared = request.headers.get('content-length');
  if (declared !== null && (!/^\d+$/.test(declared) || Number(declared) > 1024)) return json({ error: 'Request too large' }, 413);
  inFlight++;
  try {
    let size = 0; const chunks = []; const reader = request.body?.getReader();
    if (!reader) return json({ error: 'Body required' }, 400);
    try {
      for (;;) {
        const { done, value } = await reader.read(); if (done) break;
        size += value.byteLength;
        if (size > 1024) { await reader.cancel(); return json({ error: 'Request too large' }, 413); }
        chunks.push(value);
      }
    } finally { reader.releaseLock(); }
    const bytes = new Uint8Array(size); let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
    let input;
    try { input = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes)); }
    catch { return json({ error: 'Invalid JSON or UTF-8' }, 400); }
    if (!input || Array.isArray(input) || typeof input !== 'object' ||
        Object.keys(input).some(k => !['scenario', 'weights', 'baseWeights'].includes(k)) ||
        !['robot', 'room'].includes(input.scenario)) return json({ error: 'Invalid scenario or fields' }, 422);
    if (![input.weights, input.baseWeights].every(w => Array.isArray(w) && w.length === 2 && w.every(v => typeof v === 'number' && Number.isFinite(v) && Math.abs(v) <= 2))) return json({ error: 'Two finite weights per model, each in [-2,2], are required' }, 422);
    const lab = await createLab(input.scenario, 'auto', input.baseWeights);
    const evidence = await lab.evaluate(input.weights);
    return json({ ...evidence, location: 'edge', scope: 'Public fixed finite-domain evaluation; no persistence or promotion authority' });
  } catch {
    return json({ error: 'Evaluation failed' }, 500);
  } finally { inFlight--; }
}
