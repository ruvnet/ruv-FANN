import { createAgentDBRegistry } from '../../packages/registry/src/index.js';

// This private fixture deliberately exercises AgentDB's buffered WASM backend.
// Optional native SQLite availability must not change disk-flush failure oracles.
// Published registry adapters never select or alter the caller's database backend.
process.env.AGENTDB_FORCE_SQLJS = '1';

/** Explicit integration fixture. Dependencies are installed by setup, never here. */
export async function openDemoRegistry(filename, registryFactory = createAgentDBRegistry) {
  if (typeof filename !== 'string' || filename.length < 1 || filename.length > 4096 || /[\x00-\x1f]/.test(filename))
    throw new TypeError('filename must be a bounded local database path');
  if (typeof registryFactory !== 'function') throw new TypeError('registryFactory must be a trusted constructor function');
  let createDatabase, WeightPrograms;
  try {
    ({ createDatabase } = await import('agentdb/db-fallback'));
    ({ WeightPrograms } = await import('agentdb/controllers/WeightPrograms'));
  } catch (error) {
    throw new Error('AgentDB integration runtime is missing; run npm ci --prefix tooling/agentdb --ignore-scripts --omit=optional', { cause: error });
  }
  const db = await createDatabase(filename);
  let programs;
  try {
    programs = new WeightPrograms(db, { simd: false });
    return registryFactory(programs, {
      flush: () => db.save?.(),
      close: ({ poisoned }) => {
        programs.dispose();
        // Every acknowledged mutation was already flushed. Avoid sql.js close()
        // performing another write, including from a stale read-only instance.
        // On poison this also discards uncertain unflushed memory; inspect disk
        // after reopen rather than assuming a failed acknowledgement was absent.
        if (typeof db.save === 'function') db.save = () => {};
        db.close();
      }
    });
  } catch (error) {
    programs?.dispose();
    if (typeof db.save === 'function') db.save = () => {};
    try { db.close(); } catch { /* preserve setup error */ }
    throw error;
  }
}
