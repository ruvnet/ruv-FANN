import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { RuvSwarm } from 'ruv-swarm';
const require = createRequire(import.meta.url);
const apiRequire = createRequire(require.resolve('ruv-swarm/package.json'));
const Database = apiRequire('better-sqlite3');
const database = new Database(':memory:');
try { assert.equal(database.prepare('SELECT 2 + 3 AS result').get().result, 5); }
finally { database.close(); }
assert.equal(RuvSwarm.getVersion(), '2.0.2');
const runtime = await RuvSwarm.initialize();
const network = runtime.createNeuralNetwork([2, 1], 'Linear');
try {
  network.set_weights(Float64Array.of(2, -1));
  assert.deepEqual(Array.from(network.run(Float64Array.of(3, 4))), [2]);
} finally { network.free(); }
const model = runtime.createForecastingModel('linear');
try { assert.deepEqual(Array.from(model.predict(Float64Array.of(1, 3, 5))), [7]); }
finally { model.free(); }
console.log('Current packed candidate: native SQLite, actual inference and forecast passed.');
