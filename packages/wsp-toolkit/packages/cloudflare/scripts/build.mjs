import { buildWorker } from '../src/build.js';
const output = process.argv[2] ?? new URL('../dist', import.meta.url).pathname;
console.log(JSON.stringify(await buildWorker(output), null, 2));
