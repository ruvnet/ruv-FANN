/** Reproducible local inference benchmark; no synthetic scores. */
import { neuralCLI } from './neural.js';
import { readFile, writeFile } from 'node:fs/promises';

export class BenchmarkCLI {
  async run(args = []) {
    const result = await neuralCLI.benchmark(args);
    const output = this.getArg(args, '--output');
    if (output) {
      await writeFile(output, `${JSON.stringify(result, null, 2)}\n`);
    }
    return result;
  }
  async compare(args = []) {
    if (args.length !== 2) {
      throw new TypeError('compare requires two measured benchmark JSON files');
    }
    const data = await Promise.all(args.map(async file => JSON.parse(await readFile(file, 'utf8'))));
    for (const item of data) {
      if (item.kind !== 'dense-inference' || !Number.isFinite(item.operations_per_second) || item.operations_per_second <= 0 ||
          !Number.isSafeInteger(item.iterations) || item.checksum !== item.iterations * 2) {
        throw new TypeError('Only verified dense-inference benchmark records can be compared');
      }
    }
    const result = { kind: 'dense-inference-comparison', operations_per_second: data.map(item => item.operations_per_second),
      ratio: data[1].operations_per_second / data[0].operations_per_second };
    console.log(JSON.stringify(result, null, 2));
    return result;
  }
  getArg(args, flag) {
    const index = args.indexOf(flag); return index >= 0 ? args[index + 1] : null;
  }
}
export const benchmarkCLI = new BenchmarkCLI();
