import { test } from 'node:test';
import { writeFile } from 'node:fs/promises';
import { externalEvidencePath } from './fixture.mjs';
import { runAcceptance } from './acceptance.mjs';

test('native FANN model -> actual core/SIMD inference -> RuVector flat/HNSW search -> real Darwin tuning', { timeout: 120000 }, async () => {
  const result = await runAcceptance();
  const target = process.env.RUV_FANN_INTEGRATION_EVIDENCE;
  if (target) {
    await writeFile(await externalEvidencePath(target), JSON.stringify(result, null, 2) + '\n', { flag: 'wx' });
  }
});
