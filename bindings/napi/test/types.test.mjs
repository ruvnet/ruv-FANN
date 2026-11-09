/** Compile the published API using the already locked optional integration tooling. */
import test from 'node:test';
import assert from 'node:assert/strict';
import ts from '../../../integrations/ruv-fann/node_modules/typescript/lib/typescript.js';
import { fileURLToPath } from 'node:url';

test('public declarations type actual inference/training use and reject malformed calls', () => {
  const program = ts.createProgram([fileURLToPath(new URL('./types.fixture.mts', import.meta.url))], {
    target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.NodeNext,
    moduleResolution: ts.ModuleResolutionKind.NodeNext, strict: true, noEmit: true, types: [],
  });
  const diagnostics = ts.getPreEmitDiagnostics(program);
  assert.deepEqual(diagnostics.map(diagnostic => ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n')), []);
});
