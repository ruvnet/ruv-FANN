# ruv-fann-wsp — weight-space programming for JavaScript

Propose, evaluate and version direct neural-network weight edits in **Node.js and browsers**. `ruv-fann-wsp` runs real ruv-FANN WebAssembly inference to compare a candidate with its baseline, then lets your application explicitly promote or reject the proposal. No gradient updates are involved.

An LLM, a human or an algorithm supplies the patch. This package does not call a model provider, execute generated code, or guarantee that a weight edit generalizes beyond your evaluation cases.

## Install

```sh
npm install ruv-fann-wsp
```

ESM, Node.js 20.9+; TypeScript declarations included. The package depends on `ruv-fann@0.1.2`, which supplies the WASM runtime. Browser use requires a secure context with Web Crypto and the core package's [WASM asset deployment layout](https://github.com/ruvnet/ruv-FANN/tree/main/packages/core#browser-setup).

## Repair a neural network without training

This complete example repairs an XOR network by directly setting its two output weights. The hidden layer already computes `ReLU(x − y)` and `ReLU(y − x)`. The patch makes the output their sum.

```js
import { createWeightWorkspace } from 'ruv-fann-wsp';

const model = {
  schema: 'ruv-fann.dense-layers/1',
  layers: [
    { inputs: 2, outputs: 2, weights: [1, -1, -1, 1], bias: [0, 0],
      activation: 'ReLU', steepness: 1 },
    { inputs: 2, outputs: 1, weights: [0, 0], bias: [0],
      activation: 'Linear', steepness: 1 }
  ]
};
const workspace = await createWeightWorkspace(model);
const baseline = workspace.snapshot();
const proposal = await workspace.propose({
  schema: 'ruv-fann.weight-patch/1',
  baseHash: baseline.hash,
  edits: [
    { layer: 1, parameter: 'weights', index: 0, value: 1 },
    { layer: 1, parameter: 'weights', index: 1, value: 1 }
  ]
}, {
  inputs: [0, 0, 0, 1, 1, 0, 1, 1],
  targets: [0, 1, 1, 0],
  rows: 4
}, {
  maxMSE: 0,
  maxRegressionMSE: 0,
  maxAbsoluteError: 0
});
console.log(proposal.evaluation.accepted);       // true
console.log(proposal.evaluation.baseline.mse);   // 0.5
console.log(proposal.evaluation.candidate.mse);  // 0

const promoted = workspace.promote(proposal.id, baseline.revision);
console.log(promoted.revision); // 2

// Rollback creates another revision; it does not rewind the revision counter.
const restored = workspace.rollback(baseline.revision, promoted.revision);
console.log(restored.revision); // 3
console.log(restored.hash === baseline.hash); // true
```

The four cases above exhaust the binary XOR domain. For larger tasks, keep evaluation cases separate from examples shown to the patch generator. The package cannot verify whether supplied cases were previously seen by an LLM.

## Ask an agent for a patch

```js
import { createWeightEditPrompt } from 'ruv-fann-wsp';

// `model` is your dense model checkpoint.
const prompt = await createWeightEditPrompt(
  model,
  'Make the output compute XOR for binary inputs. Edit only the output weights.'
);
// Send `prompt` through your existing agent/provider integration.
// Parse its JSON response as `patch`, then evaluate before promotion.
```

The prompt includes the normalized checkpoint, model fingerprint, edit bounds and strict JSON patch format. Natural-language requests are instructions for the generator, not enforced semantic guarantees. Use structural constraints and evaluation policy to determine acceptance. `applyWeightPatch(model, patch)` validates and returns a new checkpoint; it does not certify behavior or promote it.

## Browser deployment

Serve the installed packages on your own origin, keeping `ruv-fann-wasm/src` beside `ruv-fann-wasm/dist`. The WSP and core packages each need their `src` directory. Map their package names before importing WSP:

```html
<script type="importmap">
{
  "imports": {
    "ruv-fann-wsp": "/vendor/ruv-fann-wsp/src/index.js",
    "ruv-fann": "/vendor/ruv-fann/src/index.js",
    "ruv-fann-wasm": "/vendor/ruv-fann-wasm/src/index.js"
  }
}
</script>
<script type="module">
  import { createWeightWorkspace } from 'ruv-fann-wsp';
  // Use the same model and proposal API as the Node.js example above.
</script>
```

Serve `.wasm` assets as `application/wasm`; allow WebAssembly compilation in your Content Security Policy. HTTPS or localhost is required for Web Crypto. Bundlers must preserve the runtime's dynamic imports and relative WASM asset URLs.

## API

| Export | Purpose |
| --- | --- |
| `fingerprintModel(model)` | Validate and SHA-256 hash a canonical model |
| `applyWeightPatch(model, patch, limits?)` | Apply bounded parameter replacements to a copy |
| `evaluateWeightPatch(model, patch, dataset, policy, options?)` | Compare baseline/candidate with real WASM predictions |
| `createWeightEditPrompt(model, specification, limits?)` | Produce a bounded provider-neutral prompt |
| `createWeightWorkspace(model, options?)` | Manage in-memory proposals, promotion and rollback |

Model inputs accept a `DenseModel` object or JSON string. Weights use row-major `[outputs, inputs]` order. Patch indices are zero-based. Missing/null biases normalize to explicit zeros; negative zero normalizes to zero. Equivalent normalized checkpoints share a fingerprint.

### Patch format

```json
{
  "schema": "ruv-fann.weight-patch/1",
  "baseHash": "<64-character SHA-256 of the normalized baseline>",
  "edits": [
    { "layer": 0, "parameter": "weights", "index": 0, "value": 1 }
  ]
}
```

Edits replace individual weights or biases. They cannot change layer dimensions, activation functions or topology. Empty edits, duplicate locations, unknown fields, out-of-range indices, nonfinite values and incorrect baseline hashes reject. Caller inputs are not modified.

### Evaluation policy

Both `maxMSE` and `maxRegressionMSE` are required, finite and nonnegative. A candidate passes only when:

- Candidate MSE is at most `maxMSE`.
- Candidate MSE minus baseline MSE is at most `maxRegressionMSE`.
- If supplied, candidate maximum absolute error is at most `maxAbsoluteError`.

The result records baseline/candidate errors, acceptance reasons, the candidate checkpoint, backend, row count, and model/dataset/policy hashes. Hashes bind evidence to content; they are not signatures or proof that a caller-supplied result is genuine. In-memory promotion uses an internally retained proposal, never a result object supplied by the caller.

### Workspace lifecycle

`workspace.snapshot()` returns a defensive checkpoint copy. `propose()` evaluates without changing the active model. `promote(proposalId, expectedRevision)` requires a passing, current proposal and the expected revision. `rollback(targetRevision, expectedRevision)` restores a retained checkpoint as a new revision. `history()` returns checkpoint metadata.

Up to 16 pending proposals and 32 checkpoints are retained. Stale proposals cannot overwrite a newer revision. Workspace history is in memory; persist checkpoints or use the AgentDB integration for durable storage. A previously prepared core model is an immutable snapshot: call `runtime.prepare(newCheckpoint)` after promotion to run the new weights.

## Resource bounds

The initial release targets small dense networks and bounded evaluation batches:

| Resource | Maximum/default |
| --- | --- |
| Model parameters, including normalized biases | 65,536 |
| Model JSON | 1 MiB |
| Evaluation rows | 1,024 |
| Input/target array elements | 65,536 each |
| Evaluation work | `rows × parameterCount ≤ 10,000,000` |
| Patch edits | 128 |
| Absolute replacement value | 1,000,000 |
| Absolute change from baseline | 1,000 |
| Prompt specification | 8,000 characters |
| Prompt size | 64 KiB; model portion at most 48 KiB |

Patch limits may be tightened, not raised beyond these defaults. The core runtime's native-memory bounds also apply. Evaluation is synchronous once the runtime is loaded; use a Worker for workloads that would block your application's event loop. Unsupported operations and numerical failures reject rather than yielding a passing evaluation.

## AgentDB and MCP

The AgentDB integration adds durable programs, proposals and revision history using the same patch/evaluation API. It separates proposal evaluation from promotion and retains rollback checkpoints. MCP access policies restrict mutations; no generated code is executed. See [AgentDB](https://github.com/ruvnet/agentdb) for the published integration version and tool documentation.

## Scope

For research context, see [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). This package supplies execution and evaluation tooling; it does not reproduce the reported research results.

This is a runtime and review workflow for direct weight edits, not a transformer checkpoint editor, LLM provider, automatic optimizer or proof of generalization. Small symbolic examples demonstrate execution and lifecycle behavior; useful task-specific edits still need appropriate evaluation data and application-level acceptance criteria.

[Source](https://github.com/ruvnet/ruv-FANN/tree/main/packages/wsp) · [Issues](https://github.com/ruvnet/ruv-FANN/issues) · [MIT license](https://github.com/ruvnet/ruv-FANN/blob/main/packages/wsp/LICENSE)
