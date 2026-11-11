# ruv-fann-wsp-huggingface — integrity-pinned dense checkpoints

Load a public dense-layer JSON checkpoint from an immutable Hub commit and verify its exact bytes before parsing. An explicit affine-head adapter covers simple weight vectors without pretending to load an arbitrary transformer.

**Topics:** Hugging Face Hub · dense JSON checkpoints · SHA-256 integrity · affine heads.

```sh
npm install ruv-fann-wsp-huggingface@0.1.0
```

[Package on npm](https://www.npmjs.com/package/ruv-fann-wsp-huggingface) · [Core 0.2.0 entrypoint: `ruv-fann/huggingface`](https://github.com/ruvnet/ruv-FANN/tree/main/packages/core#capability-entrypoints) · [Interactive ruv-fann field guide](https://ruvnet.github.io/ruv-FANN/) · [Toolkit API and release guide](https://github.com/ruvnet/ruv-FANN/tree/main/packages/wsp-toolkit#readme)

Choose this standalone package for its declared dependency footprint, or use `ruv-fann/huggingface` from the complete `ruv-fann@0.2.0` facade. Package version: **0.1.0**. [Umbrella release and evidence](https://github.com/ruvnet/ruv-FANN/releases/tag/ruv-fann-toolkit-v0.2.0).

Load a public Hugging Face **dense-layers JSON** checkpoint using an immutable
commit revision and a caller-supplied SHA-256. The adapter verifies downloaded
bytes before parsing and validates the model with `ruv-fann-wsp@0.2.0`.

Configuration template: replace the repository, immutable commit and independently reviewed raw-file digest before running.

```js
import { loadHubModel } from 'ruv-fann-wsp-huggingface';

const checkpoint = await loadHubModel({
  repo: 'your-owner/your-dense-model',
  revision: '<40 lowercase hexadecimal commit characters>',
  path: 'model.json',
  sha256: '<64 lowercase hexadecimal raw-file digest characters>'
});
console.log(checkpoint.inspection.baseHash); // Canonical WSP model hash.
```

Supply an independently reviewed digest and commit SHA in application
configuration. Neither metadata field establishes who authored the model. The
raw-file digest can differ from the canonical WSP model hash because whitespace
and absent zero biases affect file bytes. The returned provenance records both
through `provenance.sha256` and `inspection.baseHash`.

`maxBytes` can tighten the 1 MiB download cap. `timeoutMs` defaults to 15000 and
cannot exceed 60000. `signal` supports cancellation. A trusted `fetch` function
can be injected for transport integration or deterministic contract tests; it
must honor `AbortSignal` and manual redirect mode.

At most three redirects may stay on `https://huggingface.co`. Requests omit
cookies, credentials and referrers; this package accepts no API token. External
CDN/LFS redirects are rejected. Place a small JSON file directly in the Hub
repository rather than using LFS. Server-side integrity checks also require a
readable response stream; opaque browser redirects fail closed.

`loadHubLinearHead(source, options?)` explicitly loads a different JSON format:
exactly `{ "weights": [/* nonempty finite numbers */], "bias": 0 }`. It interprets
these parameters as **one affine projection** `sum(weights[i] * input[i]) + bias`
and constructs a single-output `Linear` layer with steepness 1. Its result has
schema `ruv-fann.hub-linear-head/1` and `interpretation: 'affine-projection'`.
The same pinned download, digest, byte and redirect restrictions apply. Unknown
keys, nonfinite numbers and more than 65535 weights are rejected. The generic
`loadHubModel()` does not infer or convert this format automatically.

This explicit affine interpretation does not reproduce the source model's
activations, feature extraction, preprocessing or complete inference pipeline,
and it establishes no task accuracy. Supply the appropriate feature vector and
independent evaluation cases for the behavior you intend to test.

This adapter does not convert transformer checkpoints, load pickle, interpret
safetensors, call inference endpoints, upload repositories or promote patches.
Node.js 20.9+ and secure-context browsers with Web Crypto are supported.

Run `npm test` inside this package after installing workspace dependencies. Tests
use injected transports and explicitly do not claim live Hub execution.

<!-- constellation-links:start -->

Explore the [constellation onramp](../constellation/README.md) and [shared contracts](../contracts/README.md). Optional ecosystem primitives: [ruv-fann-wsp](https://www.npmjs.com/package/ruv-fann-wsp), [AgentDB](https://www.npmjs.com/package/agentdb), [RuVector](https://www.npmjs.com/package/ruvector). Optional orchestration: [MetaHarness](https://www.npmjs.com/package/metaharness) and [Ruflo](https://www.npmjs.com/package/ruflo).

These links describe integrations; they do not add install-time dependencies. Workspace package candidates and namespace proposals are listed explicitly in the catalog.

<!-- constellation-links:end -->
