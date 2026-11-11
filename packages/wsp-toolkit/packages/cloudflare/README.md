# ruv-fann-wsp-cloudflare — stateless neural operations for Workers

Expose bounded weight evaluation and diagnostics through stateless Cloudflare Worker routes. The dedicated build helper packages static scalar/SIMD WASM assets; it does not deploy an account or create persistent model authority.

**Topics:** Cloudflare Workers · static WebAssembly · authenticated neural evaluation · bounded HTTP APIs.

```sh
npm install ruv-fann-wsp-cloudflare@0.1.0
```

[Package on npm](https://www.npmjs.com/package/ruv-fann-wsp-cloudflare) · [Core 0.2.0 entrypoint: `ruv-fann/cloudflare`](https://github.com/ruvnet/ruv-FANN/tree/main/packages/core#capability-entrypoints) · [Interactive ruv-fann field guide](https://ruvnet.github.io/ruv-FANN/) · [Toolkit API and release guide](https://github.com/ruvnet/ruv-FANN/tree/main/packages/wsp-toolkit#readme)

Choose this standalone package for its declared dependency footprint, or use `ruv-fann/cloudflare` from the complete `ruv-fann@0.2.0` facade. Package version: **0.1.0**. [Umbrella release and evidence](https://github.com/ruvnet/ruv-FANN/releases/tag/ruv-fann-toolkit-v0.2.0).

Stateless HTTP operations over `ruv-fann-wsp@0.2.0`, with a build adapter that imports real precompiled WebAssembly modules into Cloudflare Workers. Reuses one initialized runtime per backend. It never downloads Wasm at runtime and has no JavaScript numerical fallback.

```js
import { createWorkerHandler } from 'ruv-fann-wsp-cloudflare';

export default {
  fetch(request, env) {
    return createWorkerHandler({ bearerToken: env.WSP_BEARER_TOKEN }).fetch(request);
  }
};
```

Use the provided static-Wasm build for deployment. Bundling the handler with a generic browser build leaves the underlying runtime's asset loader unsuitable for Workers.

```sh
npm run build --workspace ruv-fann-wsp-cloudflare
```

The build emits `dist/worker.js`, `dist/core.wasm` and `dist/simd.wasm` within this package. `buildWorker(directory)` is also available from `ruv-fann-wsp-cloudflare/build` for custom output directories. The included `wrangler.jsonc` prepares a Worker named `ruv-fann-wsp-validation`; review the name and deployment account before deployment. Set `WSP_BEARER_TOKEN` with the Cloudflare secret flow. Tokens must contain 24..1024 printable non-space ASCII characters. Nothing in the build command deploys resources.

Cloudflare's current Workers runtime supports the imported SIMD module. `simd: false` selects the scalar module, and `true` or `auto` selects SIMD. Both backends are exercised in real local `workerd`, with independent XOR and derivative oracles. There is no runtime compilation or Node compatibility requirement.

| Route | Method | JSON body |
| --- | --- | --- |
| `/health` | GET | No body. Public, reports configuration and byte limit. |
| `/evaluate` | POST | `{model, patch, dataset, policy, options?}` |
| `/trace` | POST | `{model, batch, options?}` |
| `/sensitivity` | POST | `{model, dataset, locations, options?}` |
| `/ablation` | POST | `{model, layer, neuron, limits?}` |

Authenticated operations require `Authorization: Bearer …` and `Content-Type: application/json`. Their payloads and successful responses use the WSP types and schemas. `/evaluate` returns HTTP 200 with `accepted: false` when a valid patch fails its supplied evaluation policy. A caller-selected policy is evidence for that request, not authorization to change a live model.

Unknown paths return 404, unsupported methods 405, missing configuration 503, invalid credentials 401, malformed UTF-8/JSON 400, unsupported media types 415, oversized bodies 413, and bounded WSP input errors 422. Unexpected execution failures return a generic 500 without a stack trace. Responses set `Cache-Control: no-store`. Query parameters and unknown payload fields are rejected. There is no CORS policy or outbound URL selection.

`createWorkerHandler` accepts `maxRequestBytes` to tighten the default 512 KiB request cap, up to the hard 1 MiB maximum. It counts streamed bytes independently of `Content-Length`. Existing WSP limits additionally bound dimensions, rows, parameters, numerical sensitivity and aggregate work. Authentication fails closed unless a token is configured; `allowUnauthenticated: true` is an explicit option for controlled local tests.

This package stores no checkpoints, accepts no generated code, and exposes no promotion, rollback or shell endpoint. Application owners must provide trusted datasets and policies, request rate/concurrency controls and a durable revision authority if they add state. Hosted deployment has not been implied by successful local `workerd` validation.

<!-- constellation-links:start -->

Explore the [ruv-fann field guide](https://ruvnet.github.io/ruv-FANN/), [constellation onramp](https://www.npmjs.com/package/ruvnet-constellation) and [shared contracts](https://www.npmjs.com/package/ruvnet-constellation-contracts). Optional ecosystem primitives: [ruv-fann-wsp](https://www.npmjs.com/package/ruv-fann-wsp), [AgentDB](https://www.npmjs.com/package/agentdb), [RuVector](https://www.npmjs.com/package/ruvector). Optional orchestration: [MetaHarness](https://www.npmjs.com/package/metaharness) and [Ruflo](https://www.npmjs.com/package/ruflo).

These links describe integrations; package.json defines install-time dependencies. The [versioned catalog](https://github.com/ruvnet/ruv-FANN/blob/main/packages/wsp-toolkit/packages/constellation/catalog.json) declares release profiles and package versions; registry availability and publication provenance require independent verification.

<!-- constellation-links:end -->

## Discoverability topics

Cloudflare Workers · static WebAssembly modules · edge neural inference · authenticated evaluation routes · bounded HTTP APIs · activation tracing · neuron ablation and sensitivity · scalar and SIMD assets.
