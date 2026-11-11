# ruv-fann-wsp toolkit

Twelve independent ESM subpackages extend `ruv-fann-wsp@0.2.0` with regression gates, bounded search, checkpoint transport and platform adapters. They use the published `ruv-fann@0.1.2` / `ruv-fann-wasm@0.1.2` runtime. The existing core package and its release lanes remain unchanged. This workspace prepares candidate packages; version `0.1.0` has not been published.

| Package | Purpose | Public API |
| --- | --- | --- |
| [ruv-fann-wsp-lifecycle](packages/lifecycle/README.md) | CI regression suites and isolated candidate snapshots | `runRegressionGate`, `evaluatePatchBranches`, `wsp-gate` CLI |
| [ruv-fann-wsp-search](packages/search/README.md) | Bounded discrete optimization; experimental symbolic/plasticity proposals | `searchWeightPatch`, `compileBooleanTable`, `createHebbianPatch` |
| [ruv-fann-wsp-huggingface](packages/huggingface/README.md) | Commit-pinned, digest-verified public Hub JSON | `loadHubModel`, `loadHubLinearHead` |
| [ruv-fann-wsp-cloudflare](packages/cloudflare/README.md) | Authenticated stateless HTTP operations and static WASM build | `createWorkerHandler`, `buildWorker` |
| [ruv-fann-wsp-ruos](packages/ruos/README.md) | WASM lifecycle canary, shell plan, executor completion checks | `runWasmCanary`, `createValidationCommand`, `assertRuOSCompletion` |
| [ruv-fann-wsp-invariants](packages/invariants/README.md) | Finite-domain output constraints and exhaustive Boolean checks | `checkModelInvariants`, `estimateInvariantWork`, `createBooleanInvariant` |
| [ruv-fann-wsp-swarm](packages/swarm/README.md) | Real bounded proposal workers with separate WASM instances | `runProposalSwarm`, `estimateSwarmWork` |
| [ruv-fann-wsp-registry](packages/registry/README.md) | AgentDB revision authority with durable acknowledgements | `createAgentDBRegistry` |
| [ruv-fann-wsp-loop](packages/loop/README.md) | Autonomous local proposal/evaluation/promotion loop | `runEvolutionLoop` |
| [ruvnet-constellation-contracts](packages/contracts/README.md) | Dependency-free catalog/module contracts and canonical IDs | `CONTRACTS`, `validateCatalog`, `validateModuleManifest` |
| [ruvnet-constellation](packages/constellation/README.md) | Lightweight explicit discover/plan/scaffold onramp | `readCatalog`, `planProfile`, `scaffoldSatellite` |
| [ruvnet-constellation-telemetry](packages/telemetry/README.md) | Public cohort metrics, sampled project attach and standalone dashboard | `collectPublicMetrics`, `analyzeConstellation`, `renderConstellationDashboard` |

## Constellation onramp and dashboard

```sh
cd packages/wsp-toolkit
npm run constellation -- plan weights
npm run constellation -- plan local-loop
npm run constellation:report
```

The onramp plans explicit installs or source candidates and scaffolds reviewable local satellites. It installs no ML stack and collects no telemetry. The dashboard collects public metrics only when invoked; its HTML works offline. Default coverage is 12 verified public packages plus 12 local candidates. Measured project attach needs a separate consenting snapshot input; download correlations cannot establish it. See the [growth constellation guide](../../docs/constellation.md), [catalog](packages/constellation/catalog.json) and [protected OIDC setup](../../docs/constellation-release.md).

## Run

The qualification workspace needs Node 22.13+ (Miniflare requires Node 22). Published package candidates support Node 20.9+; the original five-package scenarios were also run on Node 20.20.2 through ruOS. The new loop uses Node worker threads; it does not run inside a browser or Cloudflare Worker.

```sh
cd packages/wsp-toolkit
npm ci --ignore-scripts --no-audit --no-fund
npm run setup:agentdb
npx playwright install --with-deps chromium
npm run validate
npm run demo:loop
```

Validation requires strict declarations, passing tests with zero skips/todos, matching packed-consumer behavior, a benchmark oracle and zero npm advisories across both installed locked dependency trees. The real AgentDB integration runtime is isolated under `tooling/agentdb`; setup omits its unrelated optional native/model packages and never downloads a model. `.artifacts/validation.json` records the test counts, benchmark, lockfile digest and source-file digests. `npm test`, `npm run test:unit`, `npm run test:e2e`, `npm run check:types`, `npm run benchmark`, `npm run audit` and `npm run test:loop` are also available separately.

The opt-in live Hub check uses a pinned public [source record](docs/huggingface-source.json):

```sh
npm run test:hub:live
# In managed Node 24 environments using an inherited HTTP proxy:
NODE_USE_ENV_PROXY=1 npm run test:hub:live
```

`ruv/ruview/presence-head.json` is an affine vector and bias. The explicit adapter maps these to a Linear layer and tests its arithmetic with synthetic inputs. This validates transport and declared affine semantics; it does not measure WiFi presence accuracy or reconstruct the source pipeline. No Hub token, upload or GPU job is needed.

## Run an autonomous loop

`npm run demo:loop` executes five proposal jobs with concurrency three against two small models. Search workers find shortcuts on development cases; the coordinator rejects three of them using held-out suites, validates the useful candidates against independent output constraints, then promotes two checkpoints through real AgentDB. The demo closes and reopens its SQLite file, verifies the exact revisions, rolls both models back, and reopens again to check persistence. It writes an inspectable report and database in a unique `.artifacts/evolution-*` directory.

```text
active AgentDB revision → fixed-operation proposal workers
                       → independent regression + invariant gates
                       → fresh AgentDB proposal → revision CAS promotion
                       → next bounded generation / restart / rollback
```

The loop has explicit round, proposal, work, worker, timeout and cancellation limits. Workers receive development cases only. Held-out data and acceptance rules remain with the coordinator. A worker result cannot authorize a commit. Active hashes are refreshed before declaring goals met, so another coordinator's rollback invalidates cached goal evidence. Tests exercise multiple generations, two concurrent loops, stale results, aborts and a real post-promotion disk-flush failure. See the [runnable scenario](test/evolution-scenario.mjs), [loop API](packages/loop/README.md), [delivery contract](docs/loop-design.md), and [loop validation](docs/loop-validation.md).

AgentDB 3.10.0/sql.js performs actual persistence in this implementation. The registry adapter accepts a caller-owned controller, serializes operations across awaits, and acknowledges writes only after its persistence callback succeeds. Failed persistence blocks further work until reopen. The fixture uses AgentDB's own WSP 0.1.0 controller dependency, while the coordinator independently checks candidates with WSP 0.2.0. There are no automatic provider calls, generated-code execution, public package publication or billing actions. RuVector/MetaHarness/Agentic Flow routing and universal verification remain extension points.

## Repair and gate

```js
import { searchWeightPatch } from 'ruv-fann-wsp-search';
import { runRegressionGate } from 'ruv-fann-wsp-lifecycle';
import { createWeightWorkspace } from 'ruv-fann-wsp';

// model, dataset and policy are application-owned inputs.
const result = await searchWeightPatch(model, dataset, policy, {
  locations: [0, 1].map(index => ({layer: 1, parameter: 'weights', index, values: [0, 1]})),
  maxEvaluations: 8,
  maxPasses: 2
});
if (result.patch && result.evaluation.accepted) {
  const gate = await runRegressionGate({model, patch: result.patch, suites: [
    {name: 'heldout', dataset: independentDataset, policy}
  ]});
  if (gate.accepted) {
    const workspace = await createWeightWorkspace(model);
    const original = workspace.snapshot();
    // Re-evaluate the promotion policy inside the workspace; a gate receipt is not authority.
    const proposal = await workspace.propose(result.patch, independentDataset, policy);
    if (proposal.evaluation.accepted) {
      const next = workspace.promote(proposal.id, original.revision);
      workspace.rollback(original.revision, next.revision);
    }
  }
}
```

For a PR gate, write `{model,patch,suites:[{name,dataset,policy}]}` to JSON and run `wsp-gate file.json`: exit 0 accepts, 1 rejects, 2 reports invalid input/runtime failure. The [complete executable scenario](test/scenario.mjs) covers two-weight XOR repair, activation and derivative oracles, rejected ablation, branching, compiled Boolean logic, localized plasticity, a Hub transport fixture and rollback in Node and real Chromium.

## Platforms and boundaries

Cloudflare's builder statically imports both precompiled WASM modules. Its actual `workerd` tests verify scalar/SIMD evaluation, tracing, sensitivity, ablation and authentication. Default configuration denies operations without a bearer secret. The Worker has no persistence, promotion, outbound model fetch or code execution endpoint. See its package README for building and optional deployment; no live Worker was deployed by this task.

ruOS validation uses the observed Work Desktop executor, installs the same SHA-256-checked npm archives into a fresh temporary directory, builds the Worker, runs both packed WASM scenarios and repeats the live Hub check. The completion helper requires host-owned structured completion and never grants success from stdout text. It is a consistency check, not a cryptographic attestation or deployment permission.

The Boolean compiler covers complete truth tables of at most six binary inputs. Hebbian updates are bounded proposals requiring independent evaluation. Search allows trusted synchronous local objective callbacks, including discontinuous scores; its winner can still fail the authoritative regression policy. Callbacks are not sandboxes. Candidate branches share a runtime and are evaluated sequentially; they do not provide process isolation. No generalization, arbitrary safety-input repair, transformer editing, hidden behavioral keys or autonomous provider execution is claimed.

Performance qualification compares an identical 16-candidate one-coordinate sequence: cached baseline screening plus final authoritative validation uses 19 inference passes versus 32 for repeatedly evaluating both models. Timings are local observations. See the [original five-package validation](docs/validation.md), [current loop qualification](docs/loop-validation.md) and [delivery contract](docs/design.md).
