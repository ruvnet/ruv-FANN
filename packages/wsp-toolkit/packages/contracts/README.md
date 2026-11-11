# ruvnet-constellation-contracts — bounded package and module metadata

Validate bounded constellation catalogs and module manifests without installing a neural runtime. These metadata contracts identify formats; they do not replace numerical validation or prove registry ownership.

**Topics:** package catalogs · module contracts · JSON validation · dependency-free metadata.

```sh
npm install ruvnet-constellation-contracts@0.1.0
```

[Package on npm](https://www.npmjs.com/package/ruvnet-constellation-contracts) · [Core 0.2.0 entrypoint: `ruv-fann/contracts`](https://github.com/ruvnet/ruv-FANN/tree/main/packages/core#capability-entrypoints) · [Interactive ruv-fann field guide](https://ruvnet.github.io/ruv-FANN/) · [Toolkit API and release guide](https://github.com/ruvnet/ruv-FANN/tree/main/packages/wsp-toolkit#readme)

Choose this standalone package for its declared dependency footprint, or use `ruv-fann/contracts` from the complete `ruv-fann@0.2.0` facade. Package version: **0.1.0**. [Umbrella release and evidence](https://github.com/ruvnet/ruv-FANN/releases/tag/ruv-fann-toolkit-v0.2.0).

Dependency-free ESM metadata contracts for the constellation catalog and satellite module manifests. Node 20.9+ or browsers with `TextEncoder`/`URL`.

`validateCatalog(value)` and `validateModuleManifest(value)` return owned, validated JSON data. They reject unknown fields, accessors, sparse arrays, duplicate identifiers, unsafe npm names/paths/URLs, unversioned schema IDs and excessive data. A catalog can contain up to 1,000 packages and 64 profiles within a 2 MiB canonical byte budget. `published` is caller-supplied status; validation does not prove namespace ownership or check a registry.

`CONTRACTS` names the existing WSP dense model, patch, evaluation, policy, dataset and finite-domain formats alongside the catalog/module metadata schemas. Versioned custom schema IDs are permitted; listing a contract does not establish behavioral compatibility. Real WSP validation, model hashing, evaluation and AgentDB revision authority remain separate.

JSON Schema documents are exported under `ruvnet-constellation-contracts/schemas/`. `module.schema.json` and `catalog.schema.json` describe metadata structure; JavaScript additionally enforces cross-field references, aggregate byte budgets and safe own data properties. `wsp-references.schema.json` records canonical schema references without reimplementing numerical/model limits.

No SDK, native driver, provider, telemetry collector, network client or install hook is included.

<!-- constellation-links:start -->

Explore the [ruv-fann field guide](https://ruvnet.github.io/ruv-FANN/), [constellation onramp](https://www.npmjs.com/package/ruvnet-constellation) and [shared contracts](https://www.npmjs.com/package/ruvnet-constellation-contracts). Optional ecosystem primitives: [ruv-fann-wsp](https://www.npmjs.com/package/ruv-fann-wsp), [AgentDB](https://www.npmjs.com/package/agentdb), [RuVector](https://www.npmjs.com/package/ruvector). Optional orchestration: [MetaHarness](https://www.npmjs.com/package/metaharness) and [Ruflo](https://www.npmjs.com/package/ruflo).

These links describe integrations; package.json defines install-time dependencies. The [versioned catalog](https://github.com/ruvnet/ruv-FANN/blob/main/packages/wsp-toolkit/packages/constellation/catalog.json) declares release profiles and package versions; registry availability and publication provenance require independent verification.

<!-- constellation-links:end -->

## Discoverability topics

package catalogs · module manifests · JSON schemas · bounded JSON validation · integration contracts · package metadata · dependency-free validation · TypeScript declarations.
