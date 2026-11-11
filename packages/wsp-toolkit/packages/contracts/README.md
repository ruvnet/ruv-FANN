# ruvnet-constellation-contracts

Dependency-free ESM metadata contracts for the constellation catalog and satellite module manifests. Node 20.9+ or browsers with `TextEncoder`/`URL`. This candidate has not been published.

`validateCatalog(value)` and `validateModuleManifest(value)` return owned, validated JSON data. They reject unknown fields, accessors, sparse arrays, duplicate identifiers, unsafe npm names/paths/URLs, unversioned schema IDs and excessive data. A catalog can contain up to 1,000 packages and 64 profiles within a 2 MiB canonical byte budget. `published` is caller-supplied status; validation does not prove namespace ownership or check a registry.

`CONTRACTS` names the existing WSP dense model, patch, evaluation, policy, dataset and finite-domain formats alongside the catalog/module metadata schemas. Versioned custom schema IDs are permitted; listing a contract does not establish behavioral compatibility. Real WSP validation, model hashing, evaluation and AgentDB revision authority remain separate.

JSON Schema documents are exported under `ruvnet-constellation-contracts/schemas/`. `module.schema.json` and `catalog.schema.json` describe metadata structure; JavaScript additionally enforces cross-field references, aggregate byte budgets and safe own data properties. `wsp-references.schema.json` records canonical schema references without reimplementing numerical/model limits.

No SDK, native driver, provider, telemetry collector, network client or install hook is included.

<!-- constellation-links:start -->

Explore the [constellation onramp](../constellation/README.md) and [shared contracts](README.md). Optional ecosystem primitives: [ruv-fann-wsp](https://www.npmjs.com/package/ruv-fann-wsp), [AgentDB](https://www.npmjs.com/package/agentdb), [RuVector](https://www.npmjs.com/package/ruvector). Optional orchestration: [MetaHarness](https://www.npmjs.com/package/metaharness) and [Ruflo](https://www.npmjs.com/package/ruflo).

These links describe integrations; they do not add install-time dependencies. Workspace package candidates and namespace proposals are listed explicitly in the catalog.

<!-- constellation-links:end -->
