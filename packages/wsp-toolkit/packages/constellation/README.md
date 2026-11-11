# ruvnet-constellation

A lightweight explicit onramp for a constellation of independently useful packages. ESM, Node 20.9+. Its only runtime dependency is `ruvnet-constellation-contracts@0.1.0`; no ML runtime, model, provider or telemetry collector is pulled in by this package. This source candidate has not been published.

```sh
constellation help
constellation catalog
constellation plan weights
constellation plan local-loop
constellation scaffold ./new-json-adapter --name @your-scope/json-adapter --integration ruv-fann-wsp
```

`readCatalog(path?)` reads a bounded local JSON file; by default it uses the packaged `catalog.json`. `planProfile(catalog,id)` validates and owns the catalog, and returns `registrySpecs` for declared published packages, `localCandidates` separately, and `publicInstallAvailable:false` if any profile member is only a candidate. Planning executes no install command and never substitutes candidates for public packages. Catalog status and proposed scopes are metadata, not namespace ownership proofs.

`scaffoldSatellite({directory,name,description?,integrations?})` creates a new explicit directory containing a bounded identity JSON adapter, TypeScript declaration, module manifest, independent malformed-input/snapshot tests, a small local benchmark and publication-preparation guidance. It refuses existing targets, symlinks, traversal and unknown/executable options. Parents must already exist and consist of real directories. Scaffolding requires Linux with `O_DIRECTORY`, `O_NOFOLLOW` and `/proc/self/fd`; it fails closed on unsupported platforms, while catalog and planning remain portable. It opens every parent/child directory through a held directory descriptor, anchors exclusive leaf writes to those descriptors, and verifies directory inode relationships before writes and completion. Child-directory substitution by symlinks cannot redirect output into an existing outside path. It performs no recursive cleanup on failure. These controls do not make a hardened sandbox against arbitrary same-UID code, privileged mount changes, or an attacker moving entire owned directory trees. An I/O or detected namespace-change failure can leave an inspectable partial candidate rather than risk deleting unrelated paths.

The scaffold is a template, not an AI-generated domain solution. Its default checks do not establish WSP correctness, task accuracy or benchmark superiority. Add useful domain mapping and independent cases before publication. Optional integrations are metadata; no irrelevant peer dependencies or nucleus installs are added to inflate dependency volume.

All commands are local unless the user separately runs a package manager or provider. No automatic install, provider request, background telemetry, registry publication, scope claim or credential configuration occurs. `--json` is accepted for catalog/plan/scaffold; data output is JSON by default. CLI usage/input failures exit 2.

<!-- constellation-links:start -->

Explore the [constellation onramp](README.md) and [shared contracts](../contracts/README.md). Optional ecosystem primitives: [ruv-fann-wsp](https://www.npmjs.com/package/ruv-fann-wsp), [AgentDB](https://www.npmjs.com/package/agentdb), [RuVector](https://www.npmjs.com/package/ruvector). Optional orchestration: [MetaHarness](https://www.npmjs.com/package/metaharness) and [Ruflo](https://www.npmjs.com/package/ruflo).

These links describe integrations; they do not add install-time dependencies. Workspace package candidates and namespace proposals are listed explicitly in the catalog.

<!-- constellation-links:end -->
