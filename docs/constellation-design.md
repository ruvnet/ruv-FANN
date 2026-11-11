# Growth constellation implementation contract

## Outcome and acceptance

Extend the local WSP toolkit into a concrete constellation reference: a lightweight discover/plan/scaffold CLI, dependency-free shared module/contracts metadata, an explicit catalog with verified public anchors and local candidates, a standalone interactive telemetry dashboard, deterministic README cross-links, and a genuinely tokenless OIDC path for the existing protected WSP release. Preserve previous uncommitted work and package names. No registry publication, scope claim, remote repository change or scheduled collection is performed by this task.

Acceptance: a fresh packed consumer runs the onramp without pulling an ML stack; scaffolds a package in a new temporary directory; runs its built-in checks; the telemetry aggregator handles a real public cohort and an explicit local adoption sample without inventing attach rates; its HTML works in Chromium; the existing release's source/archive checks remain intact while token fallback is eliminated; all existing toolkit tests and new contract/security checks pass.

## Ownership

- Onramp worker: `packages/wsp-toolkit/packages/contracts/**` and `packages/wsp-toolkit/packages/constellation/**`, EXCEPT `packages/constellation/catalog.json` owned by coordinator. Packages `ruvnet-constellation-contracts@0.1.0` and `ruvnet-constellation@0.1.0`.
- Telemetry worker: `packages/wsp-toolkit/packages/telemetry/**` only. Package `ruvnet-constellation-telemetry@0.1.0`.
- Release worker: `.github/workflows/protected-wsp-release.yml`, `scripts/wsp-release-lane.mjs`, new `scripts/constellation-oidc.mjs`, `scripts/constellation-oidc.test.mjs`, `docs/constellation-release.md` only.
- Coordinator: catalog.json, root manifests/lockfiles, root integration/CLI/report/readme scripts, toolkit E2E/browser/type tests, other documentation/qualification workflow.

Workers own their source/declarations/README/LICENSE/package manifest/tests; no root lockfile edits. New packages ESM, src/index.js and src/index.d.ts, version0.1.0, Node20.9+ (qualification Node22.13+). Dependency-free contracts and telemetry; CLI depends only on contracts0.1.0. No lifecycle install hooks or default data collection.

## Catalog and contract

Catalog schema `ruvnet.constellation-catalog/1` has `{schema, repository:'ruvnet/ruv-FANN', packages:[...], profiles:[...]}`. Package record is `{name, role:'nucleus'|'primitive'|'satellite', family, status:'published'|'candidate', version, repository:'owner/repo', directory?:relativeRepoPath, description, contracts?:string[], links?:{npm?:httpsURL,source?:httpsURL}, proposedScope?:string}`. Only verified actual public names have status published. Candidate means local source, not a claim about npm ownership/existence. Unique names, bounded counts<=1000, safe names/paths and explicit schema IDs. No renaming published packages or adding irrelevant peer dependencies to manufacture volume.

Profile record `{id,description,packages:string[]}`; allow only known catalog package names. CLI planning returns package specs/links/availability, does not install them. Catalog lives in CLI package as `catalog.json`, coordinator will populate it. Actual public anchors include metaharness0.4.23, @metaharness/router0.4.1, @claude-flow/cli3.25.6 validated pin (registry latest observed3.75.3), ruflo3.75.3, ruvector0.3.24, agentdb3.10.0, agentic-flow3.5.4, @ruvector/attention2.3.1, @ruvector/ruvllm4.2.2, ruv-fann-wsp0.2.0, ruv-fann0.1.2, ruv-fann-wasm0.1.2. @metaharness/cli returns404 and is not listed as published.

Contracts library exports CONTRACTS exact IDs (model dense-layers/1, patch weight-patch/1, evaluation weight-evaluation/1, module ruvnet.constellation-module/1), `validateCatalog(catalog)` returning owned validated data, and `validateModuleManifest({schema:'ruvnet.constellation-module/1',name,version,role,contracts,integrations:string[]})`. Include JSON schemas for module manifest/catalog and references to canonical existing WSP formats; schema/metadata validation does not replace real WSP numerical validation. Strict plain data, bounds, unknown fields/accessors/unsafe paths refusal, snapshots. No SDK/network dependencies.

## CLI

Expose `readCatalog()` and `planProfile(catalog,id)`, `scaffoldSatellite({directory,name,description?,integrations?})`, plus bin `constellation`. Commands help/catalog/plan <profile>/scaffold <directory> --name <npmName>. Help links nucleus and core packages. No shell interpolation, provider calls, registry publication, automatic install or hidden telemetry. Scaffold to a new directory only, refuse existing path/symlink and traversal; produce standalone ESM package, tests, README crosslinks, module manifest. Use explicit optional integration metadata; do not force nucleus dependencies into every satellite. Include package quality scripts and draft protected-publication guidance. Only write files below the explicit new target. Tests cover fresh scaffold/npm test, refusal existing files, safe names and paths, no exec/telemetry, packed catalog access. Catalog import validation must work after installation.

## Telemetry interfaces

`collectPublicMetrics(catalog,{start,end,fetch?,timeoutMs?,concurrency?})` returns owned observations with schema `ruvnet.constellation-observations/1`, observedAt, start,end, packages entries `{name,status:'ok'|'unavailable'|'candidate',days:[{day,downloads}],version?,dependencies?:string[],peerDependencies?:string[],optionalDependencies?:string[],error?,repository?}`, and repositories array `{repository,status,stars?,openIssuesAndPRs?,error?}`. Fixed HTTPS hosts registry.npmjs.org, api.npmjs.org, api.github.com; validate catalog and names/dates; finite byte/deadline/concurrency caps, explicit per-source unavailable status, no token/cookie persistence, no hidden model calls. Per-package downloads use `/downloads/range/YYYY-MM-DD:YYYY-MM-DD/<encoded package>`. Repo metadata `open_issues_count` includes PRs and must be labeled correctly. Query only published entries; candidate stats unknown, never zero-filled.

`analyzeConstellation(catalog,observations,{adoption?}?)` returns report schema `ruvnet.constellation-report/1`, observedAt,start,end, packages rows, totals, correlations/weekday summaries if data permits, repositories, attach. Package row has name,role,family,status, latestDayDownloads,last7Downloads,last30Downloads,weekdayBaseline,weekdayDeltaPercent (nullable when incomplete/zero denominator) plus actual declared dependency links. Do not conflate transitive downloads with independent users or sum overlapping anchor cohorts as unique adoption.

Optional adoption schema `{schema:'ruvnet.constellation-adoption/1',consent:true,samples:[{sampleId:64hex,observedAt:ISO8601,packages:string[]}]}`. Process explicit local data only; deduplicate by sampleId and latest observation, conflicting duplicates reject. Attach is a percentage of observed consenting project snapshots containing a nucleus and a primitive, never users. Without adoption input return attach.status:'unavailable' and null percentages. Returned reports must not contain sample IDs/raw snapshots. No auto scanning or upload. Include coverage/denominators and metric definitions.

`renderConstellationDashboard(report)` returns self-contained HTML with family/role/search/period controls, package table/trend visualization, repository issue signal, candidate/unavailable states, source/coverage labels and attach explanation. Use no external JS/fonts/trackers, escape all untrusted fields/embedded JSON, textContent for dynamic data, safe links only. Browser errors/XSS regression and controls must be tested with actual Chromium by coordinator. A locally generated dashboard is the deliverable; no hosting operation is implicit.

## Release changes

Official npm docs currently require npm>=11.5.1 and Node>=22.14.0 for trusted publishing. Existing workflow grants id-token:write but injects NPM_TOKEN, and existing WSP publisher refuses missing NODE_AUTH_TOKEN. Remove token requirement/fallback for this protected lane, keep exact-main/source/archive/provenance checks, add explicit runtime/OIDC-context checks and pin supported Node/npm versions. No live publish or trusted-publisher configuration change. Record npm per-package trusted publisher owner/repository/workflow/environment setup as external prerequisite; inability to verify it is not success. Do not assume scoped names establish enterprise trust or replace dependency review.

## Evidence boundaries

The user-provided cohort/download figures are hypotheses until their source cohort/window is supplied. Start with verified repo packages and public anchors; support a supplied full catalog later. Aggregate npm download series measure downloads, including CI/transitive activity. Synchronization is correlation, not proof of attach or causal cross-pollination. Shared standards support compatibility; exact numerical evaluation/hash/revision authority stays with the existing WSP/AgentDB components. Historical validation receipts remain historical; create separate constellation receipts.
