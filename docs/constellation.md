# Growth constellation reference

This checkout now implements the operational pieces of the constellation strategy: a small discover/plan/scaffold entry point, common module metadata, explicit optional integrations, consistent documentation links, public cohort reporting and a protected tokenless WSP publication path. It extends the existing WSP swarm loop without renaming its packages or adding orchestration dependencies to every primitive.

## Start here

```sh
cd packages/wsp-toolkit
npm ci --ignore-scripts --no-audit --no-fund
npm run constellation -- help
npm run constellation -- plan weights
npm run constellation -- plan local-loop
```

The `weights` profile returns the verified published pin `ruv-fann-wsp@0.2.0`. The `local-loop` profile returns candidate source locations and explicitly declines to represent those candidates as public npm install specs. Planning installs nothing and sends no telemetry. The CLI depends only on the dependency-free contracts package; the fresh-consumer test verifies that installing the onramp installs exactly those two packages and no ML stack.

Create a local, independently useful satellite template in a new directory:

```sh
npm run constellation -- scaffold /tmp/my-constellation-adapter --name my-constellation-adapter
cd /tmp/my-constellation-adapter
npm test
npm run benchmark
npm run pack:check
```

The scaffold is a bounded JSON-copy adapter template with independent validation cases. It does not generate arbitrary domain logic or prove domain accuracy. It adds module metadata, tests, documentation and publication guidance, with no forced nucleus dependency. Scaffolding uses held directory handles on Linux; catalog/planning remain portable Node APIs. The CLI refuses unsupported secure-scaffold routes instead of silently falling back to pathname-only writes. Concurrent same-user code is not a hardened isolation boundary.

## Catalog and scopes

The authoritative [catalog](../packages/wsp-toolkit/packages/constellation/catalog.json) contains 24 entries: 12 verified public anchors/specialists and 12 local package candidates. It records roles, families, exact planning pins, repositories, source directories, existing shared formats and four profiles. Collection can accept another validated catalog of up to 1,000 entries, so a full cohort can be supplied without changing code.

The supplied 405-package cohort and its original measurement windows were not present in this checkout. The default catalog therefore does not claim to cover that cohort or reproduce its aggregate figures. The public `metaharness` entry exists; `@metaharness/cli` returned npm404 and is not offered as a published package. GitHub's former `ruvnet/claude-flow` repository resolves to `ruvnet/ruflo`, which the catalog uses. The validated Ruflo CLI planning pin remains 3.25.6 even when registry metadata observes a newer version.

`proposedScope` values are migration proposals, not assertions of namespace ownership. Existing published identities remain compatible. Before any scoped migration, verify account ownership, choose a version/alias/deprecation plan, preserve downstream imports and separately configure the new package's trusted publisher. A scope is an organizational label; provenance, dependency review and runtime behavior establish the relevant security evidence.

Every local candidate includes a `constellation.json` module manifest. The new contracts package publishes catalog/module JSON schemas and identifiers for the existing WSP model, patch, evaluation and related formats. These metadata contracts do not replace WSP's bounded numerical validation, SHA-256 model checks or AgentDB revision authority. Actual core packages retain their existing dependency requirements; this work does not claim that AgentDB, RuVector or WSP are dependency-free.

## README routing and optional integration

```sh
cd packages/wsp-toolkit
npm run constellation:links
npm run constellation:check
```

The generator writes marked, deterministic cross-links and module manifests for the 12 workspace candidates. Qualification checks for drift. Cross-links point to the onramp, shared standards, relevant primitives and optional orchestration. Integration metadata derives from actual declared dependencies/peers; the generator adds no runtime dependencies. This provides discoverability without manufacturing package downloads through unrelated peer dependencies.

## Public telemetry dashboard

```sh
cd packages/wsp-toolkit
npm run constellation:report
# For inherited proxy settings on managed Node24:
NODE_USE_ENV_PROXY=1 npm run constellation:report
```

The command writes `.artifacts/constellation/{observations.json,report.json,dashboard.html}`. Open the HTML locally; it has no external scripts, fonts, trackers or runtime network requests. Family/role/search/window controls and per-package trends operate entirely on the embedded report. Collection is an explicit read of public npm/GitHub endpoints, with fixed hosts, bounded responses, a finite deadline and concurrency limits.

Collection defaults to 42 complete UTC days ending yesterday. Download totals require complete coverage; missing packages/days remain unknown. The live reference collected through 2026-11-10 has nine measured download series, three FANN download endpoints returning404, 12 source candidates and six observed repositories. All 12 public package metadata requests succeeded. Registry metadata confirms those FANN packages exist; an unavailable downloads endpoint is not evidence of zero downloads. The report retains this distinction.

Repository counts use GitHub's `open_issues_count`, which includes open pull requests. Declared dependency links describe manifests, not measured co-installation. Weekday comparisons use the preceding four matching weekdays; correlations are descriptive and cannot establish causal cross-pollination or distinct users. The manually dispatched [observatory workflow](../.github/workflows/constellation-observatory.yml) uploads report artifacts only. No recurring job, public dashboard hosting or remote configuration has been enabled here.

## Measured attach requires a separate input

Without explicit joint-adoption data, attach is unavailable. Aggregate downloads and correlations cannot supply its denominator. To analyze a consenting local sample:

```sh
npm run constellation:report -- --adoption /path/to/consenting-project-snapshots.json
```

The input schema is `ruvnet.constellation-adoption/1`, with `consent:true` and samples containing a pseudonymous lowercase SHA-256 `sampleId`, an ISO UTC `observedAt`, and package names. Keep raw project identifiers out of this file. Consent is the caller's assertion, not a cryptographic proof or permission grant.

The analyzer deduplicates snapshots by project and timestamp, rejects contradictory duplicates even out of order, and keeps the latest valid snapshot per project in the window. The denominator is sampled projects containing any nucleus; the numerator contains a primitive as well. It reports overall/per-primitive attach with explicit denominators and the unit `consenting-project-snapshots`. This is a sample statistic, not a percentage of npm users or developers. Sample IDs and raw snapshots stay out of reports and network requests. No workspace scanning or automatic collection is added.

## Trusted publishing

The existing WSP protected lane previously granted OIDC permissions while still requiring `NPM_TOKEN`. The revised path pins supported Node/npm, rejects token fallback, retains exact-main/sealed-archive/provenance checks and publishes from isolated auth-free npm configuration. A registration matching the exact package/owner/repository/workflow/environment is still required at npm. No registration, scope claim, initial publication, live OIDC exchange or package upload was performed. See [release setup and boundaries](constellation-release.md).

## Qualification

```sh
cd packages/wsp-toolkit
npm run setup:agentdb
npx --no-install playwright install --with-deps chromium
npm run validate
```

Qualification includes the existing real-WASM/swarm/AgentDB scenarios, all new contracts/onramp/telemetry tests, the root OIDC-context/release-evidence tests, strict types, deterministic documentation metadata, standalone browser controls and a fresh consumer of all12 archives. A separate consumer proves the two-package onramp footprint. Security regressions cover conflicting adoption snapshots, embedded HTML/script injection, report-output symlink replacement and concurrent scaffold directory substitution.

Historical WSP/loop receipts remain unchanged. New source-bound qualification receipts are stored as `constellation-node-22-validation.json` and `constellation-node-24-validation.json` under the toolkit's `docs/evidence` directory. Live observations and summary evidence are separate from synthetic test fixtures. Successful local qualification does not assert that GitHub CI, npm trusted-publisher registration or hosted publication has executed.
