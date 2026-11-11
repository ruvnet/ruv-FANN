# ruvnet-constellation-telemetry — public package observations and local dashboards

Collect public package observations only when invoked, then render a standalone HTML dashboard. Joint adoption needs a separate consenting project sample; download correlations cannot establish an attach rate.

**Topics:** npm download observations · GitHub activity · consenting adoption samples · offline dashboard.

```sh
npm install ruvnet-constellation-telemetry@0.1.0
```

[Package on npm](https://www.npmjs.com/package/ruvnet-constellation-telemetry) · [Core 0.2.0 entrypoint: `ruv-fann/telemetry`](https://github.com/ruvnet/ruv-FANN/tree/main/packages/core#capability-entrypoints) · [Interactive ruv-fann field guide](https://ruvnet.github.io/ruv-FANN/) · [Toolkit API and release guide](https://github.com/ruvnet/ruv-FANN/tree/main/packages/wsp-toolkit#readme)

Choose this standalone package for its declared dependency footprint, or use `ruv-fann/telemetry` from the complete `ruv-fann@0.2.0` facade. Package version: **0.1.0**. [Umbrella release and evidence](https://github.com/ruvnet/ruv-FANN/releases/tag/ruv-fann-toolkit-v0.2.0).

Dependency-free public package observations, explicitly sampled project attach metrics, and a standalone interactive dashboard. ESM; Node 20.9+; no install hooks, SDKs, provider calls, hidden collection or uploads.

Integration sketch: supply a validated constellation `catalog`; the call performs explicit public network reads.

```js
import { collectPublicMetrics, analyzeConstellation, renderConstellationDashboard }
  from 'ruvnet-constellation-telemetry';
const observations = await collectPublicMetrics(catalog, {
  start: '2026-09-30', end: '2026-11-10', timeoutMs: 20000, concurrency: 4
});
const report = analyzeConstellation(catalog, observations);
const html = renderConstellationDashboard(report);
```

`collectPublicMetrics` reads only fixed HTTPS endpoints on `api.npmjs.org`, `registry.npmjs.org`, and `api.github.com`. Package and repository names are validated before URL construction. Redirects, credentials and cookies are excluded. Each response is limited to 2 MiB, concurrency to eight tasks, date windows to 366 completed UTC days, and the whole collection to at most 60 seconds. Its injected fetch capability must honor AbortSignal for transport cleanup. Failures are explicit observations; a failed metadata fetch does not discard independently fetched download data. Candidate packages are never queried. GitHub's `open_issues_count` is labeled **open issues + pull requests**.

`analyzeConstellation` returns owned data, complete-window daily/7-day/30-day counts, coverage, the previous four matching weekdays, declared registry dependency links and bounded Pearson download correlations. Missing points and candidate observations remain unknown; incomplete cohort totals are null. Observed partial sums are separately labeled. Correlations examine at most 32 measured packages and retain ten pairs with at least seven shared days. Correlated download counts, dependency links and transitive installs do not establish distinct users, causality or attach rates.

Attach is available only with an explicit local adoption input:

```js
const report = analyzeConstellation(catalog, observations, {
  adoption: {
    schema: 'ruvnet.constellation-adoption/1', consent: true,
    samples: [{sampleId: projectHash, observedAt: '2026-11-10T12:00:00Z', packages: ['metaharness', 'agentdb']}]
  }
});
```

The library does not generate identifiers, inspect project files or send these samples anywhere. It keeps the latest snapshot per caller-supplied SHA-256 project identifier within the report window. Conflicting snapshots at the same instant reject. The denominator is **observed consenting project snapshots containing any catalog nucleus**; per-primitive and any-primitive percentages use that same denominator. This is not a percentage of developers or npm users. Sample IDs and raw snapshots never enter the returned report. No supplied sample, or no qualifying denominator, means `attach.status: 'unavailable'` and null percentages.

`renderConstellationDashboard` returns a portable HTML artifact with no external scripts, fonts, trackers or network fetches. Controls: `#period-filter` (1/7/30), `#family-filter`, `#role-filter`, `#package-search`, and `#trend-package`. Package rows in `#package-table` carry `data-package`; the chart is `#trend-chart`. `#attach-rate`, `#download-total`, and `#coverage-text` expose summary values. Dynamic text uses textContent, embedded JSON escapes HTML delimiters, and links allow only HTTPS npm/GitHub targets. The page supports local report export.

Public metrics are point-in-time observations. Users must supply an accurate catalog and an appropriately consented, representative adoption sample; the library cannot prove consent or correct sample selection. No claim is made that scoped names establish security or that download volume measures unique adoption.

<!-- constellation-links:start -->

Explore the [ruv-fann field guide](https://ruvnet.github.io/ruv-FANN/), [constellation onramp](https://www.npmjs.com/package/ruvnet-constellation) and [shared contracts](https://www.npmjs.com/package/ruvnet-constellation-contracts). Optional ecosystem primitives: [ruv-fann-wsp](https://www.npmjs.com/package/ruv-fann-wsp), [AgentDB](https://www.npmjs.com/package/agentdb), [RuVector](https://www.npmjs.com/package/ruvector). Optional orchestration: [MetaHarness](https://www.npmjs.com/package/metaharness) and [Ruflo](https://www.npmjs.com/package/ruflo).

These links describe integrations; package.json defines install-time dependencies. The [versioned catalog](https://github.com/ruvnet/ruv-FANN/blob/main/packages/wsp-toolkit/packages/constellation/catalog.json) declares release profiles and package versions; registry availability and publication provenance require independent verification.

<!-- constellation-links:end -->
