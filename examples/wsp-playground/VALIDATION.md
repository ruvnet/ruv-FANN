# WSP Live Lab validation

Date: 2026-11-12

Preview: https://wsp-live-lab.cognitum-consulting-mail.workers.dev

Review: https://github.com/ruvnet/ruv-FANN/pull/230

## Executed evidence

`node --test test/engine.test.mjs`: seven tests passed on Node 24.22.1.
Both scalar and SIMD backends reject the synthetic shortcut, pass the repair,
enforce stale revision checks and restore the exact starting hash. Additional
tests exercise bounded search, arithmetic benchmark oracles, request sizes,
malformed UTF-8, method/content types and caller policy injection.

`npm audit --json`: zero advisories across 39 dependencies including development
dependencies, as observed during this session. This is a dated registry result.

The actual deployed Worker loaded successfully, with static WASM modules,
Node compatibility disabled, 100 ms CPU cap and zero reported startup time.
Live browser operations on the connected Brave browser confirmed:

| Behavior | Observed result |
| --- | --- |
| Browser runtime | WASM-SIMD ONLINE |
| Robot shortcut [1, 0] | Development MSE 0; separate MSE 0.272; 55 finite-grid violations; promotion disabled |
| Robot repair [1, -1] | Browser and edge agree; all gates pass |
| Explicit promotion | Revision 2, hash `c69c0190e40913fc4348ba019b1d694a785e970f94d8450d8c083669aa4b71cd` |
| Rollback | Revision 3, restored hash `03b14212bc91c32aefa02c640c93a53db09ce084e35aad796c50bec5822ab087` |
| Ambient shortcut [1, 1] | Rejected; promotion disabled |
| Ambient repair [1, 0] | Browser and edge agree; all gates pass |
| Edit after passing gate | Returns to UNEVALUATED; promotion disabled |
| Bounded robot search | Six screens; development MSE 0; separate gates reject |
| Mobile layout | At viewport 390 by 844, document width is 390; no horizontal overflow |
| Scalar benchmark | 0.0200 ms median / 0.0400 ms p95 per 256-row batch, zero reference error |
| SIMD benchmark | 0.0120 ms median / 0.0300 ms p95 per 256-row batch, zero reference error |
| Browser console | No warnings or errors at the observed check |

Timings are one browser/device observation, not a portable performance claim.
The robots and room are illustrations driven by model outputs; these are not
robotics or WiFi accuracy benchmarks.

## Environment limits and reproducibility

The local automated Chromium test could not launch because the managed process
environment denies the Unix socket used by Chromium's singleton mechanism.
The test is retained and a GitHub Actions workflow installs Chromium and runs
the complete desktop/mobile acceptance suite. The dedicated workflow passed
for commit `28e443ad1ea2914f0f0adcf8a4bda0cb73793ebc`:
https://github.com/ruvnet/ruv-FANN/actions/runs/43274468988

That successful automated suite also verifies receipt downloads, bounded search,
edit invalidation and simulated edge network failure blocking promotion. Eight
tests passed in CI: the browser acceptance test and seven numerical/API tests.
The initial artifact upload omitted the hidden output directory; the workflow
now explicitly includes hidden files. Local socket restrictions remain a local
environment limitation, separate from the successful hosted browser test.

## Threat review and residual limits

The public endpoint accepts only known scenario names and two pairs of finite
weights in [-2,2]. It refuses arbitrary models, datasets, policies and URLs;
streams at most 1024 bytes and caps concurrent evaluation per isolate. Source
review checked its method, content type, encoding and input boundaries. There
is no credential, model upload, shell, outbound fetch, persistence or edge
promotion endpoint. CSP confines application code and connections to this origin;
font resources are the only external presentation dependency.

The per-isolate concurrency cap is not global abuse protection. Monitor account
usage and add account-level rate rules if needed. Every asset is currently
served by Worker code and counts as a Worker invocation. Static Assets is the
cost optimization path for a higher-traffic deployment.

All test fixtures are public and finite. Promotion affects only the current
browser session. Passing cases does not establish behavior on unseen inputs.
