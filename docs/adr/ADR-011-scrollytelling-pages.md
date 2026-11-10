# ADR-011: An illustrated, locally executed neural-network field guide

Status: Implemented
Date: 2026-11-10
Related: ADR-009, ADR-010

## Context

The repository spans several independent workspaces. Package descriptions and
API tables alone do not explain how a prediction, a weight edit, an evaluation,
and a promotion differ. Readers need a plain-language introduction, a real
experiment, and links to the exact published packages and evidence.

## Decision

Publish a static GitHub Pages field guide at
https://ruvnet.github.io/ruv-FANN/. Use original SVG illustrations, locally hosted
OFL fonts, native scrolling, a sticky diagram that follows four chapters, a
pause control, and reduced-motion support. The full narrative, package guide,
benchmark table and examples remain readable without JavaScript. Link the
README's animated SVG cover directly to the guide. Its short animation runs
once; the site provides explicit control over its continuous signal animation.

Run the actual published `ruv-fann-wsp@0.2.0`, `ruv-fann@0.1.2` and
`ruv-fann-wasm@0.1.2` in a bounded four-row XOR demonstration after an explicit
Start click. Vendor the public archives after checking pinned SHA-512 integrity.
Serve every runtime asset from the Pages origin under a strict CSP. There are no
provider calls, analytics or remote fonts. A candidate preview never changes the
active model. Evaluation, promotion, invalidation and rollback remain separate
operations, with visible outputs, error and revisions.

Explain inspection, layer traces, outgoing-neuron ablation, finite-difference
sensitivity, patches and workspaces. Preserve both Iris and Wine benchmark
results, all compared baselines, train/validation/test boundaries, and negative
results. Separate published npm versions from the other repository families.
Include Rust, AgentDB, Candle and forecasting entry points without implying that
every subsystem shares the same execution backend or release version.

Use one exact-source deployment artifact. A read-only PR workflow builds the
site, runs real Chromium and axe checks plus hostile manifest/path tests, and
seals the deterministic archive with source, manifest and evidence hashes. An
explicit protected dispatch requires the unchanged qualified SHA on current
main and zero open release-blocker issues. It verifies and deploys that archive,
then downloads every public file and reruns browser checks on the public URL.
The workflow never publishes npm packages. Ignore generated outputs locally,
but reject ignored or untracked files in shipped site source when sealing.

## Acceptance criteria

1. Four chapters update the diagram during native scrolling, without scroll
   interception. Decorative animation can be paused and honors reduced motion.
2. Actual public WASM demonstrates a rejected edit, an accepted preview, explicit
   promotion, stale-receipt invalidation, rollback and reset.
3. Desktop, 390px and 360px layouts have no page overflow; keyboard/copy controls,
   no-JavaScript readability, CSP and same-origin runtime requests pass. Axe
   WCAG 2/2.1 A/AA reports zero violations at desktop and mobile widths.
4. Unsafe paths, symlinks, unexpected files, wrong source and modified bytes fail
   verification. Both browser and security receipts are source/manifest bound.
5. README cover and text links open the public guide; protected publication
   verifies the exact candidate bytes and behavior at the public URL.

## Limits

Automated accessibility checks do not replace usability testing with assistive
technology. Browser qualification covers Chromium, not every engine or device.
The four-row example teaches lifecycle behavior; it does not demonstrate learned
generalization or agent-task improvement. Benchmark confidence intervals overlap
and are descriptive across fixed splits. No state-of-the-art, physical-GPU
throughput, LLM-optimizer or automatic-online-learning claim is made. Numerical
sensitivity is a finite-difference diagnostic, not autograd. A model fingerprint
binds content and does not grant promotion authority.
