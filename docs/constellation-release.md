# Protected npm publication for the ruv-fann toolkit

Status: Implemented release architecture; publication is established separately by exact-source receipts. Updated: 2026-11-11.

The `ruv-fann-toolkit-v0.2.0` release contains thirteen new npm archives: `ruv-fann@0.2.0` and twelve toolkit packages at `0.1.0`. Existing `ruv-fann-wsp@0.2.0`, `ruv-fann@0.1.2` and `ruv-fann-wasm@0.1.2` remain published dependencies. The toolkit lane does not rebuild or republish those earlier releases.

## Two separate publishers

| Lane | Scope | Authentication |
| --- | --- | --- |
| [protected-wsp-release.yml](../.github/workflows/protected-wsp-release.yml) | Existing standalone WSP release contract | Existing `NPM_TOKEN` publisher is preserved |
| [protected-toolkit-release.yml](../.github/workflows/protected-toolkit-release.yml) | Core facade plus twelve toolkit archives | Existing protected npm credential bootstraps the new packages |
| [constellation-oidc.mjs](../scripts/constellation-oidc.mjs) | Optional future configuration guard | Requires deliberately opted-in, package-specific npm trusted-publisher registration |

Current CI calls `wsp-toolkit.yml` for the core 0.2.0 release set. The older
`npm-modules.yml` pair lane remains available only by explicit manual or reusable
invocation against its 0.1.2 generation; it rejects newer package identities.

A package's catalog status, a successful local pack or possession of the helper does not establish namespace ownership, a registered trusted publisher or permission to upload. The optional OIDC helper is not the active bootstrap publisher, and this documentation does not claim a live OIDC exchange.

## Exact archives move through qualification and publication

1. Review one immutable source SHA. Qualification packs all thirteen packages once and records their identities and archive digests.
2. Install those exact archives into fresh consumers. Require actual Node.js 22/24 behavior, strict TypeScript declarations, real Chromium/WASM checks, applicable worker/workerd/AgentDB cases, source-bound security checks and zero failed, skipped or todo tests.
3. Seal the exact candidate archives and required receipts together. A passing source checkout alone is insufficient; a receipt from an older generation does not qualify the candidate.
4. Merge the unchanged qualified SHA to upstream `main`. The protected manual publisher checks the input SHA, checkout, GitHub context, current upstream main and open `release-blocker` issues before using the successful qualification artifact.
5. Revalidate the seal and publish in dependency order, with the core facade after its new dependencies. Publish the carried archive bytes without repacking. An existing registry version is accepted only if its bytes match the sealed candidate; a different archive is a release failure, not a reason to overwrite it.
6. Download every public archive, compare it with the seal and repeat fresh installed-consumer qualification. Attach archives, checksums and publication evidence to the umbrella GitHub release. Publication without successful post-install verification remains incomplete.

The release controller, not a local shell, performs privileged registry and GitHub release writes. Logs and receipts must not contain npm credentials or secret values. Do not weaken source, archive, blocker or consumer checks to get past an authentication or registry failure.

## Optional future trusted publishing

An authorized package owner may separately register an npm GitHub Actions trusted publisher for each package. The registration must match the exact owner/repository, workflow filename, any environment and allowed operation. Bootstrap and namespace requirements must be verified independently for new packages. See the official [npm trusted publisher documentation](https://docs.npmjs.com/trusted-publishers/).

The OIDC helper checks reviewed configuration, GitHub event/job/source constraints, supported toolchain and the short-lived Actions capability. A `configuration-validated` result is not an identity attestation or proof that npm accepted an OIDC upload. Source changes that opt a package into this path require review and the same artifact/public-install contract. There is no automatic fallback that silently changes a package's chosen publisher policy.

## Catalog declarations and public availability

The 24-entry constellation catalog records release planning identities and statuses. Its `published` field is declared metadata, not a registry query or namespace proof. Public observations may remain missing or stale; a missing npm downloads endpoint is not a zero-download observation. Exact source/archive/public-consumer receipts establish what this release actually published and verified.

Historical toolkit evidence remains unchanged in `packages/wsp-toolkit/docs/evidence`. Current counts, hashes and final publication outcomes belong to the current qualification and release artifacts. The separate [observatory workflow](../.github/workflows/constellation-observatory.yml) collects optional public observations and has no publication authority.
