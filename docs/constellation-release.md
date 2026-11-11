# Tokenless trusted publishing for constellation packages

The protected WSP release now uses npm's GitHub OIDC path without a long-lived npm token fallback. This changes repository code only. No npm name, scope, publisher registration, GitHub environment or release was created or changed, and no live publish was attempted.

## Existing WSP lane

The publication entry point remains [protected-wsp-release.yml](../.github/workflows/protected-wsp-release.yml), with workflow name `protected-release`, job `publish`, a manual exact-SHA input and the upstream main-branch restriction. Its toolchain is pinned to Node `24.22.1` and npm `11.9.0`. npm is installed into a runner-temporary prefix without lifecycle scripts; the release PATH selects that explicit CLI. Official npm documentation currently requires npm `11.5.1+` and Node `22.14.0+` for trusted publishing.

The protected script retains the existing checks:

- The checkout, reviewed input, GitHub SHA, workflow ref and current upstream main must agree.
- Open `release-blocker` issues prevent publication.
- Only the successful qualification run's exact artifact is downloaded.
- The source-bound seal, tarball SHA-256/SHA-512, Node 22/24 receipts, browser proof, mutation checks, benchmark and dependency audit are revalidated.
- npm receives the exact verified archive bytes, copied into a private temporary publication directory without repacking, plus `--ignore-scripts --access public --provenance` and the fixed public registry.
- Public bytes are downloaded and compared; an already published version is accepted only when its bytes match. Public installation and GitHub release-asset checks still run afterward.

[scripts/constellation-oidc.mjs](../scripts/constellation-oidc.mjs) adds a reusable, side-effect-free configuration guard. It requires the expected repository, workflow filename/name, branch, job, dispatch event, reviewed SHA and matching checkout; supported stable Node/npm versions; and the GitHub Actions OIDC request URL/capability. A fork, PR event, different job, missing OIDC permission, stale checkout or npm credential fallback fails before npm can publish. Error messages never include credential values.

Environment strings are not an identity attestation. The helper reports `configuration-validated`, and npm must still obtain the actual GitHub token and match its claims against the package's registered trusted publisher. An identical-existing-version receipt can succeed without issuing any new OIDC upload. The receipt records `oidc-only` policy and whether it verified existing bytes or published/recovered an upload; it does not invent an OIDC exchange proof.

The child npm process runs outside the checkout with controlled project, user and global `.npmrc` files and a fresh cache. Its environment omits npm auth values and inherited GitHub API tokens while retaining the short-lived Actions OIDC capability, proxy configuration and CA trust. The parent still uses the job's GitHub token for read-only source/blocker/artifact checks and the later release step. Auth files and environment objects are never written to receipts or logs.

## Required external npm setup

An authorized package owner must configure and verify the package's trusted-publisher relationship on npm before this lane can make a new upload. This task did not inspect or assert that registration.

For the current WSP package, the registration must match:

| npm field | Expected value |
| --- | --- |
| CI/CD provider | GitHub Actions on a GitHub-hosted runner |
| Organization or user | `ruvnet` |
| Repository | `ruv-FANN` |
| Workflow filename | `protected-wsp-release.yml` (filename only) |
| Environment name | None is declared by this workflow; any configured environment must agree with a deliberately reviewed job configuration |
| Allowed actions | Direct `npm publish` must be permitted; permission for staged publishing alone does not enable this command |

The package's `repository.url` must match the upstream GitHub repository. Public-package provenance also requires a public repository. Names, scopes, ownership, access policy and initial-publication/bootstrap requirements for new satellites must be confirmed independently; local candidate manifests and scope suggestions do not establish npm ownership. Do not add a token fallback to bypass missing trusted-publisher setup. A registry authentication failure is an incomplete external setup result, not a successful release.

Official reference: [npm trusted publishers](https://docs.npmjs.com/trusted-publishers/), inspected 2026-11-11. npm does not validate a saved publisher configuration automatically; its precise values and the first actual authorized publish require verification.

## Applying the standard to satellites

Keep independent package names and useful optional integrations. Scope branding alone does not establish security-scanner trust, ownership, a clean dependency tree or adoption.

For each reviewed release:

1. Establish package-name/scope ownership and exact repository metadata.
2. Run qualification on the exact reviewed source. Test the actual packed archive, declared integrations and fresh public consumer behavior.
3. Store the archive and evidence hashes in a source-bound seal. Carry those exact bytes through publication; do not rebuild during the privileged job.
4. Configure the package-specific npm trusted publisher to the reviewed repository/workflow/environment and permitted operation.
5. Use GitHub-hosted publication with minimal permissions, `id-token: write`, supported pinned Node/npm versions and tokenless isolated npm configuration.
6. Validate identity/context independently of the artifact checks, publish with provenance, and verify public bytes and consumers. Preserve partial/failure results rather than treating an existing version or coordination record as proof of a new upload.

The helper can receive an explicit reviewed policy for another workflow. Changing the policy is a source change requiring the same review as the workflow; it is not a remote capability discovery or authorization mechanism. The existing WSP source/archive invariants are not automatically inherited by another package merely because it imports the helper.

## Local validation

```sh
node --test scripts/constellation-oidc.test.mjs scripts/wsp-release-evidence.test.mjs
```

Tests cover minimum/unsupported toolchains, fork/PR/job/event/source mismatches, missing or external OIDC endpoints, credential fallback spellings, getters and malformed data, child-environment isolation, and actual publication-entry rejection before any registry or GitHub operation. They also verify that the protected workflow retains OIDC permissions, explicit tool pins and source/archive checks. These are configuration and failure-path tests; no live OIDC exchange or publication is claimed.
