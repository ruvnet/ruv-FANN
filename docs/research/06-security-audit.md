# 06 — Security Audit (ruv-FANN)

**Date:** 2026-08-09
**Scope:** root Rust workspace, `ruv-swarm` Rust workspace (via `Cargo.lock`), `ruv-swarm/npm` package, `neuro-divergent` (build check only), tracked-file secrets scan, unsafe-code inventory, CI gating.
**Method:** cargo-deny 0.19.2 (`check advisories`, RustSec advisory-db snapshot cloned 2026-08-09), cargo-audit 0.22.0 (offline, same DB snapshot), `npm audit` (live registry), pattern-based grep over `git ls-files`.
**Provenance:** everything below marked *verified locally* was reproduced in this session; items marked *needs CI run* have not been exercised on GitHub Actions yet.

---

## 1. Rust advisories — root workspace (`/home/user/ruv-FANN`)

Verified locally. Two states matter because a sibling dependency-modernization task updated `Cargo.lock` during this session:

### 1.1 Against the pre-update lockfile (state at session start)

| ID | Crate (version) | Type / Severity | Fix | Status |
|---|---|---|---|---|
| RUSTSEC-2026-0204 | crossbeam-epoch 0.9.18 | vulnerability (invalid pointer deref in `fmt::Pointer`) | upgrade ≥0.9.20 | **fixed in working tree** (now 0.9.20) |
| RUSTSEC-2026-0190 | anyhow 1.0.98 (dev-dep) | unsound (`Error::downcast_mut` UB) | upgrade ≥1.0.103 | **fixed in working tree** (now 1.0.104) |
| RUSTSEC-2025-0141 | bincode 1.3.3 | unmaintained | no safe upgrade (1.x frozen) | accepted risk, documented ignore in `deny.toml` |
| RUSTSEC-2026-0097 | rand 0.9.x (dev-dep) | unsound (custom logger) | via proptest ≥1.11 | no longer in lockfile |

### 1.2 Against the current working-tree lockfile

`cargo deny check advisories` with the repo's `deny.toml` **passes** ("advisories ok") — verified locally with the updated `Cargo.lock`. Two ignore entries in `deny.toml` are now stale and emit `advisory-not-detected` warnings (`RUSTSEC-2024-0436` paste, `RUSTSEC-2026-0097` rand). They should be pruned **after** the sibling's lockfile update is committed; left untouched here to avoid breaking the pre-update state.

## 2. Rust advisories — ruv-swarm workspace

The workspace itself **cannot currently be resolved by cargo** (verified locally): `ruv-swarm-wasm` requires `ruv-fann = "^0.1.5"` via path dep, but the root crate is `0.2.1`, so `cargo metadata` fails. This also means no cargo-deny/CI can gate this workspace until the version requirement is fixed. The audit below was therefore run directly against `ruv-swarm/Cargo.lock` with cargo-audit (offline DB): **17 vulnerabilities** (verified locally).

| ID | Crate (version) | Severity | Title | Fix |
|---|---|---|---|---|
| RUSTSEC-2026-0189 | rmcp 0.2.1 | 8.8 high | DNS rebinding in Streamable HTTP server transport | ≥1.4.0 |
| RUSTSEC-2026-0037 | quinn-proto 0.11.12 | 8.7 high | DoS in Quinn endpoints | ≥0.11.14 |
| RUSTSEC-2026-0185 | quinn-proto 0.11.12 | 7.5 high | Remote memory exhaustion (out-of-order stream reassembly) | ≥0.11.15 |
| RUSTSEC-2026-0009 | time 0.3.41 | 6.8 medium | DoS via stack exhaustion | ≥0.3.47 |
| RUSTSEC-2023-0071 | rsa 0.9.8 | 5.9 medium | Marvin attack (timing sidechannel key recovery) | **no fix available** (transitive via sqlx) |
| RUSTSEC-2026-0098/0099/0104 | rustls-webpki 0.101.7 **and** 0.103.3 (two copies) | advisory (no CVSS) | wildcard/URI name-constraint bypass; CRL-parsing panic | ≥0.103.12 / ≥0.103.13 |
| RUSTSEC-2026-0049 | rustls-webpki 0.103.3 | advisory | CRL distribution-point matching flaw | ≥0.103.10 |
| RUSTSEC-2026-0007 | bytes 1.10.1 | advisory | integer overflow in `BytesMut::reserve` | ≥1.11.1 |
| RUSTSEC-2026-0204 | crossbeam-epoch 0.9.18 | advisory | same as root finding | ≥0.9.20 |
| RUSTSEC-2025-0047 | slab 0.4.10 | advisory | OOB access in `get_disjoint_mut` | ≥0.4.11 |
| RUSTSEC-2024-0363 | sqlx 0.7.2 | advisory | binary protocol misinterpretation (overflow casts) | ≥0.8.1 |
| RUSTSEC-2025-0055 | tracing-subscriber 0.3.19 | advisory | ANSI-escape log poisoning from user input | ≥0.3.20 |

(cargo-audit also emitted 18 allowed warnings — unmaintained/unsound informational entries — and spurious "yanked check" errors caused by the offline index; those are not vulnerabilities.)

**Highest concern:** `rmcp` 0.2.1 (DNS rebinding, 8.8) — `ruv-swarm-mcp` is a network-exposed MCP server, so this is directly in the threat path. The rustls-webpki name-constraint bypasses affect TLS certificate validation for any networked component.

Dependency upgrades are intentionally **not** performed here (owned by the dependency-modernization workstream).

## 3. npm audit — `ruv-swarm/npm` (lockfile present)

Verified locally against the live npm registry.

- **Full tree (incl. dev):** 38 vulnerabilities — 3 critical, 23 high, 7 moderate, 5 low.
  - Critical: `vitest`/`@vitest/ui` (arbitrary file read+execute via UI server; dev-only, fix is semver-major), `tar` (many path-traversal/DoS advisories; transitive, fix semver-major).
  - Notable high: `axios` (large advisory cluster incl. SSRF/credential leak), `ws`, `rollup`, `vite`, `wasm-pack`/`binary-install`, `serialize-javascript`. `wasm-opt` currently has **no fix** (pulls vulnerable `tar`).
- **Production dependencies only** (`npm audit --omit=dev`): 3 findings — the realistic runtime exposure:

| Package | Severity | Finding | Fix |
|---|---|---|---|
| ws | high | uninitialized memory disclosure (GHSA-58qx-3vcg-4xpx); memory-exhaustion DoS (GHSA-96hv-2xvq-fx4p) | `npm audit fix` (non-breaking) |
| tar-fs | high | symlink validation bypass | `npm audit fix` (non-breaking) |
| uuid | moderate | missing buffer bounds check (v3/v5/v6 with `buf`) | semver-major (uuid 14) |

## 4. Unsafe-code inventory — `src/simd/mod.rs`

Verified locally. The deep review's count of **10 unsafe AVX2 sites is confirmed**: 5 `unsafe { … }` dispatch blocks (lines 110, 127, 144, 161, 183) and 5 `unsafe fn` implementations (`matmul_avx2` :244, `matvec_avx2` :322, `add_bias_avx2` :370, `apply_activation_avx2` :443, `activation_derivatives_avx2` :543).

**None of the 10 sites carries a `// SAFETY:` comment or `# Safety` doc section** — a case-insensitive search for "safety" in the file returns zero matches. Recommended: document invariants (CPU-feature guard, slice-length/alignment preconditions) at each site and adopt `#![deny(unsafe_code)]` crate-wide with a scoped `#[allow]` in `src/simd` (matches review recommendation #7). Not changed here — `src/` is owned by a sibling task.

## 5. Secrets scan of tracked files

Pattern-based scan over all 2,314 `git ls-files` entries (AWS keys, GitHub/GitLab tokens, OpenAI/Slack/Google keys, private-key blocks, credentialed DB URLs). Verified locally. **Result: NOT clean — one true-positive file.**

- **`.claude.json` (tracked, ~1.7 MB)** contains captured session logs with **live third-party credentials** (values redacted here):
  - Supabase Postgres pooler connection string embedding a `sbp_…` access token as password (`postgresql://postgres.efdn…:sbp_c391…@aws-0-us-west-1.pooler.supabase.com:6543/postgres`)
  - a plaintext `SUPABASE_PASSWORD` (`oQzI…`)
  - an ElevenLabs API key (`sk_49b5…`)
  - a Google Gemini API key (`AIzaSyD7…`)
  - Supabase anon JWTs (designed to be public, but bundled alongside the above)
  These belong to an unrelated project ("dental" schema app) and are exposed to anyone with repo read access — and to the public if this repo/history is public. **Rotation is required regardless of removal**, since the values are in git history.
- `ruv-swarm/crates/ruv-swarm-mcp/src/error.rs:168` matches the DB-URL pattern but is a dummy string in a test (`postgres://user:pass@localhost/db`) — false positive.
- No `.env` files are tracked. No AWS/GitHub/GitLab/private-key material found elsewhere in tracked files.

## 6. CI gating changes made in this pass

- `.github/workflows/ci.yml` — added `pull_request` trigger (main/develop); added blocking `pr-gate` job (`cargo check` + `cargo test`, root crate, ubuntu/stable); added non-blocking `neuro-divergent-build` job (`continue-on-error: true`). The neuro-divergent workspace **fails `cargo check` at manifest-parse time** (verified locally): feature `wasm` declares `dep:ruv-fann/wasm`, which cargo rejects ("remove the `dep:` prefix").
- `.github/workflows/comprehensive-testing.yml` — added `pull_request` trigger (main/develop) so the npm-package pipeline gates PRs.
- `.github/workflows/security.yml` (new) — cargo-deny advisories (blocking; passes against working-tree lockfile) + npm audit (non-blocking until `npm audit fix` lands), on PRs to main, pushes to main, and a weekly Monday schedule.
- `wasm-build.yml` and `npm-release.yml` already had path-scoped `pull_request` triggers; `swarm-coordination.yml` is issue-event-driven — all left unchanged.

All three touched workflow files parse as valid YAML (verified locally with PyYAML). Actual execution on GitHub Actions **needs a CI run** to confirm.

## 7. Prioritized remediation

| # | Priority | Action | Owner / Notes |
|---|---|---|---|
| 1 | **P0** | Rotate ALL credentials in `.claude.json` (Supabase `sbp_` token + DB password, ElevenLabs key, Gemini key), then `git rm --cached .claude.json`, add to `.gitignore`, and purge from history (BFG/`git filter-repo`) | Rotation first — removal alone does not help; values are in history |
| 2 | **P0** | Land the PR-gating changes (ci.yml / comprehensive-testing.yml / security.yml) and mark `pr-gate` as a required status check | This audit; needs CI run + branch-protection setting |
| 3 | **P1** | ruv-swarm: bump `rmcp` ≥1.4.0 (8.8 DNS rebinding in MCP server), `quinn-proto` ≥0.11.15, `rustls-webpki` ≥0.103.13, `sqlx` ≥0.8.1 | Dependency workstream; rmcp first — network-exposed |
| 4 | **P1** | Run `npm audit fix` in `ruv-swarm/npm` (fixes prod-facing `ws`, `tar-fs` non-breakingly), then flip the `npm-audit` job in security.yml to blocking | Dependency workstream |
| 5 | **P2** | Fix `ruv-swarm-wasm`'s `ruv-fann = "0.1.5"` version req (root is 0.2.1) so the ruv-swarm workspace resolves and can be CI-audited; fix neuro-divergent `dep:ruv-fann/wasm` feature syntax and remove `continue-on-error` from its CI job | Currently blocks any workspace-level tooling |
| 6 | **P2** | Add `// SAFETY:` documentation to the 10 unsafe AVX2 sites in `src/simd/mod.rs`; adopt `#![deny(unsafe_code)]` + scoped allow | src/ owner |
| 7 | **P3** | Prune stale `deny.toml` ignores (paste, rand) once the lockfile update is committed; plan bincode 1.x migration (unmaintained, no safe upgrade); track `rsa` Marvin advisory (no fix; transitive via sqlx — sqlx 0.8 upgrade may change exposure) | Housekeeping |
| 8 | **P3** | Consider adding a dedicated secret scanner (e.g. gitleaks) to security.yml once finding #1 is remediated | Avoids re-introduction |
