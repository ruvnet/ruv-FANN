# Review and regression matrix

Resolve these paths inside the selected checkout. Check the current source because tests and constants can evolve.

| Family/boundary | Read and test |
| --- | --- |
| Core topology/SIMD/I/O | `src/tests/core_regressions.rs`, `src/io/limits.rs`, `docs/review-core.md` |
| Repeated inference allocation | `tests/inference_allocations.rs` |
| Native device compute/storage | `tests/webgpu_release_gate.rs`, `src/webgpu/buffer_storage.rs` |
| Real Rust browser compute | `scripts/verify-rust-browser-gpu.cjs` |
| Direct shader correctness | `scripts/verify-webgpu.cjs` |
| Node process/SQL/integrity | `ruv-swarm/npm/test/security-boundaries.test.js` |
| Node framing/backpressure | `ruv-swarm/npm/test/mcp-framing.test.js` |
| Actual Node MCP state/compute | `ruv-swarm/npm/test/mcp-runtime.test.js` |
| Node source/artifact matching | `ruv-swarm/npm/test/wasm-provenance.test.js` |
| Dependency tooling poisoning | `ruv-swarm/npm/test/tooling-security.test.js` |
| Rust scheduling/message limits | `ruv-swarm/crates/ruv-swarm-core/src/tests/scheduling_regressions.rs`, transport/persistence tests |
| Forecast leakage/numerics/state | `neuro-divergent/tests/README.md`, `docs/review-forecasting.md` |
| CUDA memory/parser safety | `cuda-wasm/tests/safety_regressions.rs`, `docs/review-cuda-cv.md` |
| Portable image arithmetic | `opencv-rust/opencv-core/tests/matrix_safety.rs` |
| Native ABI errors/lifetimes | `opencv-rust/opencv-sys/tests/` |
| Browser image lifetime | `opencv-rust/tests/browser-smoke.cjs` |
| Safe release staging | `scripts/tests/test_verify_release.py` |

## Suggested abuse cases

Empty/nonfinite/ragged numeric input; enormous declared dimensions with tiny payload; integer overflow; compressed expansion; malformed/truncated model data; source indices outside the previous layer; mismatched gradient buffers; invalid vector tails; unsupported CPU features; cloned/spoofed JavaScript typed-array prototypes; FIFO/symlink replacement; shell/SQL metacharacters; oversized JSON-RPC followed by a valid frame; full queues; racing claims; native ROI aliases after parent free; WASM memory growth after exported pixel data.

Choose cases that reach the changed production boundary. Tests should assert the security/correctness contract, not duplicate implementation internals. Use task-owned files, isolated databases and bounded subprocesses.

## Evidence standard

A completed task ledger is coordination evidence. A real invocation is execution evidence. A numeric oracle establishes the checked computation. A held-out evaluation establishes accuracy only for its disclosed data/split. A measured workload establishes performance only for its configuration. Keep those assertions separate.
