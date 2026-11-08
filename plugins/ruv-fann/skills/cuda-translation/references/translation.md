# Translation evidence

Relevant checkout paths:

- `cuda-wasm/src/parser/`: recursive parser and AST contracts.
- `cuda-wasm/src/transpiler/`: Rust/WGSL source generation; inspect the actual tree before choosing a file.
- `cuda-wasm/src/bin/cuda2rust.rs`: compiler CLI.
- `cuda-wasm/dist/index.js`, `dist/index.mjs`, `dist/index.d.ts`: public Node entry points.
- `cuda-wasm/cli/index.js`: literal-argument wrapper CLI.
- `cuda-wasm/tests/`: parser fidelity, safety and integration regressions.
- `docs/review-cuda-cv.md`: local implementation/runtime results.

## Evidence stages

1. A parser accepts a known source unit and rejects unsupported/oversize syntax.
2. Generator output preserves the requested operation and includes required declarations.
3. The intended Rust compiler or WGSL validator accepts the output.
4. A real runtime executes it with defined buffers and scheduling.
5. Numeric values agree with a reference across representative and boundary shapes.
6. A fixed-hardware benchmark measures setup, execution, transfer and synchronization costs.

Report the highest stage actually reached. Never infer later stages from an earlier pass. Node unit fixtures prove wrapper behavior; they do not prove translator or kernel correctness.

## Features

Default serialization/CPU use does not discover GPU drivers or cloud metadata. Optional `gpu`/`webgpu`, `native-gpu`, `cuda-backend`, `vulkan`, and `nutanix` have additional prerequisites and incomplete operations. Read the actual Cargo feature graph and backend error paths before enabling them. A Nutanix discovery response does not establish migration or cloud execution.

## Future improvements

Use NVIDIA's CUDA language/programming guide and the W3C WGSL specification for semantic questions. Before adopting a fused attention or new WGPU backend, implement its actual synchronization/layout contract, validate numerical tolerances and test the relevant hardware. External paper throughput and old synthetic timing estimates must not become local release claims.
