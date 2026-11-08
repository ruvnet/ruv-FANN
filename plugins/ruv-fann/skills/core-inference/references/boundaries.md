# Core boundary and GPU checklist

These defaults describe the modernized checkout. Read implementation constants before changing or relying on them.

| Boundary | Current default |
| --- | --- |
| FANN text model / estimated decoded topology | 256 MiB each |
| Individual text line | 16 MiB |
| Binary / JSON input | 256 MiB |
| Training input / header estimate | 1 GiB each |
| Expanded streaming gzip output | 1 GiB; caller-selected cap available |
| Prepared snapshot plus one workspace estimate | 64 MiB; source model, allocator overhead/capacity and other concurrent workspaces excluded |
| Prepared neurons / connections | 1,048,576 / 16,777,216; checked before packing |
| Browser GPU buffer | 64 MiB and actual device limits |
| Browser matrix multiply work | 67,108,864 multiply-adds per call |

Use the exported custom-limit readers when the application needs a tighter budget. `decompress_data_with_limit` must enforce the expanded output cap. Retained caller data, containers, spare capacity and concurrent operations consume additional memory.

## Native numerical changes

Test empty/rejected shapes, one-layer construction, wide fan-in, sparse source indices, irregular SIMD tails, all declared activations and gradients. Invalid dimensions must reject before unsafe loads/stores. Preserve f64 values rather than silently narrowing through GPU f32 storage.

## Browser API changes

Read `src/webgpu/wasm_gpu_bridge.rs` and `src/webgpu/wasm_gpu_dispatch.rs` when present in the checkout and use their exported API. Supply actual `Float32Array` values; strings, ordinary arrays, spoofed prototypes, nonfinite values and invalid dimensions are rejection cases. Oversized arrays must reject before copying into WASM. Verify behavior when an input aliases WASM memory and memory growth detaches it.

Cover simultaneous calls, buffer cleanup, device loss, `destroy`, and reinitialization. Compare all produced values with a CPU reference, including tail workgroups. Treat shader validation, numerical agreement and device throughput as separate evidence.

## Performance evidence

`tests/inference_allocations.rs` checks allocation behavior; `benches/neural_network.rs` measures the root workload. Record model shape, weights, dtype, compiler flags, CPU features, threads, warm-up and samples. Include upload/readback and pipeline setup in the appropriate GPU end-to-end result. A smoke run of Criterion is not a speed comparison.

## Prepared snapshot regression cases

Read `src/prepared.rs`, `tests/prepared_inference.rs`, and
`tests/prepared_allocations.rs`. Compare regular and prepared f32/f64 inference
with exact weights, all activations, one-layer models, dense/sparse/shuffled/
duplicate connections, and canonical bias neurons. Mutate/drop the source after
preparation and confirm the snapshot remains independent. Share a snapshot
across workers with separate workspaces; a shared mutable workspace is not the
concurrency contract.

Reject nonunit/nonterminal/connected biases, input-neuron connections, nonfinite
weights/steepness/input, bad indices, insufficient workspace, and resource limits.
Exercise nonfinite weighted sums/activation results and check that output remains
unchanged on every failure. Workspace contents may change and remain reusable.
The byte estimate counts requested packed descriptors/parameters/indices and one
workspace's two buffers; it is not peak RAM or a process-wide allocation quota.
