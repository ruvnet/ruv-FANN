//! Existing CPU kernel-dispatch and tiled reference-attention measurements.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cuda_rust_wasm::runtime::flash_attention::{FlashAttention, FlashAttentionConfig};
use cuda_rust_wasm::runtime::{
    launch_kernel, Block, Grid, KernelFunction, LaunchConfig, ThreadContext,
};
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};

#[derive(Clone)]
struct Inputs {
    a: Arc<Vec<f32>>,
    b: Arc<Vec<f32>>,
    out: Arc<Vec<AtomicU32>>,
}
struct Add;
impl KernelFunction<Inputs> for Add {
    fn execute(&self, args: Inputs, context: ThreadContext) {
        let index = context.global_thread_id();
        if index < args.a.len() {
            args.out[index].store((args.a[index] + args.b[index]).to_bits(), Ordering::Relaxed);
        }
    }
    fn name(&self) -> &str {
        "cpu_add"
    }
}
fn dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_emulated_kernel_dispatch_and_add");
    for len in [1024usize, 16384] {
        let args = Inputs {
            a: Arc::new(vec![1.0; len]),
            b: Arc::new(vec![2.0; len]),
            out: Arc::new((0..len).map(|_| AtomicU32::new(0)).collect()),
        };
        let run = || {
            launch_kernel(
                Add,
                LaunchConfig::new(Grid::new(len.div_ceil(256) as u32), Block::new(256u32)),
                args.clone(),
            )
            .unwrap()
        };
        run();
        assert!(args
            .out
            .iter()
            .all(|x| f32::from_bits(x.load(Ordering::Relaxed)) == 3.0));
        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::new("elements", len), &len, |b, _| {
            b.iter(|| {
                run();
                black_box(&args.out);
            })
        });
    }
    group.finish();
}
fn attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_tiled_attention_reference");
    for sequence in [32usize, 64, 128] {
        let input = vec![0.125; sequence * 32];
        let engine = FlashAttention::new(FlashAttentionConfig::new(32, 1));
        let reference = engine.forward(&input, &input, &input).unwrap();
        assert!(reference.output.iter().all(|x| (*x - 0.125).abs() < 1e-5));
        group.bench_with_input(BenchmarkId::new("sequence", sequence), &sequence, |b, _| {
            b.iter(|| black_box(engine.forward(black_box(&input), &input, &input).unwrap()));
        });
    }
    group.finish();
}
criterion_group!(benches, dispatch, attention);
criterion_main!(benches);
