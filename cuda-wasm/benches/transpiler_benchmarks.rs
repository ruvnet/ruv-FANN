//! Parse and generate actual Rust/WGSL text; this does not measure execution.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cuda_rust_wasm::{parser::CudaParser, transpiler::Transpiler};
fn code_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda_source_processing");
    let parser = CudaParser::new();
    let generator = Transpiler::new();
    for kernels in [1usize, 8, 32] {
        let source: String = (0..kernels)
            .map(|i| {
                format!(
            "__global__ void copy_{i}(float* a, float* b) {{ int i = threadIdx.x; b[i] = a[i]; }}\n"
        )
            })
            .collect();
        let ast = parser.parse(&source).unwrap();
        assert_eq!(ast.items.len(), kernels);
        assert!(!generator.transpile(ast.clone()).unwrap().is_empty());
        assert!(!generator.to_wgsl(ast.clone()).unwrap().is_empty());
        group.throughput(Throughput::Bytes(source.len() as u64));
        group.bench_with_input(BenchmarkId::new("parse", kernels), &source, |b, source| {
            b.iter(|| black_box(parser.parse(black_box(source)).unwrap()));
        });
        // The public generator consumes the AST; cloning is part of this measurement.
        group.bench_with_input(
            BenchmarkId::new("rust_text_with_ast_clone", kernels),
            &ast,
            |b, ast| {
                b.iter(|| black_box(generator.transpile(black_box(ast.clone())).unwrap()));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("wgsl_text_with_ast_clone", kernels),
            &ast,
            |b, ast| {
                b.iter(|| black_box(generator.to_wgsl(black_box(ast.clone())).unwrap()));
            },
        );
    }
    group.finish();
}
criterion_group!(benches, code_generation);
criterion_main!(benches);
