//! Real CPU allocation/copy measurements; no pinned or device-memory claims.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cuda_rust_wasm::memory::{HostBuffer, MemoryPool, PoolConfig, UnifiedMemory};

fn allocations(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_allocations");
    for bytes in [1024usize, 16384, 262144] {
        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("initialized_host", bytes),
            &bytes,
            |b, &len| {
                b.iter(|| black_box(HostBuffer::<u8>::new(black_box(len)).unwrap()));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("initialized_raw_host", bytes),
            &bytes,
            |b, &len| {
                b.iter(|| black_box(UnifiedMemory::new(black_box(len)).unwrap()));
            },
        );
        let pool = MemoryPool::with_config(PoolConfig {
            prealloc_count: 0,
            ..PoolConfig::default()
        });
        pool.deallocate(pool.allocate(bytes));
        group.bench_with_input(
            BenchmarkId::new("pool_reuse_and_zero", bytes),
            &bytes,
            |b, &len| {
                b.iter(|| {
                    let buffer = pool.allocate(black_box(len));
                    black_box(&buffer);
                    pool.deallocate(buffer);
                });
            },
        );
    }
    group.finish();
}

fn copies_and_fill(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_host_operations");
    for len in [1024usize, 16384, 262144] {
        let input = vec![3.0f32; len];
        let mut buffer = HostBuffer::<f32>::new(len).unwrap();
        group.throughput(Throughput::Bytes((len * 4) as u64));
        group.bench_with_input(BenchmarkId::new("slice_copy", len), &len, |b, _| {
            b.iter(|| buffer.copy_from_slice(black_box(&input)).unwrap());
        });
        assert_eq!(buffer.as_slice(), input);
        group.bench_with_input(
            BenchmarkId::new("fill_existing_allocation", len),
            &len,
            |b, _| {
                b.iter(|| buffer.fill(black_box(2.0)));
            },
        );
        assert!(buffer.as_slice().iter().all(|&x| x == 2.0));
    }
    group.finish();
}
criterion_group!(benches, allocations, copies_and_fill);
criterion_main!(benches);
