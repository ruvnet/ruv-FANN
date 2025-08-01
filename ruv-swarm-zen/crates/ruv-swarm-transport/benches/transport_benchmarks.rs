//! Benchmarks for transport implementations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ruv_swarm_transport::{
    in_process::InProcessTransport,
    protocol::Message,
    shared_memory::{SharedMemoryInfo, SharedMemoryTransport},
    Transport, TransportConfig,
};
use tokio::runtime::Runtime;

fn bench_in_process_send(c: &mut Criterion) {
    let mut group = c.benchmark_group("in_process_send");
    let rt = Runtime::new().unwrap();

    for size in &[64, 256, 1024, 4096, 16384] {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                rt.block_on(async {
                    let config = TransportConfig::default();
                    let (transport1, mut transport2) = InProcessTransport::create_pair(
                        "sender".to_string(),
                        "receiver".to_string(),
                        config,
                    )
                    .await
                    .unwrap();

                    // Create message of specified size
                    let data = vec![0u8; size];
                    let msg = Message::event(
                        "sender".to_string(),
                        "bench".to_string(),
                        serde_json::to_value(&data).unwrap(),
                    );

                    // Send message
                    transport1.send("receiver", black_box(msg)).await.unwrap();

                    // Receive to complete the cycle
                    let _ = transport2.receive().await.unwrap();
                })
            });
        });
    }
    group.finish();
}

fn bench_in_process_broadcast(c: &mut Criterion) {
    let mut group = c.benchmark_group("in_process_broadcast");
    let rt = Runtime::new().unwrap();

    for receivers in &[2, 5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::from_parameter(receivers),
            receivers,
            |b, &receivers| {
                b.iter(|| {
                    rt.block_on(async {
                        let config = TransportConfig::default();
                        let builder =
                            ruv_swarm_transport::in_process::InProcessTransportBuilder::new();

                        // Create sender
                        let sender = builder
                            .build("sender".to_string(), config.clone())
                            .await
                            .unwrap();

                        // Create receivers
                        let mut _receivers = Vec::new();
                        for i in 0..receivers {
                            let receiver = builder
                                .build(format!("receiver{}", i), config.clone())
                                .await
                                .unwrap();
                            _receivers.push(receiver);
                        }

                        // Create message
                        let msg = Message::broadcast(
                            "sender".to_string(),
                            "topic".to_string(),
                            serde_json::json!({"data": "broadcast"}),
                        );

                        // Broadcast
                        sender.broadcast(black_box(msg)).await.unwrap();
                    })
                });
            },
        );
    }
    group.finish();
}

fn bench_shared_memory_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("shared_memory_throughput");
    let rt = Runtime::new().unwrap();

    group.throughput(Throughput::Elements(1000));
    group.bench_function("ring_buffer_1000_messages", |b| {
        use ruv_swarm_transport::shared_memory::RingBuffer;

        let buffer = RingBuffer::new(1024 * 1024); // 1MB buffer
        let data = vec![0u8; 256]; // 256 byte messages

        b.iter(|| {
            for _ in 0..1000 {
                buffer.write(black_box(&data)).unwrap();
                let _ = black_box(buffer.read().unwrap());
            }
        });
    });

    group.finish();
}

fn bench_message_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("message_serialization");

    use ruv_swarm_transport::protocol::{BinaryCodec, JsonCodec, MessageCodec};

    let binary_codec = BinaryCodec;
    let json_codec = JsonCodec;

    let msg = Message::request(
        "agent1".to_string(),
        "compute".to_string(),
        serde_json::json!({
            "input": vec![1.0; 100],
            "params": {
                "iterations": 1000,
                "threshold": 0.001
            }
        }),
    );

    group.bench_function("binary_encode", |b| {
        b.iter(|| {
            black_box(binary_codec.encode(&msg).unwrap());
        });
    });

    group.bench_function("json_encode", |b| {
        b.iter(|| {
            black_box(json_codec.encode(&msg).unwrap());
        });
    });

    let binary_data = binary_codec.encode(&msg).unwrap();
    let json_data = json_codec.encode(&msg).unwrap();

    group.bench_function("binary_decode", |b| {
        b.iter(|| {
            black_box(binary_codec.decode(&binary_data).unwrap());
        });
    });

    group.bench_function("json_decode", |b| {
        b.iter(|| {
            black_box(json_codec.decode(&json_data).unwrap());
        });
    });

    group.finish();
}

fn bench_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression");

    // Test with different data patterns
    let random_data = vec![rand::random::<u8>(); 1024];
    let repetitive_data = vec![42u8; 1024];
    let text_data = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        .repeat(20)
        .into_bytes();

    use ruv_swarm_transport::websocket::WebSocketTransport;

    group.bench_function("compress_random", |b| {
        b.iter(|| {
            black_box(WebSocketTransport::compress(&random_data).unwrap());
        });
    });

    group.bench_function("compress_repetitive", |b| {
        b.iter(|| {
            black_box(WebSocketTransport::compress(&repetitive_data).unwrap());
        });
    });

    group.bench_function("compress_text", |b| {
        b.iter(|| {
            black_box(WebSocketTransport::compress(&text_data).unwrap());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_in_process_send,
    bench_in_process_broadcast,
    bench_shared_memory_throughput,
    bench_message_serialization,
    bench_compression
);
criterion_main!(benches);
