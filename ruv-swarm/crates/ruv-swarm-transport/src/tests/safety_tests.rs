//! Regressions for bounded transport frames and concurrent ring access.
use crate::{shared_memory::RingBuffer, websocket::WebSocketTransport, TransportError};
use std::collections::HashSet;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Barrier, Mutex,
};
use std::time::{Duration, Instant};

#[test]
fn gzip_expansion_is_bounded_and_exact_limit_is_accepted() {
    let data = vec![0; 4096];
    let compressed = WebSocketTransport::compress(&data).unwrap();
    assert_eq!(
        WebSocketTransport::decompress_with_limit(&compressed, 4096).unwrap(),
        data
    );
    assert!(matches!(
        WebSocketTransport::decompress_with_limit(&compressed, 4095),
        Err(TransportError::MessageTooLarge { max: 4095, .. })
    ));
    assert!(WebSocketTransport::decompress_with_limit(&compressed, 0).is_err());
    assert!(WebSocketTransport::decompress_with_limit(&compressed[..10], 4096).is_err());
}

#[test]
fn ring_full_failure_preserves_existing_frame() {
    let buffer = RingBuffer::new(8);
    buffer.write(b"abcd").unwrap();
    assert_eq!(buffer.available_space(), 0);
    assert!(buffer.write(b"x").is_err());
    assert_eq!(buffer.read().unwrap(), b"abcd");
    assert_eq!(buffer.available_space(), 8);
    buffer.write(b"").unwrap();
    assert_eq!(buffer.read().unwrap(), b"");
    assert!(RingBuffer::new(0).write(b"").is_err());
}

#[test]
fn concurrent_ring_frames_are_delivered_exactly_once() {
    const WRITERS: usize = 4;
    const READERS: usize = 2;
    const PER_WRITER: usize = 500;
    let buffer = Arc::new(RingBuffer::new(127)); // Deliberately unaligned wrap points.
    let barrier = Arc::new(Barrier::new(WRITERS + READERS));
    let received = Arc::new(Mutex::new(HashSet::new()));
    let count = Arc::new(AtomicUsize::new(0));
    let deadline = Instant::now() + Duration::from_secs(10);
    std::thread::scope(|scope| {
        for writer in 0..WRITERS {
            let (buffer, barrier) = (buffer.clone(), barrier.clone());
            scope.spawn(move || {
                barrier.wait();
                for index in 0..PER_WRITER {
                    let frame = ((writer * PER_WRITER + index) as u64).to_le_bytes();
                    while buffer.write(&frame).is_err() {
                        assert!(Instant::now() < deadline, "writer stalled");
                        std::thread::yield_now();
                    }
                }
            });
        }
        for _ in 0..READERS {
            let (buffer, barrier, received, count) = (
                buffer.clone(),
                barrier.clone(),
                received.clone(),
                count.clone(),
            );
            scope.spawn(move || {
                barrier.wait();
                while count.load(Ordering::Acquire) < WRITERS * PER_WRITER {
                    assert!(Instant::now() < deadline, "reader stalled");
                    if let Some(frame) = buffer.read() {
                        let value = u64::from_le_bytes(frame.try_into().expect("corrupt frame"));
                        assert!(received.lock().unwrap().insert(value), "duplicate frame");
                        count.fetch_add(1, Ordering::Release);
                    } else {
                        std::thread::yield_now();
                    }
                }
            });
        }
    });
    assert_eq!(received.lock().unwrap().len(), WRITERS * PER_WRITER);
    assert!(buffer.is_empty());
    assert_eq!(buffer.available_space(), 127);
}

#[tokio::test]
async fn broadcast_uses_public_receive_and_receiver_limit_is_enforced() {
    use crate::{
        in_process::{InProcessRegistry, InProcessTransport},
        protocol::Message,
        Transport, TransportConfig,
    };
    let registry = InProcessRegistry::new();
    let sender = InProcessTransport::new(
        "sender".into(),
        TransportConfig::default(),
        registry.clone(),
    )
    .await
    .unwrap();
    let mut receiver = InProcessTransport::new(
        "receiver".into(),
        TransportConfig::default(),
        registry.clone(),
    )
    .await
    .unwrap();
    let message = Message::event("sender".into(), "event".into(), serde_json::Value::Null);
    sender.broadcast(message.clone()).await.unwrap();
    let (from, actual) = tokio::time::timeout(Duration::from_secs(1), receiver.receive())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(from, "sender");
    assert_eq!(actual.id, message.id);
    assert!(InProcessTransport::new(
        "receiver".into(),
        TransportConfig::default(),
        registry.clone()
    )
    .await
    .is_err());
    let _small = InProcessTransport::new(
        "small".into(),
        TransportConfig {
            max_message_size: 1,
            ..Default::default()
        },
        registry,
    )
    .await
    .unwrap();
    assert!(matches!(
        sender.send("small", message).await,
        Err(TransportError::MessageTooLarge { max: 1, .. })
    ));
}
