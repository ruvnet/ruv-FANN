//! Actual WASM runtime coverage; native network/shared-memory backends are platform-specific.
use crate::{
    in_process::{InProcessRegistry, InProcessTransport},
    protocol::Message,
    Transport, TransportConfig, TransportError,
};
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test]
async fn in_process_messages_and_broadcasts_work_in_wasm() {
    let registry = InProcessRegistry::new();
    let mut first =
        InProcessTransport::new("first".into(), TransportConfig::default(), registry.clone())
            .await
            .unwrap();
    let mut second = InProcessTransport::new("second".into(), TransportConfig::default(), registry)
        .await
        .unwrap();
    let message = Message::event(
        "first".into(),
        "event".into(),
        serde_json::json!({"value": 42}),
    );
    first.send("second", message).await.unwrap();
    let (sender, message) = second.receive().await.unwrap();
    assert_eq!(sender, "first");
    assert_eq!(message.source, "first");
    second
        .broadcast(Message::broadcast(
            "second".into(),
            "topic".into(),
            serde_json::json!({"ready": true}),
        ))
        .await
        .unwrap();
    let (sender, message) = first.receive().await.unwrap();
    assert_eq!(sender, "second");
    assert_eq!(message.source, "second");
    first.close().await.unwrap();
    assert!(!first.is_connected());
}

#[wasm_bindgen_test]
async fn receiver_limits_and_duplicate_ids_hold_in_wasm() {
    let registry = InProcessRegistry::new();
    let sender = InProcessTransport::new(
        "sender".into(),
        TransportConfig::default(),
        registry.clone(),
    )
    .await
    .unwrap();
    let receiver = InProcessTransport::new(
        "receiver".into(),
        TransportConfig {
            max_message_size: 128,
            ..Default::default()
        },
        registry.clone(),
    )
    .await
    .unwrap();
    assert!(
        InProcessTransport::new("receiver".into(), TransportConfig::default(), registry)
            .await
            .is_err()
    );
    let message = Message::event(
        "sender".into(),
        "large".into(),
        serde_json::json!({"value": "x".repeat(512)}),
    );
    assert!(matches!(
        sender.send("receiver", message).await,
        Err(TransportError::MessageTooLarge { max: 128, .. })
    ));
    assert_eq!(receiver.stats().messages_received, 0);
}
