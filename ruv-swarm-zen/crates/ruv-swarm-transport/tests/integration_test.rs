//! Integration tests for transport layer

use ruv_swarm_transport::{
    in_process::InProcessTransport, protocol::Message, Transport, TransportConfig,
};

#[tokio::test]
async fn test_in_process_communication() {
    // Create a pair of connected transports
    let config = TransportConfig::default();
    let (mut transport1, mut transport2) =
        InProcessTransport::create_pair("agent1".to_string(), "agent2".to_string(), config)
            .await
            .unwrap();

    // Test basic send/receive
    let msg = Message::event(
        "agent1".to_string(),
        "test_event".to_string(),
        serde_json::json!({
            "data": "Hello from agent1",
            "timestamp": 12345
        }),
    );

    // Send from agent1 to agent2
    transport1.send("agent2", msg.clone()).await.unwrap();

    // Receive on agent2
    let (from, received) = transport2.receive().await.unwrap();
    assert_eq!(from, "agent1");
    assert_eq!(received.source, "agent1");

    // Test request/response pattern
    let request = Message::request(
        "agent2".to_string(),
        "compute".to_string(),
        serde_json::json!({
            "input": [1, 2, 3, 4, 5],
            "operation": "sum"
        }),
    );

    let correlation_id = match &request.payload {
        ruv_swarm_transport::protocol::MessageType::Request { correlation_id, .. } => {
            *correlation_id
        }
        _ => panic!("Expected request message"),
    };

    // Send request from agent2 to agent1
    transport2.send("agent1", request).await.unwrap();

    // Receive request on agent1
    let (from, received_request) = transport1.receive().await.unwrap();
    assert_eq!(from, "agent2");

    // Send response back
    let response = Message::response(
        "agent1".to_string(),
        correlation_id,
        Ok(serde_json::json!({
            "result": 15,
            "computation_time_ms": 5
        })),
    );

    transport1.send("agent2", response).await.unwrap();

    // Receive response on agent2
    let (from, received_response) = transport2.receive().await.unwrap();
    assert_eq!(from, "agent1");

    match received_response.payload {
        ruv_swarm_transport::protocol::MessageType::Response {
            correlation_id: resp_id,
            result,
        } => {
            assert_eq!(resp_id, correlation_id);
            assert!(result.is_ok());
        }
        _ => panic!("Expected response message"),
    }
}

#[tokio::test]
async fn test_broadcast() {
    use ruv_swarm_transport::in_process::InProcessTransportBuilder;

    let builder = InProcessTransportBuilder::new();
    let config = TransportConfig::default();

    // Create three agents
    let mut agent1 = builder
        .build("agent1".to_string(), config.clone())
        .await
        .unwrap();
    let mut agent2 = builder
        .build("agent2".to_string(), config.clone())
        .await
        .unwrap();
    let mut agent3 = builder
        .build("agent3".to_string(), config.clone())
        .await
        .unwrap();

    // Subscribe to broadcasts
    let mut broadcast_rx2 = agent2.broadcast_tx.subscribe();
    let mut broadcast_rx3 = agent3.broadcast_tx.subscribe();

    // Broadcast a message from agent1
    let broadcast_msg = Message::broadcast(
        "agent1".to_string(),
        "system_update".to_string(),
        serde_json::json!({
            "status": "active",
            "version": "1.0.0"
        }),
    );

    agent1.broadcast(broadcast_msg.clone()).await.unwrap();

    // Both agent2 and agent3 should receive the broadcast
    let received2 = broadcast_rx2.recv().await.unwrap();
    let received3 = broadcast_rx3.recv().await.unwrap();

    assert_eq!(received2.source, "agent1");
    assert_eq!(received3.source, "agent1");

    match (&received2.payload, &received3.payload) {
        (
            ruv_swarm_transport::protocol::MessageType::Broadcast { topic: topic2, .. },
            ruv_swarm_transport::protocol::MessageType::Broadcast { topic: topic3, .. },
        ) => {
            assert_eq!(topic2, "system_update");
            assert_eq!(topic3, "system_update");
        }
        _ => panic!("Expected broadcast messages"),
    }
}

#[tokio::test]
async fn test_transport_stats() {
    let config = TransportConfig::default();
    let (mut transport1, mut transport2) = InProcessTransport::create_pair(
        "stats_test1".to_string(),
        "stats_test2".to_string(),
        config,
    )
    .await
    .unwrap();

    // Send multiple messages
    for i in 0..5 {
        let msg = Message::event(
            "stats_test1".to_string(),
            format!("event_{}", i),
            serde_json::json!({"index": i}),
        );
        transport1.send("stats_test2", msg).await.unwrap();
        let _ = transport2.receive().await.unwrap();
    }

    // Check stats
    let stats1 = transport1.stats();
    let stats2 = transport2.stats();

    assert_eq!(stats1.messages_sent, 5);
    assert_eq!(stats2.messages_received, 5);
    assert!(stats1.bytes_sent > 0);
    assert!(stats2.bytes_received > 0);
    assert!(stats1.last_activity.is_some());
    assert!(stats2.last_activity.is_some());
}
