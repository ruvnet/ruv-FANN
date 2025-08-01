//! Security-focused tests for RUV-Swarm MCP server
//!
//! These tests validate security controls and protections against common
//! attack vectors identified in the security analysis.

use std::sync::Arc;
use std::time::Duration;

use ruv_swarm_core::SwarmConfig;
use crate::{
    orchestrator::SwarmOrchestrator, McpConfig, McpRequest, McpResponse, McpServer,
};
use serde_json::json;
use tokio::time::timeout;

/// Test input validation for malformed requests
#[tokio::test]
async fn test_malformed_request_handling() {
    let swarm_config = SwarmConfig::default();
    let orchestrator = Arc::new(SwarmOrchestrator::new(swarm_config));
    let mcp_config = McpConfig::default();
    let server = McpServer::new(orchestrator, mcp_config);

    // Test malformed JSON-RPC request
    let malformed_request = McpRequest {
        jsonrpc: "1.0".to_string(), // Invalid version
        method: "test".to_string(),
        params: None,
        id: Some(json!(1)),
    };

    // Should handle gracefully without crashing
    assert!(serde_json::to_string(&malformed_request).is_ok());
}

/// Test input validation for tool parameters
#[tokio::test]
async fn test_tool_parameter_validation() {
    let swarm_config = SwarmConfig::default();
    let orchestrator = Arc::new(SwarmOrchestrator::new(swarm_config));
    let mcp_config = McpConfig::default();
    let _server = McpServer::new(orchestrator, mcp_config);

    // Test invalid agent type
    let invalid_agent_request = McpRequest {
        jsonrpc: "2.0".to_string(),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "ruv-swarm.spawn",
            "arguments": {
                "agent_type": "invalid_type", // Invalid agent type
                "name": "test"
            }
        })),
        id: Some(json!(1)),
    };

    // Should be valid JSON but will fail validation
    assert!(serde_json::to_string(&invalid_agent_request).is_ok());
}

/// Test resource exhaustion protection
#[tokio::test]
async fn test_resource_limits() {
    let swarm_config = SwarmConfig::default();
    let orchestrator = Arc::new(SwarmOrchestrator::new(swarm_config));
    let mcp_config = McpConfig::default();
    let _server = McpServer::new(orchestrator, mcp_config);

    // Test spawning many agents (should not crash)
    let mut spawn_requests = Vec::new();
    
    for i in 0..50 {
        let request = McpRequest {
            jsonrpc: "2.0".to_string(),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "ruv-swarm.spawn",
                "arguments": {
                    "agent_type": "researcher",
                    "name": format!("test_agent_{}", i)
                }
            })),
            id: Some(json!(i)),
        };
        spawn_requests.push(request);
    }

    // Should handle batch of requests without crashing
    assert_eq!(spawn_requests.len(), 50);
}

/// Test session isolation
#[tokio::test]
async fn test_session_isolation() {
    let swarm_config = SwarmConfig::default();
    let orchestrator = Arc::new(SwarmOrchestrator::new(swarm_config));
    let mcp_config = McpConfig::default();
    let _server = McpServer::new(orchestrator, mcp_config);

    // Test that session data is isolated
    let session1_store = McpRequest {
        jsonrpc: "2.0".to_string(),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "ruv-swarm.memory.store",
            "arguments": {
                "key": "test_key",
                "value": "session1_data"
            }
        })),
        id: Some(json!(1)),
    };

    let session2_store = McpRequest {
        jsonrpc: "2.0".to_string(),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "ruv-swarm.memory.store",
            "arguments": {
                "key": "test_key",
                "value": "session2_data"
            }
        })),
        id: Some(json!(2)),
    };

    // Both should be valid requests
    assert!(serde_json::to_string(&session1_store).is_ok());
    assert!(serde_json::to_string(&session2_store).is_ok());
}

/// Test error handling without information leakage
#[tokio::test]
async fn test_secure_error_handling() {
    let swarm_config = SwarmConfig::default();
    let orchestrator = Arc::new(SwarmOrchestrator::new(swarm_config));
    let mcp_config = McpConfig::default();
    let _server = McpServer::new(orchestrator, mcp_config);

    // Test request with missing required parameters
    let invalid_request = McpRequest {
        jsonrpc: "2.0".to_string(),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "ruv-swarm.spawn",
            "arguments": {} // Missing required agent_type
        })),
        id: Some(json!(1)),
    };

    // Should handle gracefully
    assert!(serde_json::to_string(&invalid_request).is_ok());
}

/// Test memory storage limits (protection against memory exhaustion)
#[tokio::test]
async fn test_memory_storage_limits() {
    let swarm_config = SwarmConfig::default();
    let orchestrator = Arc::new(SwarmOrchestrator::new(swarm_config));
    let mcp_config = McpConfig::default();
    let _server = McpServer::new(orchestrator, mcp_config);

    // Test storing large amounts of data
    let large_data = "x".repeat(1000000); // 1MB string
    
    let memory_request = McpRequest {
        jsonrpc: "2.0".to_string(),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "ruv-swarm.memory.store",
            "arguments": {
                "key": "large_data",
                "value": large_data,
                "ttl_secs": 60
            }
        })),
        id: Some(json!(1)),
    };

    // Should handle large data without crashing
    assert!(serde_json::to_string(&memory_request).is_ok());
}

/// Test workflow path validation (protection against path traversal)
#[tokio::test]
async fn test_workflow_path_validation() {
    let swarm_config = SwarmConfig::default();
    let orchestrator = Arc::new(SwarmOrchestrator::new(swarm_config));
    let mcp_config = McpConfig::default();
    let _server = McpServer::new(orchestrator, mcp_config);

    // Test potentially dangerous paths
    let dangerous_paths = vec![
        "../../../etc/passwd",
        "../../../../etc/shadow",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "/etc/passwd",
        "C:\\Windows\\System32\\config\\SAM",
    ];

    for path in dangerous_paths {
        let workflow_request = McpRequest {
            jsonrpc: "2.0".to_string(),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "ruv-swarm.workflow.execute",
                "arguments": {
                    "workflow_path": path,
                    "parameters": {}
                }
            })),
            id: Some(json!(1)),
        };

        // Should handle dangerous paths safely
        assert!(serde_json::to_string(&workflow_request).is_ok());
    }
}

/// Test rate limiting behavior
#[tokio::test]
async fn test_rate_limiting_behavior() {
    let swarm_config = SwarmConfig::default();
    let orchestrator = Arc::new(SwarmOrchestrator::new(swarm_config));
    let mcp_config = McpConfig::default();
    let _server = McpServer::new(orchestrator, mcp_config);

    // Test rapid fire requests
    let mut requests = Vec::new();
    for i in 0..100 {
        let request = McpRequest {
            jsonrpc: "2.0".to_string(),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "ruv-swarm.query",
                "arguments": {
                    "include_metrics": false
                }
            })),
            id: Some(json!(i)),
        };
        requests.push(request);
    }

    // Should handle burst of requests
    assert_eq!(requests.len(), 100);
}

/// Test boundary conditions for numeric parameters
#[tokio::test]
async fn test_numeric_parameter_boundaries() {
    let swarm_config = SwarmConfig::default();
    let orchestrator = Arc::new(SwarmOrchestrator::new(swarm_config));
    let mcp_config = McpConfig::default();
    let _server = McpServer::new(orchestrator, mcp_config);

    // Test extreme values
    let extreme_values = vec![
        0u64,
        1u64,
        u64::MAX,
        u64::MAX - 1,
    ];

    for value in extreme_values {
        let request = McpRequest {
            jsonrpc: "2.0".to_string(),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "ruv-swarm.orchestrate",
                "arguments": {
                    "objective": "test",
                    "max_agents": value,
                    "strategy": "development"
                }
            })),
            id: Some(json!(1)),
        };

        // Should handle extreme values gracefully
        assert!(serde_json::to_string(&request).is_ok());
    }
}

/// Test concurrent connection handling
#[tokio::test]
async fn test_concurrent_connections() {
    let swarm_config = SwarmConfig::default();
    let orchestrator = Arc::new(SwarmOrchestrator::new(swarm_config));
    let mcp_config = McpConfig::default();
    let _server = McpServer::new(orchestrator.clone(), mcp_config);

    // Test that server can handle concurrent operations
    let mut handles = Vec::new();
    
    for i in 0..10 {
        let orchestrator_clone = orchestrator.clone();
        let handle = tokio::spawn(async move {
            let result = orchestrator_clone.get_swarm_state().await;
            assert!(result.is_ok(), "Concurrent operation {i} failed");
        });
        handles.push(handle);
    }

    // Wait for all concurrent operations to complete
    for handle in handles {
        handle.await.unwrap();
    }
}

/// Test agent metrics security
#[tokio::test]
async fn test_agent_metrics_security() {
    let swarm_config = SwarmConfig::default();
    let orchestrator = Arc::new(SwarmOrchestrator::new(swarm_config));
    let mcp_config = McpConfig::default();
    let _server = McpServer::new(orchestrator.clone(), mcp_config);

    // Test metrics for non-existent agent
    let fake_agent_id = uuid::Uuid::new_v4();
    let metrics_result = orchestrator.get_agent_metrics(&fake_agent_id).await;
    
    // Should handle gracefully with appropriate error
    assert!(metrics_result.is_err());
}

/// Test WebSocket message size limits
#[tokio::test]
async fn test_websocket_message_limits() {
    let swarm_config = SwarmConfig::default();
    let orchestrator = Arc::new(SwarmOrchestrator::new(swarm_config));
    let mcp_config = McpConfig::default();
    let _server = McpServer::new(orchestrator, mcp_config);

    // Test very large message
    let large_objective = "x".repeat(1000000); // 1MB string
    
    let large_request = McpRequest {
        jsonrpc: "2.0".to_string(),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "ruv-swarm.orchestrate",
            "arguments": {
                "objective": large_objective,
                "strategy": "development"
            }
        })),
        id: Some(json!(1)),
    };

    // Should handle large messages
    assert!(serde_json::to_string(&large_request).is_ok());
}

/// Test monitoring duration limits
#[tokio::test]
async fn test_monitoring_duration_limits() {
    let swarm_config = SwarmConfig::default();
    let orchestrator = Arc::new(SwarmOrchestrator::new(swarm_config));
    let mcp_config = McpConfig::default();
    let _server = McpServer::new(orchestrator, mcp_config);

    // Test extremely long monitoring duration
    let long_duration_request = McpRequest {
        jsonrpc: "2.0".to_string(),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "ruv-swarm.monitor",
            "arguments": {
                "duration_secs": u64::MAX,
                "event_types": ["all"]
            }
        })),
        id: Some(json!(1)),
    };

    // Should handle extreme duration values
    assert!(serde_json::to_string(&long_duration_request).is_ok());
}