//! Basic usage examples for ruv-swarm-mcp
//!
//! This example demonstrates how to create and configure an MCP server
//! for basic swarm orchestration tasks.

use std::sync::Arc;
use ruv_swarm_core::SwarmConfig;
use ruv_swarm_mcp::{orchestrator::SwarmOrchestrator, McpConfig, McpServer};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Create swarm configuration
    let swarm_config = SwarmConfig::default();

    // Create the swarm orchestrator
    let orchestrator = Arc::new(SwarmOrchestrator::new(swarm_config));

    // Configure the MCP server
    let mcp_config = McpConfig {
        bind_addr: "127.0.0.1:3000".parse().unwrap(),
        max_connections: 100,
        request_timeout_secs: 300,
        debug: true,
        allowed_origins: vec![], // Default to localhost only
    };

    // Create and start the MCP server
    let server = McpServer::new(orchestrator, mcp_config);
    
    println!("🚀 Starting ruv-swarm MCP server on 127.0.0.1:3000");
    println!("📊 MCP server ready with 11 available tools");
    println!("🔗 WebSocket endpoint: ws://127.0.0.1:3000/mcp");
    println!("🔗 Health endpoint: http://127.0.0.1:3000/health");
    println!("🔗 Tools endpoint: http://127.0.0.1:3000/tools");
    
    // Start the server (this will block)
    server.start().await?;

    Ok(())
}