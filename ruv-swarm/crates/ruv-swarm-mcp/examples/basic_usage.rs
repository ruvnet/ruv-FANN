//! Basic usage examples for ruv-swarm-mcp
//!
//! This example demonstrates how to create and configure an MCP server
//! for basic swarm orchestration tasks.

use anyhow::Context;
use ruv_swarm_core::SwarmConfig;
use ruv_swarm_mcp::{orchestrator::SwarmOrchestrator, McpConfig, McpServer};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let configured = match std::env::var("RUV_SWARM_MCP_BIND_ADDR") {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => "127.0.0.1:3000".to_string(),
        Err(error) => return Err(error).context("Failed to read RUV_SWARM_MCP_BIND_ADDR"),
    };
    let bind_addr = configured
        .parse()
        .context("RUV_SWARM_MCP_BIND_ADDR must be an IP socket address")?;

    // Create swarm configuration
    let swarm_config = SwarmConfig::default();

    // Create the swarm orchestrator
    let orchestrator = Arc::new(SwarmOrchestrator::new(swarm_config).await);

    // Configure the MCP server
    let mcp_config = McpConfig {
        bind_addr,
        max_connections: 100,
        request_timeout_secs: 300,
        debug: true,
        allowed_origins: vec![], // Default to localhost only
    };

    // Create and start the MCP server
    let server = McpServer::new(orchestrator, mcp_config);

    println!("Starting ruv-swarm MCP server on {}", bind_addr);
    println!("The post-bind listener log reports readiness; /tools lists current schemas.");
    println!("WebSocket endpoint: ws://{}/mcp", bind_addr);
    println!("Health endpoint: http://{}/health", bind_addr);
    println!("Tools endpoint: http://{}/tools", bind_addr);

    // Start the server (this will block)
    server.start().await?;

    Ok(())
}
