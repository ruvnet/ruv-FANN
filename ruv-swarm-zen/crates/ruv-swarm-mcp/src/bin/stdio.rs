//! Stdio transport binary for Claude Code integration
//!
//! This binary provides MCP server functionality over stdin/stdout,
//! enabling direct integration with Claude Code and other MCP clients
//! that use stdio transport.

use rmcp::{ServiceExt, transport::stdio};
use ruv_swarm_mcp::service::RealSwarmService;
use ruv_swarm_mcp::orchestrator::SwarmOrchestrator;
use tracing_subscriber::{EnvFilter};
use std::sync::Arc;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging - all logs go to stderr to keep stdout clean for MCP
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .with_writer(std::io::stderr)  // Critical: logs to stderr, not stdout
        .with_ansi(false)
        .init();

    tracing::info!("Starting ruv-swarm-mcp stdio server");

    // Create the orchestrator and service
    let orchestrator = Arc::new(SwarmOrchestrator::new().await);
    let service = RealSwarmService::new(orchestrator);
    
    // Create and run the server with STDIO transport
    let server = service.serve(stdio()).await.inspect_err(|e| {
        tracing::error!("Error starting server: {}", e);
    })?;

    server.waiting().await?;

    Ok(())
}