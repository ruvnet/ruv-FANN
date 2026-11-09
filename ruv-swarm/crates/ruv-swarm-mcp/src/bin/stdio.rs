//! Stdio transport binary for Claude Code integration
//!
//! This binary provides MCP server functionality over stdin/stdout,
//! enabling direct integration with Claude Code and other MCP clients
//! that use stdio transport.

use rmcp::{
    transport::{async_rw::AsyncRwTransport, stdio_pipes},
    ServiceExt,
};
use ruv_swarm_core::SwarmConfig;
use ruv_swarm_mcp::orchestrator::SwarmOrchestrator;
use ruv_swarm_mcp::service::{RealSwarmService, MAX_STDIO_MESSAGE_BYTES};
use std::sync::Arc;
use tracing_subscriber::EnvFilter;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging - all logs go to stderr to keep stdout clean for MCP
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .with_writer(std::io::stderr) // Critical: logs to stderr, not stdout
        .with_ansi(false)
        .init();

    tracing::info!("Starting ruv-swarm-mcp stdio server");

    // Create the orchestrator and service
    let swarm_config = SwarmConfig::default();
    let orchestrator = Arc::new(SwarmOrchestrator::new(swarm_config).await);
    let service = RealSwarmService::new(orchestrator);

    // Bound each incoming line before buffering; recover after discarding an oversized line.
    let (input, output) = stdio_pipes();
    let transport =
        AsyncRwTransport::new_server(input, output).with_max_line_length(MAX_STDIO_MESSAGE_BYTES);
    let server = service.serve(transport).await.inspect_err(|e| {
        tracing::error!("Error starting server: {}", e);
    })?;

    server.waiting().await?;

    Ok(())
}
