//! Example MCP server for RUV-Swarm

use std::sync::Arc;

use ruv_swarm_core::SwarmConfig;
use ruv_swarm_mcp::{orchestrator::SwarmOrchestrator, McpConfig, McpServer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ruv_swarm_mcp=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Create swarm config
    let swarm_config = SwarmConfig::default();

    // Create orchestrator
    let orchestrator = Arc::new(SwarmOrchestrator::new(swarm_config));

    // Create MCP config
    let mcp_config = McpConfig {
        bind_addr: "127.0.0.1:3000".parse()?,
        max_connections: 100,
        request_timeout_secs: 300,
        debug: true,
        allowed_origins: vec![], // Default to localhost only
    };

    // Create and start MCP server
    let server = McpServer::new(orchestrator, mcp_config);

    tracing::info!("Starting RUV-Swarm MCP server on http://127.0.0.1:3000");
    tracing::info!("WebSocket endpoint: ws://127.0.0.1:3000/mcp");
    tracing::info!("Available tools: http://127.0.0.1:3000/tools");

    server.start().await?;

    Ok(())
}
