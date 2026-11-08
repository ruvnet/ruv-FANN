//! Example MCP server for RUV-Swarm

use std::{net::SocketAddr, sync::Arc};

use anyhow::Context;
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

    let configured_bind_addr = match std::env::var("RUV_SWARM_MCP_BIND_ADDR") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(error).context("Failed to read RUV_SWARM_MCP_BIND_ADDR"),
    };
    let bind_addr = parse_bind_addr(configured_bind_addr.as_deref())?;

    // Create swarm config
    let swarm_config = SwarmConfig::default();

    // Create orchestrator
    let orchestrator = Arc::new(SwarmOrchestrator::new(swarm_config).await);

    // Create MCP config
    let mcp_config = McpConfig {
        bind_addr,
        max_connections: 100,
        request_timeout_secs: 300,
        debug: true,
        allowed_origins: vec![], // Default to localhost only
    };

    // Create and start MCP server
    let server = McpServer::new(orchestrator, mcp_config);

    tracing::info!("Starting RUV-Swarm MCP server on http://{}", bind_addr);
    tracing::info!("WebSocket endpoint: ws://{}/mcp", bind_addr);
    tracing::info!("Available tools: http://{}/tools", bind_addr);

    server.start().await?;

    Ok(())
}

fn parse_bind_addr(value: Option<&str>) -> anyhow::Result<SocketAddr> {
    value
        .unwrap_or("127.0.0.1:3000")
        .parse()
        .context("RUV_SWARM_MCP_BIND_ADDR must be an IP socket address, such as 127.0.0.1:3000")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_default_and_isolated_bind_addresses() {
        assert_eq!(parse_bind_addr(None).unwrap().to_string(), "127.0.0.1:3000");
        assert_eq!(parse_bind_addr(Some("127.0.0.1:0")).unwrap().port(), 0);
        assert_eq!(parse_bind_addr(Some("[::1]:4321")).unwrap().port(), 4321);
    }

    #[test]
    fn rejects_malformed_configured_addresses() {
        for value in ["", "localhost:3000", "127.0.0.1", "127.0.0.1:65536"] {
            let error = parse_bind_addr(Some(value)).unwrap_err();
            assert!(error.to_string().contains("RUV_SWARM_MCP_BIND_ADDR"));
        }
    }
}
