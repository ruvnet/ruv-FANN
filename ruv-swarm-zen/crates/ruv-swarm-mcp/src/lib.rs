//! RUV-Swarm MCP (Model Context Protocol) Server
//!
//! This crate provides a comprehensive MCP server implementation for RUV-Swarm,
//! enabling Claude Code and other MCP-compatible clients to interact with the
//! swarm orchestration system through a standardized JSON-RPC 2.0 interface.
//!
//! ## Features
//!
//! - **Complete MCP Implementation**: Full JSON-RPC 2.0 and WebSocket support
//! - **11 Comprehensive Tools**: Agent spawning, task orchestration, monitoring
//! - **Real-time Event Streaming**: Live updates on swarm activity
//! - **Session Management**: Secure session handling with metadata support
//! - **Performance Metrics**: Built-in performance monitoring and optimization
//! - **Extensible Architecture**: Easy to add new tools and capabilities
//!
//! ## Quick Start
//!
//! ```rust
//! use std::sync::Arc;
//! use ruv_swarm_core::SwarmConfig;
//! use ruv_swarm_mcp::{orchestrator::SwarmOrchestrator, McpConfig, McpServer};
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! // Create swarm orchestrator
//! let orchestrator = Arc::new(SwarmOrchestrator::new(SwarmConfig::default()));
//!
//! // Configure MCP server
//! let config = McpConfig::default();
//!
//! // Create and start server
//! let server = McpServer::new(orchestrator, config);
//! // server.start().await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Available Tools
//!
//! The server provides the following MCP tools:
//!
//! - `ruv-swarm.spawn` - Spawn new agents
//! - `ruv-swarm.orchestrate` - Orchestrate complex tasks
//! - `ruv-swarm.query` - Query swarm state
//! - `ruv-swarm.monitor` - Monitor swarm activity
//! - `ruv-swarm.optimize` - Optimize performance
//! - `ruv-swarm.memory.store` - Store session data
//! - `ruv-swarm.memory.get` - Retrieve session data
//! - `ruv-swarm.task.create` - Create new tasks
//! - `ruv-swarm.workflow.execute` - Execute workflows
//! - `ruv-swarm.agent.list` - List active agents
//! - `ruv-swarm.agent.metrics` - Get agent metrics

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    extract::{State, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{mpsc, RwLock};
use tower_http::cors::CorsLayer;
use tracing::{debug, error, info};
use uuid::Uuid;

pub mod error;
// pub mod handlers;  // Temporarily disabled for simple service test
// pub mod limits;    // Temporarily disabled for simple service test
pub mod orchestrator;
pub mod service;
// pub mod tools;     // Temporarily disabled for simple service test
pub mod types;
// pub mod validation;   // Temporarily disabled for simple service test

use crate::orchestrator::SwarmOrchestrator;

// use crate::handlers::RequestHandler;  // Temporarily disabled
// use crate::limits::{ResourceLimiter, ResourceLimits};  // Temporarily disabled
// use crate::tools::ToolRegistry;  // Temporarily disabled

/*
/// MCP Server configuration
/// 
/// This struct defines the configuration options for the MCP server,
/// including network settings, connection limits, and debug options.
/// 
/// # Example
/// 
/// ```rust
/// use ruv_swarm_mcp::McpConfig;
/// 
/// let config = McpConfig {
///     bind_addr: "127.0.0.1:3000".parse().unwrap(),
///     max_connections: 100,
///     request_timeout_secs: 300,
///     debug: true,
///     allowed_origins: vec!["https://claude.ai".to_string()],
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    /// Server bind address (IP and port)
    pub bind_addr: SocketAddr,
    /// Maximum concurrent connections allowed
    pub max_connections: usize,
    /// Request timeout in seconds (300 = 5 minutes)
    pub request_timeout_secs: u64,
    /// Enable debug logging for troubleshooting
    pub debug: bool,
    /// Allowed CORS origins (empty = localhost only)
    #[serde(default)]
    pub allowed_origins: Vec<String>,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:3000".parse().unwrap(),
            max_connections: 100,
            request_timeout_secs: 300,
            debug: false,
            allowed_origins: vec![],
        }
    }
}

/// MCP Server state
pub struct McpServerState {
    /// Swarm orchestrator instance
    orchestrator: Arc<SwarmOrchestrator>,
    /// Tool registry
    tools: Arc<ToolRegistry>,
    /// Active sessions
    sessions: Arc<DashMap<Uuid, Arc<Session>>>,
    /// Resource limiter
    limiter: Arc<ResourceLimiter>,
    /// Server configuration
    config: McpConfig,
}

/// Client session
pub struct Session {
    pub id: Uuid,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_activity: RwLock<chrono::DateTime<chrono::Utc>>,
    pub metadata: DashMap<String, Value>,
}

/// MCP Server
/// 
/// The main MCP server that handles WebSocket connections and JSON-RPC requests.
/// This server implements the Model Context Protocol specification and provides
/// access to swarm orchestration capabilities through standardized tools.
/// 
/// # Example
/// 
/// ```rust
/// use std::sync::Arc;
/// use ruv_swarm_core::SwarmConfig;
/// use ruv_swarm_mcp::{orchestrator::SwarmOrchestrator, McpConfig, McpServer};
/// 
/// # #[tokio::main]
/// # async fn main() -> anyhow::Result<()> {
/// let orchestrator = Arc::new(SwarmOrchestrator::new(SwarmConfig::default()));
/// let config = McpConfig::default();
/// let server = McpServer::new(orchestrator, config);
/// 
/// // Start the server
/// // server.start().await?;
/// # Ok(())
/// # }
/// ```
pub struct McpServer {
    state: Arc<McpServerState>,
}

impl McpServer {
    /// Create a new MCP server
    /// 
    /// Creates a new MCP server instance with the provided orchestrator and configuration.
    /// The server will automatically register all available tools and initialize the
    /// session management system.
    /// 
    /// # Arguments
    /// 
    /// * `orchestrator` - The swarm orchestrator instance to use
    /// * `config` - Server configuration options
    /// 
    /// # Returns
    /// 
    /// A new `McpServer` instance ready to start serving requests
    pub fn new(orchestrator: Arc<SwarmOrchestrator>, config: McpConfig) -> Self {
        let tools = Arc::new(ToolRegistry::new());

        // Register all tools
        tools::register_tools(&tools);

        // Create resource limiter with default limits
        let limiter = Arc::new(ResourceLimiter::new(ResourceLimits::default()));

        let state = Arc::new(McpServerState {
            orchestrator,
            tools,
            sessions: Arc::new(DashMap::new()),
            limiter,
            config,
        });

        Self { state }
    }

    /// Start the MCP server
    /// 
    /// Starts the MCP server and begins listening for connections on the configured
    /// bind address. This method will block until the server is stopped.
    /// 
    /// # Returns
    /// 
    /// Returns `Ok(())` if the server starts successfully, or an error if there's
    /// an issue binding to the address or starting the server.
    /// 
    /// # Example
    /// 
    /// ```rust,no_run
    /// # use std::sync::Arc;
    /// # use ruv_swarm_core::SwarmConfig;
    /// # use ruv_swarm_mcp::{orchestrator::SwarmOrchestrator, McpConfig, McpServer};
    /// # #[tokio::main]
    /// # async fn main() -> anyhow::Result<()> {
    /// let orchestrator = Arc::new(SwarmOrchestrator::new(SwarmConfig::default()));
    /// let config = McpConfig::default();
    /// let server = McpServer::new(orchestrator, config);
    /// 
    /// // This will block until the server is stopped
    /// server.start().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn start(&self) -> anyhow::Result<()> {
        let app = self.build_router();
        let addr = self.state.config.bind_addr;

        info!("Starting MCP server on {}", addr);

        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }

    /// Build the router
    fn build_router(&self) -> Router {
        Router::new()
            .route("/", get(root_handler))
            .route("/mcp", get(websocket_handler))
            .route("/tools", get(list_tools_handler))
            .route("/health", get(health_handler))
            .layer(self.build_cors_layer())
            .with_state(self.state.clone())
    }

    /// Build secure CORS layer with restrictive settings
    fn build_cors_layer(&self) -> CorsLayer {
        let mut cors = CorsLayer::new();

        // Configure allowed origins
        let origins: Vec<axum::http::HeaderValue> = if self.state.config.allowed_origins.is_empty() {
            // Default to localhost only if no origins specified
            vec![
                "http://localhost:3000".parse().unwrap(),
                "http://localhost:8080".parse().unwrap(),
                "http://127.0.0.1:3000".parse().unwrap(),
                "http://127.0.0.1:8080".parse().unwrap(),
            ]
        } else {
            // Use configured origins
            self.state.config.allowed_origins
                .iter()
                .filter_map(|origin| origin.parse().ok())
                .collect()
        };

        cors = cors.allow_origin(origins)
            .allow_methods([
                axum::http::Method::GET,
                axum::http::Method::POST,
                axum::http::Method::OPTIONS,
            ])
            .allow_headers([
                axum::http::header::CONTENT_TYPE,
                axum::http::header::AUTHORIZATION,
                axum::http::header::ACCEPT,
            ])
            .allow_credentials(true)
            .max_age(std::time::Duration::from_secs(86400)); // 24 hours

        cors
    }
}

/// Root handler
async fn root_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "name": "ruv-swarm-mcp",
        "version": env!("CARGO_PKG_VERSION"),
        "protocol": "mcp/1.0",
        "endpoints": {
            "websocket": "/mcp",
            "tools": "/tools",
            "health": "/health"
        }
    }))
}

/// Health check handler
async fn health_handler(State(state): State<Arc<McpServerState>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "active_sessions": state.sessions.len(),
        "tools_count": state.tools.count(),
        "timestamp": chrono::Utc::now()
    }))
}

/// List available tools
async fn list_tools_handler(State(state): State<Arc<McpServerState>>) -> impl IntoResponse {
    let tools = state.tools.list_tools();
    Json(serde_json::json!({
        "tools": tools,
        "count": tools.len()
    }))
}

/// WebSocket handler for MCP protocol
async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<McpServerState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

/// Handle WebSocket connection
async fn handle_socket(socket: axum::extract::ws::WebSocket, state: Arc<McpServerState>) {
    let session_id = Uuid::new_v4();
    let session = Arc::new(Session {
        id: session_id,
        created_at: chrono::Utc::now(),
        last_activity: RwLock::new(chrono::Utc::now()),
        metadata: DashMap::new(),
    });

    state.sessions.insert(session_id, session.clone());
    
    // Initialize resource tracking for this session
    state.limiter.init_session(session_id).await;
    
    info!("New MCP session: {}", session_id);

    let (mut sender, mut receiver) = socket.split();
    let (tx, mut rx) = mpsc::channel(100);

    // Spawn task to handle outgoing messages
    let tx_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if sender.send(msg).await.is_err() {
                break;
            }
        }
    });

    // Create request handler
    let handler = RequestHandler::new(
        state.orchestrator.clone(),
        state.tools.clone(),
        session.clone(),
        state.limiter.clone(),
        tx.clone(),
    );

    // Handle incoming messages
    while let Some(Ok(msg)) = receiver.next().await {
        if let axum::extract::ws::Message::Text(text) = msg {
            match serde_json::from_str::<McpRequest>(&text) {
                Ok(request) => {
                    debug!("Received MCP request: {:?}", request.method);

                    // Update last activity
                    *session.last_activity.write().await = chrono::Utc::now();

                    // Handle request
                    match handler.handle_request(request).await {
                        Ok(response) => {
                            if let Ok(json) = serde_json::to_string(&response) {
                                let _ = tx.send(axum::extract::ws::Message::Text(json)).await;
                            }
                        }
                        Err(e) => {
                            let sanitized_error = crate::error::log_and_sanitize_error(
                                &e,
                                &session_id,
                                state.config.debug,
                            );
                            let error_response =
                                McpResponse::error(None, -32603, sanitized_error);
                            if let Ok(json) = serde_json::to_string(&error_response) {
                                let _ = tx.send(axum::extract::ws::Message::Text(json)).await;
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to parse MCP request: {}", e);
                    let error_response =
                        McpResponse::error(None, -32700, "Parse error".to_string());
                    if let Ok(json) = serde_json::to_string(&error_response) {
                        let _ = tx.send(axum::extract::ws::Message::Text(json)).await;
                    }
                }
            }
        }
    }

    // Cleanup
    tx_task.abort();
    state.sessions.remove(&session_id);
    state.limiter.remove_session(&session_id).await;
    info!("MCP session closed: {}", session_id);
}

/// MCP Request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpRequest {
    pub jsonrpc: String,
    pub method: String,
    pub params: Option<Value>,
    pub id: Option<Value>,
}

/// MCP Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<McpError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
}

impl McpResponse {
    pub fn success(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            result: Some(result),
            error: None,
            id,
        }
    }

    pub fn error(id: Option<Value>, code: i32, message: String) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            result: None,
            error: Some(McpError {
                code,
                message,
                data: None,
            }),
            id,
        }
    }
}

/// MCP Error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

*/

#[cfg(test)]
mod tests;
