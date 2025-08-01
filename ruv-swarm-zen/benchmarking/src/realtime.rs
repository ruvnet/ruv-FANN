//! Real-time monitoring module for benchmark execution

use anyhow::Result;
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, RwLock};
use tokio::time::interval;
use tower_http::cors::CorsLayer;
use tracing::{debug, info};

use crate::stream_parser::ClaudeStreamEvent;

use futures::{SinkExt, StreamExt};

/// Real-time monitoring system
pub struct RealTimeMonitor {
    server: MonitoringServer,
    event_channel: broadcast::Sender<RealTimeEvent>,
    active_runs: Arc<DashMap<String, RunMonitor>>,
}

impl RealTimeMonitor {
    /// Create a new real-time monitor
    pub async fn new(port: u16) -> Result<Self> {
        let (tx, _) = broadcast::channel(1024);
        let active_runs = Arc::new(DashMap::new());

        let server = MonitoringServer::new(port, tx.clone(), active_runs.clone());

        Ok(Self {
            server,
            event_channel: tx,
            active_runs,
        })
    }

    /// Start monitoring a benchmark run
    pub async fn start_monitoring(&self, run_id: &str) -> Result<()> {
        let monitor = RunMonitor::new(run_id);
        self.active_runs.insert(run_id.to_string(), monitor);

        // Send start event
        let event = RealTimeEvent::RunStarted {
            run_id: run_id.to_string(),
            timestamp: Utc::now(),
        };

        let _ = self.event_channel.send(event);

        info!("Started monitoring run: {}", run_id);
        Ok(())
    }

    /// Stop monitoring a benchmark run
    pub async fn stop_monitoring(&self, run_id: &str) -> Result<()> {
        if let Some((_, monitor)) = self.active_runs.remove(run_id) {
            // Send stop event
            let event = RealTimeEvent::RunCompleted {
                run_id: run_id.to_string(),
                duration: monitor.start_time.elapsed(),
                timestamp: Utc::now(),
            };

            let _ = self.event_channel.send(event);

            info!("Stopped monitoring run: {}", run_id);
        }

        Ok(())
    }

    /// Update metrics for a run
    pub async fn update_metrics(&self, run_id: &str, metrics: MetricsUpdate) -> Result<()> {
        if let Some(mut monitor) = self.active_runs.get_mut(run_id) {
            monitor.update_metrics(metrics.clone());

            // Broadcast update
            let event = RealTimeEvent::MetricsUpdate {
                run_id: run_id.to_string(),
                metrics,
                timestamp: Utc::now(),
            };

            let _ = self.event_channel.send(event);
        }

        Ok(())
    }

    /// Process stream event
    pub async fn process_stream_event(&self, run_id: &str, event: ClaudeStreamEvent) -> Result<()> {
        if let Some(mut monitor) = self.active_runs.get_mut(run_id) {
            monitor.add_stream_event(event.clone());

            // Broadcast stream event
            let rt_event = RealTimeEvent::StreamEvent {
                run_id: run_id.to_string(),
                event,
                timestamp: Utc::now(),
            };

            let _ = self.event_channel.send(rt_event);
        }

        Ok(())
    }

    /// Start the monitoring server
    pub async fn start_server(self) -> Result<()> {
        self.server.start().await
    }
}

/// Individual run monitor
#[derive(Debug, Clone)]
struct RunMonitor {
    run_id: String,
    start_time: Instant,
    latest_metrics: Option<MetricsUpdate>,
    stream_events: Vec<ClaudeStreamEvent>,
    event_count: u64,
}

impl RunMonitor {
    fn new(run_id: &str) -> Self {
        Self {
            run_id: run_id.to_string(),
            start_time: Instant::now(),
            latest_metrics: None,
            stream_events: Vec::new(),
            event_count: 0,
        }
    }

    fn update_metrics(&mut self, metrics: MetricsUpdate) {
        self.latest_metrics = Some(metrics);
    }

    fn add_stream_event(&mut self, event: ClaudeStreamEvent) {
        self.stream_events.push(event);
        self.event_count += 1;

        // Keep only last 1000 events to prevent memory growth
        if self.stream_events.len() > 1000 {
            self.stream_events.remove(0);
        }
    }
}

/// Real-time event types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RealTimeEvent {
    RunStarted {
        run_id: String,
        timestamp: DateTime<Utc>,
    },
    RunCompleted {
        run_id: String,
        duration: Duration,
        timestamp: DateTime<Utc>,
    },
    MetricsUpdate {
        run_id: String,
        metrics: MetricsUpdate,
        timestamp: DateTime<Utc>,
    },
    StreamEvent {
        run_id: String,
        event: ClaudeStreamEvent,
        timestamp: DateTime<Utc>,
    },
    Alert {
        run_id: String,
        level: AlertLevel,
        message: String,
        timestamp: DateTime<Utc>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsUpdate {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub active_agents: u32,
    pub messages_passed: u32,
    pub tool_invocations: u32,
    pub thinking_time_ms: u64,
    pub errors_recovered: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertLevel {
    Info,
    Warning,
    Error,
    Critical,
}

/// Monitoring HTTP/WebSocket server
pub struct MonitoringServer {
    port: u16,
    event_broadcaster: broadcast::Sender<RealTimeEvent>,
    active_runs: Arc<DashMap<String, RunMonitor>>,
}

impl MonitoringServer {
    fn new(
        port: u16,
        event_broadcaster: broadcast::Sender<RealTimeEvent>,
        active_runs: Arc<DashMap<String, RunMonitor>>,
    ) -> Self {
        Self {
            port,
            event_broadcaster,
            active_runs,
        }
    }

    /// Start the monitoring server
    pub async fn start(self) -> Result<()> {
        let addr = SocketAddr::from(([0, 0, 0, 0], self.port));
        let app = self.create_router();

        info!("Starting monitoring server on {}", addr);

        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }

    fn create_router(self) -> Router {
        let shared_state = Arc::new(ServerState {
            event_broadcaster: self.event_broadcaster,
            active_runs: self.active_runs,
        });

        Router::new()
            .route("/", get(index_handler))
            .route("/ws", get(websocket_handler))
            .route("/api/runs", get(list_runs_handler))
            .route("/api/runs/:run_id", get(get_run_handler))
            .route("/api/runs/:run_id/metrics", get(get_metrics_handler))
            .route("/api/alert", post(create_alert_handler))
            .layer(CorsLayer::permissive())
            .with_state(shared_state)
    }
}

#[derive(Clone)]
struct ServerState {
    event_broadcaster: broadcast::Sender<RealTimeEvent>,
    active_runs: Arc<DashMap<String, RunMonitor>>,
}

// HTTP handlers
async fn index_handler() -> impl IntoResponse {
    axum::response::Html(include_str!("../static/monitor.html"))
}

async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<ServerState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| websocket_connection(socket, state))
}

async fn websocket_connection(socket: WebSocket, state: Arc<ServerState>) {
    let mut rx = state.event_broadcaster.subscribe();
    let (mut sender, mut receiver) = socket.split();

    // Task to send events to client
    let send_task = tokio::spawn(async move {
        while let Ok(event) = rx.recv().await {
            if let Ok(json) = serde_json::to_string(&event) {
                if sender.send(Message::Text(json)).await.is_err() {
                    break;
                }
            }
        }
    });

    // Task to receive messages from client (for ping/pong)
    let recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            match msg {
                Message::Text(text) => {
                    debug!("Received text message: {}", text);
                }
                Message::Close(_) => {
                    debug!("Client disconnected");
                    break;
                }
                _ => {}
            }
        }
    });

    // Wait for either task to complete
    tokio::select! {
        _ = send_task => {},
        _ = recv_task => {},
    }
}

async fn list_runs_handler(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let runs: Vec<RunSummary> = state
        .active_runs
        .iter()
        .map(|entry| {
            let monitor = entry.value();
            RunSummary {
                run_id: monitor.run_id.clone(),
                elapsed: monitor.start_time.elapsed(),
                event_count: monitor.event_count,
                has_metrics: monitor.latest_metrics.is_some(),
            }
        })
        .collect();

    Json(runs)
}

async fn get_run_handler(
    axum::extract::Path(run_id): axum::extract::Path<String>,
    State(state): State<Arc<ServerState>>,
) -> impl IntoResponse {
    if let Some(monitor) = state.active_runs.get(&run_id) {
        let details = RunDetails {
            run_id: monitor.run_id.clone(),
            start_time: monitor.start_time.elapsed(),
            latest_metrics: monitor.latest_metrics.clone(),
            recent_events: monitor
                .stream_events
                .iter()
                .rev()
                .take(50)
                .cloned()
                .collect(),
            event_count: monitor.event_count,
        };

        Json(Some(details))
    } else {
        Json(None)
    }
}

async fn get_metrics_handler(
    axum::extract::Path(run_id): axum::extract::Path<String>,
    State(state): State<Arc<ServerState>>,
) -> impl IntoResponse {
    if let Some(monitor) = state.active_runs.get(&run_id) {
        Json(monitor.latest_metrics.clone())
    } else {
        Json(None)
    }
}

async fn create_alert_handler(
    State(state): State<Arc<ServerState>>,
    Json(alert): Json<AlertRequest>,
) -> impl IntoResponse {
    let event = RealTimeEvent::Alert {
        run_id: alert.run_id,
        level: alert.level,
        message: alert.message,
        timestamp: Utc::now(),
    };

    let _ = state.event_broadcaster.send(event);

    Json(AlertResponse { success: true })
}

// API types
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RunSummary {
    run_id: String,
    elapsed: Duration,
    event_count: u64,
    has_metrics: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RunDetails {
    run_id: String,
    start_time: Duration,
    latest_metrics: Option<MetricsUpdate>,
    recent_events: Vec<ClaudeStreamEvent>,
    event_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AlertRequest {
    run_id: String,
    level: AlertLevel,
    message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AlertResponse {
    success: bool,
}

/// Real-time metrics aggregator
pub struct MetricsAggregator {
    interval: Duration,
    metrics_buffer: Arc<RwLock<Vec<MetricsSnapshot>>>,
}

impl MetricsAggregator {
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            metrics_buffer: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn start_aggregation(&self) {
        let buffer = self.metrics_buffer.clone();
        let mut ticker = interval(self.interval);

        tokio::spawn(async move {
            loop {
                ticker.tick().await;

                // Aggregate metrics from all active runs
                let snapshot = MetricsSnapshot {
                    timestamp: Utc::now(),
                    total_cpu: 0.0,
                    total_memory: 0.0,
                    active_runs: 0,
                    total_events: 0,
                };

                let mut buffer = buffer.write().await;
                buffer.push(snapshot);

                // Keep only last hour of data
                let cutoff = Utc::now() - chrono::Duration::hours(1);
                buffer.retain(|s| s.timestamp > cutoff);
            }
        });
    }
}

#[derive(Debug, Clone)]
struct MetricsSnapshot {
    timestamp: DateTime<Utc>,
    total_cpu: f64,
    total_memory: f64,
    active_runs: u32,
    total_events: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_monitor_creation() {
        let monitor = RealTimeMonitor::new(0).await;
        assert!(monitor.is_ok());
    }

    #[tokio::test]
    async fn test_run_monitoring() {
        let monitor = RealTimeMonitor::new(0).await.unwrap();

        // Start monitoring
        monitor.start_monitoring("test-run").await.unwrap();

        // Check run exists
        assert!(monitor.active_runs.contains_key("test-run"));

        // Stop monitoring
        monitor.stop_monitoring("test-run").await.unwrap();

        // Check run removed
        assert!(!monitor.active_runs.contains_key("test-run"));
    }

    #[tokio::test]
    async fn test_metrics_update() {
        let monitor = RealTimeMonitor::new(0).await.unwrap();

        monitor.start_monitoring("test-run").await.unwrap();

        let metrics = MetricsUpdate {
            cpu_usage: 50.0,
            memory_usage: 1024.0,
            active_agents: 5,
            messages_passed: 100,
            tool_invocations: 20,
            thinking_time_ms: 5000,
            errors_recovered: 2,
        };

        monitor.update_metrics("test-run", metrics).await.unwrap();

        // Verify metrics were stored
        let run = monitor.active_runs.get("test-run").unwrap();
        assert!(run.latest_metrics.is_some());
    }
}
