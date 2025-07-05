//! SCell Manager gRPC Server

use clap::{Arg, Command};
use log::{error, info};
use scell_manager::{
    config::SCellManagerConfig,
    metrics::MetricsCollector,
    prediction::PredictionEngine,
    service::{create_server, SCellManagerService},
    SCellManager,
};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::signal;
use tokio::sync::RwLock;
use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();
    
    // Parse command line arguments
    let matches = Command::new("scell_manager_server")
        .version(env!("CARGO_PKG_VERSION"))
        .about("SCell Manager gRPC Server - Predictive Carrier Aggregation for RAN Intelligence")
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("FILE")
                .help("Configuration file path")
                .default_value("config.json")
        )
        .arg(
            Arg::new("bind")
                .short('b')
                .long("bind")
                .value_name("ADDRESS")
                .help("Bind address (e.g., 0.0.0.0:50051)")
        )
        .arg(
            Arg::new("data-dir")
                .short('d')
                .long("data-dir")
                .value_name("PATH")
                .help("Data directory path")
        )
        .arg(
            Arg::new("model-dir")
                .short('m')
                .long("model-dir")
                .value_name("PATH")
                .help("Model directory path")
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .action(clap::ArgAction::SetTrue)
                .help("Enable verbose logging")
        )
        .get_matches();
    
    // Load configuration
    let config_path = matches.get_one::<String>("config").unwrap();
    let mut config = if std::path::Path::new(config_path).exists() {
        SCellManagerConfig::from_file(config_path)?
    } else {
        info!("Config file not found, using defaults and environment variables");
        SCellManagerConfig::from_env()?
    };
    
    // Override config with command line arguments
    if let Some(bind_addr) = matches.get_one::<String>("bind") {
        config.server.bind_address = bind_addr.parse()?;
    }
    
    if let Some(data_dir) = matches.get_one::<String>("data-dir") {
        config.data_config.data_dir = data_dir.into();
    }
    
    if let Some(model_dir) = matches.get_one::<String>("model-dir") {
        config.model_config.model_dir = model_dir.into();
    }
    
    // Validate configuration
    config.validate()?;
    
    info!("Starting SCell Manager Server v{}", env!("CARGO_PKG_VERSION"));
    info!("Configuration loaded from: {}", config_path);
    info!("Server will bind to: {}", config.server.bind_address);
    info!("Data directory: {:?}", config.data_config.data_dir);
    info!("Model directory: {:?}", config.model_config.model_dir);
    
    // Create directories
    std::fs::create_dir_all(&config.data_config.data_dir)?;
    std::fs::create_dir_all(&config.model_config.model_dir)?;
    
    // Initialize SCell Manager
    let scell_manager = SCellManager::new(config.clone()).await?;
    
    // Initialize metrics collector
    let metrics_collector = scell_manager.metrics_collector();
    
    // Create gRPC service
    let service = SCellManagerService::new(
        scell_manager.prediction_engine(),
        metrics_collector.clone(),
        config.clone(),
    );
    
    let server_service = create_server(service);
    
    // Start Prometheus metrics server if enabled
    let _metrics_handle = if config.metrics_config.enable_prometheus {
        let metrics_port = config.metrics_config.prometheus_port;
        let metrics_collector_clone = metrics_collector.clone();
        
        Some(tokio::spawn(async move {
            start_metrics_server(metrics_port, metrics_collector_clone).await;
        }))
    } else {
        None
    };
    
    // Start health check updater
    let health_metrics_collector = metrics_collector.clone();
    let _health_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(
            std::time::Duration::from_secs(30)
        );
        
        loop {
            interval.tick().await;
            
            // Update system metrics
            let memory_usage = get_memory_usage();
            let cpu_usage = get_cpu_usage();
            
            health_metrics_collector.update_system_metrics(memory_usage, cpu_usage);
        }
    });
    
    // Start the gRPC server
    info!("SCell Manager gRPC server starting...");
    
    let listener = TcpListener::bind(&config.server.bind_address).await?;
    info!("Server listening on: {}", config.server.bind_address);
    
    // Graceful shutdown handling
    let server_future = Server::builder()
        .add_service(server_service)
        .serve_with_incoming_shutdown(
            tokio_stream::wrappers::TcpListenerStream::new(listener),
            async {
                signal::ctrl_c().await.expect("Failed to listen for ctrl-c");
                info!("Received shutdown signal, stopping server...");
            }
        );
    
    // Run the server
    if let Err(e) = server_future.await {
        error!("Server error: {}", e);
        return Err(Box::new(e));
    }
    
    info!("SCell Manager server stopped");
    Ok(())
}

/// Start Prometheus metrics HTTP server
async fn start_metrics_server(port: u16, metrics_collector: Arc<MetricsCollector>) {
    use warp::Filter;
    
    let metrics_route = warp::path("metrics")
        .and(warp::get())
        .map(move || {
            match metrics_collector.export_prometheus() {
                Ok(metrics) => warp::reply::with_status(metrics, warp::http::StatusCode::OK),
                Err(e) => {
                    error!("Failed to export metrics: {}", e);
                    warp::reply::with_status(
                        format!("Error: {}", e),
                        warp::http::StatusCode::INTERNAL_SERVER_ERROR
                    )
                }
            }
        });
    
    let health_route = warp::path("health")
        .and(warp::get())
        .map(|| {
            warp::reply::json(&serde_json::json!({
                "status": "healthy",
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "service": "scell_manager"
            }))
        });
    
    let routes = metrics_route.or(health_route);
    
    info!("Prometheus metrics server starting on port {}", port);
    
    warp::serve(routes)
        .run(([0, 0, 0, 0], port))
        .await;
}

/// Get current memory usage (placeholder implementation)
fn get_memory_usage() -> f64 {
    // In a real implementation, you would use system APIs
    // to get actual memory usage. For now, return a placeholder.
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/self/status") {
            for line in contents.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<f64>() {
                            return kb * 1024.0; // Convert KB to bytes
                        }
                    }
                }
            }
        }
    }
    
    // Fallback: estimate based on heap allocations (very rough)
    std::alloc::System.used_memory().unwrap_or(0) as f64
}

/// Get current CPU usage (placeholder implementation)
fn get_cpu_usage() -> f64 {
    // In a real implementation, you would use system APIs
    // to get actual CPU usage. For now, return a placeholder.
    
    #[cfg(target_os = "linux")]
    {
        // This is a simplified CPU usage calculation
        // In production, you'd want to use a more sophisticated method
        static mut LAST_IDLE: u64 = 0;
        static mut LAST_TOTAL: u64 = 0;
        
        if let Ok(contents) = std::fs::read_to_string("/proc/stat") {
            if let Some(line) = contents.lines().next() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 8 && parts[0] == "cpu" {
                    let user: u64 = parts[1].parse().unwrap_or(0);
                    let nice: u64 = parts[2].parse().unwrap_or(0);
                    let system: u64 = parts[3].parse().unwrap_or(0);
                    let idle: u64 = parts[4].parse().unwrap_or(0);
                    let iowait: u64 = parts[5].parse().unwrap_or(0);
                    let irq: u64 = parts[6].parse().unwrap_or(0);
                    let softirq: u64 = parts[7].parse().unwrap_or(0);
                    
                    let total = user + nice + system + idle + iowait + irq + softirq;
                    
                    unsafe {
                        if LAST_TOTAL > 0 {
                            let total_diff = total - LAST_TOTAL;
                            let idle_diff = idle - LAST_IDLE;
                            
                            if total_diff > 0 {
                                let cpu_usage = 100.0 * (1.0 - idle_diff as f64 / total_diff as f64);
                                LAST_IDLE = idle;
                                LAST_TOTAL = total;
                                return cpu_usage.max(0.0).min(100.0);
                            }
                        }
                        
                        LAST_IDLE = idle;
                        LAST_TOTAL = total;
                    }
                }
            }
        }
    }
    
    // Fallback: return 0% CPU usage
    0.0
}

// Custom allocator for memory tracking (optional)
#[cfg(feature = "memory_tracking")]
mod memory_tracking {
    use std::alloc::{GlobalAlloc, System, Layout};
    use std::sync::atomic::{AtomicUsize, Ordering};
    
    struct TrackingAllocator;
    
    static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
    
    unsafe impl GlobalAlloc for TrackingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ret = System.alloc(layout);
            if !ret.is_null() {
                ALLOCATED.fetch_add(layout.size(), Ordering::SeqCst);
            }
            ret
        }
        
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            System.dealloc(ptr, layout);
            ALLOCATED.fetch_sub(layout.size(), Ordering::SeqCst);
        }
    }
    
    impl TrackingAllocator {
        pub fn used_memory(&self) -> Option<usize> {
            Some(ALLOCATED.load(Ordering::SeqCst))
        }
    }
    
    #[global_allocator]
    static GLOBAL: TrackingAllocator = TrackingAllocator;
}

#[cfg(not(feature = "memory_tracking"))]
use std::alloc::System;

#[cfg(not(feature = "memory_tracking"))]
trait MemoryUsage {
    fn used_memory(&self) -> Option<usize>;
}

#[cfg(not(feature = "memory_tracking"))]
impl MemoryUsage for std::alloc::System {
    fn used_memory(&self) -> Option<usize> {
        None // Not available without tracking
    }
}