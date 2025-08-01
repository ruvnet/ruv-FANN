#!/usr/bin/env cargo run --bin daa-coordinator --

//! DAA Coordinator Binary
//!
//! Central coordination daemon for Decentralized Autonomous Agents (DAA)
//! Provides orchestration, monitoring, and cross-agent coordination services.

use clap::{Arg, Command};
use ruv_swarm_daa::*;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tokio::time::sleep;

/// DAA Coordinator main structure
pub struct DAACoordinator {
    agents: Arc<RwLock<HashMap<String, Box<dyn DAAAgent + Send + Sync>>>>,
    coordination_memory: Arc<Mutex<CoordinationMemory>>,
    learning_engine: Arc<Mutex<LearningEngine>>,
    neural_manager: Arc<Mutex<NeuralNetworkManager>>,
    active: Arc<Mutex<bool>>,
    metrics: Arc<Mutex<CoordinationMetrics>>,
}

#[derive(Debug, Clone)]
pub struct CoordinationMetrics {
    pub total_agents: usize,
    pub active_tasks: usize,
    pub completed_tasks: usize,
    pub failed_tasks: usize,
    pub average_response_time: f64,
    pub memory_usage: usize,
    pub uptime: Duration,
    pub coordination_events: usize,
}

impl Default for CoordinationMetrics {
    fn default() -> Self {
        Self {
            total_agents: 0,
            active_tasks: 0,
            completed_tasks: 0,
            failed_tasks: 0,
            average_response_time: 0.0,
            memory_usage: 0,
            uptime: Duration::from_secs(0),
            coordination_events: 0,
        }
    }
}

impl DAACoordinator {
    /// Initialize new DAA Coordinator
    pub async fn new() -> Result<Self, DAAError> {
        Ok(Self {
            agents: Arc::new(RwLock::new(HashMap::new())),
            coordination_memory: Arc::new(Mutex::new(CoordinationMemory::new())),
            learning_engine: Arc::new(Mutex::new(LearningEngine::new())),
            neural_manager: Arc::new(Mutex::new(NeuralNetworkManager::new())),
            active: Arc::new(Mutex::new(false)),
            metrics: Arc::new(Mutex::new(CoordinationMetrics::default())),
        })
    }

    /// Start the coordination service
    pub async fn start(&self) -> Result<(), DAAError> {
        let mut active = self.active.lock().await;
        *active = true;
        drop(active);

        println!("🚀 DAA Coordinator starting...");

        // Initialize subsystems
        self.initialize_coordination_layer().await?;
        self.start_monitoring_loop().await?;
        self.start_learning_adaptation().await?;

        println!("✅ DAA Coordinator started successfully");
        Ok(())
    }

    /// Stop the coordination service
    pub async fn stop(&self) -> Result<(), DAAError> {
        let mut active = self.active.lock().await;
        *active = false;
        drop(active);

        println!("🛑 DAA Coordinator stopping...");

        // Graceful shutdown of all agents
        let agents = self.agents.read().await;
        for (id, agent) in agents.iter() {
            if let Err(e) = agent.shutdown().await {
                eprintln!("⚠️  Error shutting down agent {}: {}", id, e);
            }
        }
        drop(agents);

        println!("✅ DAA Coordinator stopped");
        Ok(())
    }

    /// Register new DAA agent
    pub async fn register_agent(
        &self,
        agent: Box<dyn DAAAgent + Send + Sync>,
    ) -> Result<String, DAAError> {
        let agent_id = agent.get_id().await?;
        let mut agents = self.agents.write().await;
        agents.insert(agent_id.clone(), agent);
        drop(agents);

        let mut metrics = self.metrics.lock().await;
        metrics.total_agents += 1;
        drop(metrics);

        println!("📝 Registered agent: {}", agent_id);
        Ok(agent_id)
    }

    /// Unregister DAA agent
    pub async fn unregister_agent(&self, agent_id: &str) -> Result<(), DAAError> {
        let mut agents = self.agents.write().await;
        if let Some(agent) = agents.remove(agent_id) {
            agent.shutdown().await?;
            println!("🗑️  Unregistered agent: {}", agent_id);
        }
        drop(agents);

        let mut metrics = self.metrics.lock().await;
        if metrics.total_agents > 0 {
            metrics.total_agents -= 1;
        }
        drop(metrics);

        Ok(())
    }

    /// Coordinate task execution across agents
    pub async fn coordinate_task(&self, task: TaskRequest) -> Result<TaskResult, DAAError> {
        let start_time = Instant::now();

        // Find best agent for task
        let agent_id = self.select_optimal_agent(&task).await?;

        // Execute task
        let agents = self.agents.read().await;
        let agent = agents
            .get(&agent_id)
            .ok_or_else(|| DAAError::AgentNotFound {
                id: agent_id.clone(),
            })?;

        let result = agent.execute_task(task.clone()).await?;
        drop(agents);

        // Update metrics
        let mut metrics = self.metrics.lock().await;
        metrics.completed_tasks += 1;
        let response_time = start_time.elapsed().as_millis() as f64;
        metrics.average_response_time =
            (metrics.average_response_time * (metrics.completed_tasks - 1) as f64 + response_time)
                / metrics.completed_tasks as f64;
        drop(metrics);

        // Store coordination event
        let mut memory = self.coordination_memory.lock().await;
        memory
            .store_event(CoordinationEvent {
                timestamp: chrono::Utc::now(),
                event_type: CoordinationEventType::TaskAssignment,
                participants: vec![agent_id.clone()],
                outcome: json!({
                    "task_id": task.id.clone(),
                    "agent_id": agent_id.clone(),
                    "response_time_ms": response_time,
                    "success": true
                }),
            })
            .await?;
        drop(memory);

        Ok(result)
    }

    /// Get coordination status
    pub async fn get_status(&self) -> Result<Value, DAAError> {
        let agents = self.agents.read().await;
        let metrics = self.metrics.lock().await;

        let status = json!({
            "status": "active",
            "coordinator_version": env!("CARGO_PKG_VERSION"),
            "agents": {
                "total": agents.len(),
                "active": agents.len(),
                "types": self.get_agent_types(&agents).await
            },
            "metrics": {
                "completed_tasks": metrics.completed_tasks,
                "failed_tasks": metrics.failed_tasks,
                "average_response_time_ms": metrics.average_response_time,
                "uptime_seconds": metrics.uptime.as_secs(),
                "coordination_events": metrics.coordination_events
            },
            "memory": {
                "total_events": self.get_memory_size().await?,
                "usage_mb": metrics.memory_usage / 1024 / 1024
            },
            "learning": {
                "adaptation_enabled": true,
                "patterns_learned": self.get_learned_patterns_count().await?,
                "performance_improvement": self.get_performance_improvement().await?
            }
        });

        Ok(status)
    }

    /// Initialize coordination layer
    async fn initialize_coordination_layer(&self) -> Result<(), DAAError> {
        println!("🔧 Initializing coordination layer...");

        // Initialize neural coordination
        let mut neural_manager = self.neural_manager.lock().await;
        neural_manager.initialize_coordination_patterns().await?;
        drop(neural_manager);

        // Initialize learning engine
        let mut learning_engine = self.learning_engine.lock().await;
        learning_engine.initialize_meta_learning().await?;
        drop(learning_engine);

        println!("✅ Coordination layer initialized");
        Ok(())
    }

    /// Start monitoring loop
    async fn start_monitoring_loop(&self) -> Result<(), DAAError> {
        let active = Arc::clone(&self.active);
        let metrics = Arc::clone(&self.metrics);
        let agents = Arc::clone(&self.agents);

        tokio::spawn(async move {
            let start_time = Instant::now();

            while *active.lock().await {
                // Update metrics
                {
                    let mut m = metrics.lock().await;
                    m.uptime = start_time.elapsed();
                    m.memory_usage = Self::get_current_memory_usage();
                }

                // Health check agents
                {
                    let agents_guard = agents.read().await;
                    for (id, agent) in agents_guard.iter() {
                        if let Err(e) = agent.health_check().await {
                            eprintln!("⚠️  Agent {} health check failed: {}", id, e);
                        }
                    }
                }

                sleep(Duration::from_secs(10)).await;
            }
        });

        Ok(())
    }

    /// Start learning adaptation
    async fn start_learning_adaptation(&self) -> Result<(), DAAError> {
        let active = Arc::clone(&self.active);
        let learning_engine = Arc::clone(&self.learning_engine);
        let coordination_memory = Arc::clone(&self.coordination_memory);

        tokio::spawn(async move {
            while *active.lock().await {
                // Perform adaptive learning
                {
                    let mut engine = learning_engine.lock().await;
                    let memory = coordination_memory.lock().await;

                    if let Err(e) = engine
                        .adapt_from_events(&memory.get_recent_events(100).await.unwrap_or_default())
                        .await
                    {
                        eprintln!("⚠️  Learning adaptation failed: {}", e);
                    }
                }

                sleep(Duration::from_secs(30)).await;
            }
        });

        Ok(())
    }

    /// Select optimal agent for task
    async fn select_optimal_agent(&self, task: &TaskRequest) -> Result<String, DAAError> {
        let agents = self.agents.read().await;

        if agents.is_empty() {
            return Err(DAAError::NoAgentsAvailable);
        }

        // Simple selection for now - would use ML-based selection in production
        let agent_id = agents.keys().next().unwrap().clone();
        Ok(agent_id)
    }

    /// Get agent types distribution
    async fn get_agent_types(
        &self,
        agents: &HashMap<String, Box<dyn DAAAgent + Send + Sync>>,
    ) -> HashMap<String, usize> {
        let mut types = HashMap::new();

        for agent in agents.values() {
            if let Ok(agent_type) = agent.get_type().await {
                *types.entry(agent_type).or_insert(0) += 1;
            }
        }

        types
    }

    /// Get current memory usage
    fn get_current_memory_usage() -> usize {
        // Platform-specific memory usage - simplified implementation
        std::mem::size_of::<DAACoordinator>() * 1024 // Placeholder
    }

    /// Get memory event count
    async fn get_memory_size(&self) -> Result<usize, DAAError> {
        let memory = self.coordination_memory.lock().await;
        memory.get_event_count().await
    }

    /// Get learned patterns count
    async fn get_learned_patterns_count(&self) -> Result<usize, DAAError> {
        let engine = self.learning_engine.lock().await;
        engine.get_patterns_count().await
    }

    /// Get performance improvement percentage
    async fn get_performance_improvement(&self) -> Result<f64, DAAError> {
        // TODO: Implement performance improvement calculation
        // let engine = self.learning_engine.lock().await;
        // engine.get_performance_improvement().await
        Ok(0.0) // Return default value for now
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let matches = Command::new("daa-coordinator")
        .version(env!("CARGO_PKG_VERSION"))
        .about("DAA Coordination Service - Orchestrates Decentralized Autonomous Agents")
        .arg(
            Arg::new("port")
                .short('p')
                .long("port")
                .value_name("PORT")
                .help("Port to bind coordination service")
                .default_value("8080"),
        )
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("FILE")
                .help("Configuration file path")
                .default_value("daa-config.toml"),
        )
        .arg(
            Arg::new("log-level")
                .short('l')
                .long("log-level")
                .value_name("LEVEL")
                .help("Log level (debug, info, warn, error)")
                .default_value("info"),
        )
        .arg(
            Arg::new("daemon")
                .short('d')
                .long("daemon")
                .help("Run as daemon")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    // Initialize logging
    let log_level = matches.get_one::<String>("log-level").unwrap();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level)).init();

    // Initialize coordinator
    let coordinator = DAACoordinator::new().await?;

    println!("🤖 DAA Coordinator v{}", env!("CARGO_PKG_VERSION"));
    println!(
        "🔧 Configuration: {}",
        matches.get_one::<String>("config").unwrap()
    );
    println!("📡 Port: {}", matches.get_one::<String>("port").unwrap());

    // Start coordination service
    coordinator.start().await?;

    // Setup signal handling for graceful shutdown
    let coordinator_for_signal = coordinator;
    let mut interrupt = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;

    tokio::select! {
        _ = interrupt.recv() => {
            println!("\n📡 Received SIGINT, shutting down gracefully...");
        }
        _ = terminate.recv() => {
            println!("\n📡 Received SIGTERM, shutting down gracefully...");
        }
    }

    // Graceful shutdown
    coordinator_for_signal.stop().await?;

    println!("👋 DAA Coordinator shutdown complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_coordinator_initialization() {
        let coordinator = DAACoordinator::new().await.unwrap();
        let status = coordinator.get_status().await.unwrap();

        assert_eq!(status["status"], "active");
        assert_eq!(status["agents"]["total"], 0);
    }

    #[tokio::test]
    async fn test_agent_registration() {
        let coordinator = DAACoordinator::new().await.unwrap();

        // Create mock agent with proper cognitive pattern
        let agent = StandardDAAAgent::new(CognitivePattern::Systems)
            .await
            .unwrap();
        let expected_agent_id = agent.id().to_string();
        let agent_id = coordinator.register_agent(Box::new(agent)).await.unwrap();

        assert_eq!(agent_id, expected_agent_id);

        let status = coordinator.get_status().await.unwrap();
        assert_eq!(status["agents"]["total"], 1);
    }
}
