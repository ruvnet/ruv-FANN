//! GitHub Integration Example
//! 
//! This example demonstrates the production-ready GitHub API integration for
//! RUV Swarm coordinator progress reporting.
//!
//! ## Setup
//!
//! Before running this example, set up your environment:
//!
//! ```bash
//! export GITHUB_TOKEN="your_github_token_here"
//! export GITHUB_REPO_OWNER="ruvnet"
//! export GITHUB_REPO_NAME="ruv-FANN"
//! export GITHUB_ISSUE_NUMBER="123"  # Optional: specific issue to update
//! ```
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example github_integration --features="github-integration"
//! ```

use ruv_swarm_core::{
    Agent, AgentId, AgentMetadata, AgentStatus, CognitivePattern,
    Swarm, SwarmConfig, Task, TaskId, Priority, Result as SwarmResult,
    swarm_trait::{SwarmPerformanceMetrics, SwarmErrorStatistics}
};

#[cfg(feature = "github-integration")]
use ruv_swarm_github::{GitHubClient, ProgressReporter, GitHubConfig};

use async_trait::async_trait;
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::{sync::RwLock, time::{interval, sleep}};
use tracing::{info, warn, error, debug};

/// Example agent for demonstration
struct WorkerAgent {
    id: AgentId,
    metadata: AgentMetadata,
    status: AgentStatus,
    task_count: u32,
}

impl WorkerAgent {
    fn new(id: impl Into<String>, agent_type: &str) -> Self {
        let agent_id = id.into();
        Self {
            id: agent_id.clone(),
            metadata: AgentMetadata {
                name: format!("Worker Agent: {}", agent_id),
                description: format!("{} worker agent for demonstration", agent_type),
                version: "1.0.0".to_string(),
                cognitive_pattern: match agent_type {
                    "researcher" => CognitivePattern::Researcher,
                    "analyst" => CognitivePattern::Analyst,
                    "coordinator" => CognitivePattern::Coordinator,
                    _ => CognitivePattern::Worker,
                },
                capabilities: vec![
                    format!("{}_processing", agent_type),
                    "task_execution".to_string(),
                ],
                resource_requirements: Default::default(),
            },
            status: AgentStatus::Idle,
            task_count: 0,
        }
    }
}

#[async_trait]
impl Agent for WorkerAgent {
    type Input = Task;
    type Output = String;
    type Error = Box<dyn std::error::Error + Send + Sync>;

    async fn process(&mut self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        self.task_count += 1;
        
        // Simulate some work
        let work_duration = Duration::from_millis(100 + (self.task_count as u64 * 50));
        sleep(work_duration).await;
        
        // Occasionally fail to demonstrate error reporting
        if self.task_count % 10 == 0 {
            self.status = AgentStatus::Error;
            return Err("Simulated processing error".into());
        }
        
        Ok(format!("Task {} processed by {}", input.id, self.id))
    }

    fn capabilities(&self) -> &[String] {
        &self.metadata.capabilities
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn metadata(&self) -> &AgentMetadata {
        &self.metadata
    }

    fn status(&self) -> AgentStatus {
        self.status
    }

    async fn start(&mut self) -> SwarmResult<()> {
        info!("Starting worker agent: {}", self.id);
        self.status = AgentStatus::Running;
        Ok(())
    }

    async fn stop(&mut self) -> SwarmResult<()> {
        info!("Stopping worker agent: {}", self.id);
        self.status = AgentStatus::Stopped;
        Ok(())
    }

    fn can_handle(&self, _task: &Task) -> bool {
        self.status == AgentStatus::Running || self.status == AgentStatus::Idle
    }
}

/// Enhanced swarm coordinator with GitHub integration
struct GitHubIntegratedCoordinator {
    swarm: Arc<RwLock<Swarm>>,
    #[cfg(feature = "github-integration")]
    github_reporter: Option<ProgressReporter>,
    update_interval: Duration,
    task_counter: Arc<RwLock<u32>>,
}

impl GitHubIntegratedCoordinator {
    #[cfg(feature = "github-integration")]
    async fn new(update_interval_minutes: u64) -> Result<Self, Box<dyn std::error::Error>> {
        let swarm = Arc::new(RwLock::new(Swarm::new(SwarmConfig::default())));
        
        // Try to initialize GitHub integration
        let github_reporter = match GitHubConfig::from_env() {
            Ok(config) => {
                info!("Initializing GitHub integration");
                info!("Repository: {}", config.repo_identifier());
                
                if let Some(issue) = config.issue_number {
                    info!("Target issue: #{}", issue);
                } else if let Some(pr) = config.pr_number {
                    info!("Target PR: #{}", pr);
                } else {
                    info!("Will create new issues for updates");
                }
                
                let client = GitHubClient::new(config).await?;
                Some(ProgressReporter::new(client))
            }
            Err(e) => {
                warn!("GitHub integration disabled: {}", e);
                warn!("To enable, set GITHUB_TOKEN, GITHUB_REPO_OWNER, and GITHUB_REPO_NAME");
                None
            }
        };
        
        Ok(Self {
            swarm,
            github_reporter,
            update_interval: Duration::from_secs(update_interval_minutes * 60),
            task_counter: Arc::new(RwLock::new(0)),
        })
    }
    
    #[cfg(not(feature = "github-integration"))]
    async fn new(update_interval_minutes: u64) -> Result<Self, Box<dyn std::error::Error>> {
        warn!("GitHub integration is not enabled. Use --features=\"github-integration\" to enable.");
        
        Ok(Self {
            swarm: Arc::new(RwLock::new(Swarm::new(SwarmConfig::default()))),
            update_interval: Duration::from_secs(update_interval_minutes * 60),
            task_counter: Arc::new(RwLock::new(0)),
        })
    }
    
    /// Initialize the swarm with demo agents
    async fn initialize_swarm(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Initializing swarm with demo agents");
        
        let mut swarm = self.swarm.write().await;
        
        // Create diverse agent types
        let agent_configs = vec![
            ("researcher-1", "researcher"),
            ("researcher-2", "researcher"),
            ("analyst-1", "analyst"),
            ("analyst-2", "analyst"),
            ("coordinator-1", "coordinator"),
            ("worker-1", "worker"),
            ("worker-2", "worker"),
            ("worker-3", "worker"),
        ];
        
        for (id, agent_type) in agent_configs {
            let mut agent = WorkerAgent::new(id, agent_type);
            agent.start().await?;
            
            swarm.register_agent(Box::new(agent))?;
            info!("Registered {} agent: {}", agent_type, id);
        }
        
        // Start all agents
        swarm.start_all_agents().await?;
        
        info!("Swarm initialized with {} agents", swarm.metrics().total_agents);
        Ok(())
    }
    
    /// Submit demo tasks to the swarm
    async fn submit_demo_tasks(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut task_id = self.task_counter.write().await;
        let mut swarm = self.swarm.write().await;
        
        // Submit various types of tasks
        let task_types = vec![
            ("research", "Research task"),
            ("analysis", "Analysis task"),
            ("coordination", "Coordination task"),
            ("processing", "Processing task"),
        ];
        
        for (task_type, description) in task_types {
            *task_id += 1;
            let task = Task {
                id: format!("task-{:04}", *task_id),
                priority: if *task_id % 3 == 0 { Priority::High } else { Priority::Normal },
                payload: format!("{}: {}", task_type, description).into_bytes(),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("type".to_string(), task_type.to_string());
                    meta.insert("description".to_string(), description.to_string());
                    meta
                },
            };
            
            swarm.submit_task(task)?;
        }
        
        debug!("Submitted {} tasks", task_types.len());
        Ok(())
    }
    
    /// Process tasks and update status
    async fn process_tasks(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut swarm = self.swarm.write().await;
        let assignments = swarm.distribute_tasks().await?;
        
        if !assignments.is_empty() {
            info!("Distributed {} tasks to agents", assignments.len());
            
            for (task_id, agent_id) in assignments {
                debug!("Task {} assigned to agent {}", task_id, agent_id);
            }
        }
        
        Ok(())
    }
    
    /// Generate performance metrics
    fn generate_performance_metrics(&self, metrics: &ruv_swarm_core::SwarmMetrics) -> SwarmPerformanceMetrics {
        // In a real implementation, these would be tracked over time
        SwarmPerformanceMetrics {
            tasks_completed: (metrics.assigned_tasks * 8) as u64, // Simulated completion rate
            tasks_failed: (metrics.assigned_tasks / 10) as u64,   // Simulated 10% failure rate
            avg_response_time_ms: 150.0 + (metrics.queued_tasks as f64 * 5.0),
            success_rate: 0.90 - (metrics.queued_tasks as f64 * 0.001).min(0.1),
            tasks_per_second: metrics.active_agents as f64 * 0.8,
            memory_usage_percent: 35.0 + (metrics.total_agents as f64 * 3.0).min(50.0),
            cpu_usage_percent: 25.0 + (metrics.active_agents as f64 * 8.0).min(60.0),
        }
    }
    
    /// Generate error statistics
    fn generate_error_statistics(&self) -> SwarmErrorStatistics {
        use chrono::Utc;
        
        SwarmErrorStatistics {
            total_errors: 15,
            critical_errors: 2,
            warnings: 13,
            error_rate: 0.08,
            error_categories: {
                let mut categories = HashMap::new();
                categories.insert("NetworkError".to_string(), 8);
                categories.insert("ProcessingError".to_string(), 4);
                categories.insert("TimeoutError".to_string(), 3);
                categories
            },
            recent_errors: vec![
                ruv_swarm_core::swarm_trait::ErrorDetails {
                    error_type: "ProcessingError".to_string(),
                    message: "Agent worker-2 processing timeout".to_string(),
                    timestamp: Some(Utc::now() - chrono::Duration::minutes(5)),
                },
                ruv_swarm_core::swarm_trait::ErrorDetails {
                    error_type: "NetworkError".to_string(),
                    message: "Connection lost to external service".to_string(),
                    timestamp: Some(Utc::now() - chrono::Duration::minutes(15)),
                },
            ],
            error_trends: vec![
                ErrorTrend {
                    error_type: "ProcessingError".to_string(),
                    trend: "increasing".to_string(),
                    change_percent: 25.0,
                }
            ],
        }
    }
    
    /// Start the monitoring loop
    async fn start_monitoring(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Starting GitHub-integrated swarm monitoring");
        
        // Initial status report
        #[cfg(feature = "github-integration")]
        if let Some(ref mut reporter) = self.github_reporter {
            let swarm = self.swarm.read().await;
            reporter.post_swarm_status(&swarm, "🚀 Enhanced coordinator activated with GitHub integration").await?;
        }
        
        let mut status_timer = interval(self.update_interval);
        let mut task_timer = interval(Duration::from_secs(10)); // Submit tasks every 10 seconds
        let mut process_timer = interval(Duration::from_secs(5)); // Process tasks every 5 seconds
        let mut perf_timer = interval(Duration::from_secs(120)); // Performance report every 2 minutes
        
        let mut iteration = 0u32;
        
        loop {
            tokio::select! {
                _ = status_timer.tick() => {
                    iteration += 1;
                    info!("Status update iteration: {}", iteration);
                    
                    if let Err(e) = self.post_status_update(iteration).await {
                        error!("Status update failed: {}", e);
                    }
                }
                
                _ = task_timer.tick() => {
                    if let Err(e) = self.submit_demo_tasks().await {
                        error!("Task submission failed: {}", e);
                    }
                }
                
                _ = process_timer.tick() => {
                    if let Err(e) = self.process_tasks().await {
                        error!("Task processing failed: {}", e);
                    }
                }
                
                _ = perf_timer.tick() => {
                    if let Err(e) = self.post_performance_update().await {
                        error!("Performance update failed: {}", e);
                    }
                }
            }
        }
    }
    
    /// Post status update to GitHub
    async fn post_status_update(&mut self, iteration: u32) -> Result<(), Box<dyn std::error::Error>> {
        let swarm = self.swarm.read().await;
        let metrics = swarm.metrics();
        
        let context = format!(
            "📊 Status Update #{} - {} active agents, {} queued tasks, {} assigned tasks",
            iteration, metrics.active_agents, metrics.queued_tasks, metrics.assigned_tasks
        );
        
        info!("{}", context);
        
        #[cfg(feature = "github-integration")]
        if let Some(ref mut reporter) = self.github_reporter {
            reporter.post_swarm_status(&swarm, &context).await?;
            info!("Posted status update to GitHub");
        } else {
            debug!("GitHub integration not available - status logged locally");
        }
        
        Ok(())
    }
    
    /// Post performance update to GitHub
    async fn post_performance_update(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let swarm = self.swarm.read().await;
        let metrics = swarm.metrics();
        let perf_metrics = self.generate_performance_metrics(&metrics);
        
        info!("Performance metrics - Completed: {}, Failed: {}, Success Rate: {:.1}%", 
               perf_metrics.tasks_completed, perf_metrics.tasks_failed, perf_metrics.success_rate * 100.0);
        
        #[cfg(feature = "github-integration")]
        if let Some(ref mut reporter) = self.github_reporter {
            reporter.post_performance_metrics(&perf_metrics, "📈 Automated performance report").await?;
            info!("Posted performance metrics to GitHub");
            
            // Occasionally post error analysis
            if perf_metrics.tasks_failed > 10 {
                let error_stats = self.generate_error_statistics();
                reporter.post_error_statistics(&error_stats, "🚨 Error analysis report").await?;
                info!("Posted error analysis to GitHub");
            }
        } else {
            debug!("GitHub integration not available - performance metrics logged locally");
        }
        
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info,ruv_swarm_github=debug")
        .with_target(false)
        .with_thread_ids(true)
        .init();
    
    info!("🚀 RUV Swarm GitHub Integration Example");
    info!("=========================================");
    
    #[cfg(feature = "github-integration")]
    info!("✅ GitHub integration is ENABLED");
    
    #[cfg(not(feature = "github-integration"))]
    warn!("❌ GitHub integration is DISABLED - use --features=\"github-integration\" to enable");
    
    // Create and initialize coordinator
    let mut coordinator = GitHubIntegratedCoordinator::new(2).await?; // 2-minute updates
    
    // Initialize swarm
    coordinator.initialize_swarm().await?;
    
    info!("Starting monitoring loop...");
    info!("Press Ctrl+C to stop");
    
    // Start monitoring with graceful shutdown
    tokio::select! {
        result = coordinator.start_monitoring() => {
            if let Err(e) = result {
                error!("Monitoring failed: {}", e);
            }
        }
        _ = tokio::signal::ctrl_c() => {
            info!("Received shutdown signal");
        }
    }
    
    // Graceful shutdown
    {
        let mut swarm = coordinator.swarm.write().await;
        if let Err(e) = swarm.shutdown_all_agents().await {
            error!("Error during shutdown: {}", e);
        }
    }
    
    #[cfg(feature = "github-integration")]
    if let Some(ref mut reporter) = coordinator.github_reporter {
        let swarm = coordinator.swarm.read().await;
        if let Err(e) = reporter.post_swarm_status(&swarm, "🔄 Coordinator shutting down gracefully").await {
            error!("Failed to post shutdown status: {}", e);
        } else {
            info!("Posted shutdown status to GitHub");
        }
    }
    
    info!("✅ GitHub Integration Example completed successfully");
    Ok(())
}