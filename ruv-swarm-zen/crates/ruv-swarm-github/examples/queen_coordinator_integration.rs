//! Example: Queen Coordinator GitHub Integration
//! 
//! This example demonstrates how to integrate the GitHub reporter with a queen coordinator
//! agent that monitors and reports swarm progress to GitHub issues or pull requests.

use ruv_swarm_github::{GitHubClient, ProgressReporter, GitHubConfig};
use ruv_swarm_core::{
    Agent, AgentId, AgentMetadata, AgentStatus, CognitivePattern, 
    Swarm, SwarmConfig, Task, TaskId, Priority, Result as SwarmResult,
    swarm_trait::{SwarmPerformanceMetrics, SwarmErrorStatistics, ErrorTrend}
};
use async_trait::async_trait;
use std::{collections::HashMap, time::Duration};
use tokio::time::{interval, sleep};
use tracing::{info, warn, error, debug};

/// Queen Coordinator Agent with GitHub integration
pub struct QueenCoordinator {
    id: AgentId,
    metadata: AgentMetadata,
    status: AgentStatus,
    github_reporter: Option<ProgressReporter>,
    update_interval: Duration,
    performance_history: Vec<SwarmPerformanceMetrics>,
    error_history: Vec<SwarmErrorStatistics>,
    last_status_hash: Option<u64>,
}

impl QueenCoordinator {
    /// Create a new Queen Coordinator with GitHub integration
    pub async fn new(
        id: impl Into<String>,
        github_config: Option<GitHubConfig>,
        update_interval_minutes: u64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let agent_id = id.into();
        
        // Initialize GitHub reporter if config provided
        let github_reporter = if let Some(config) = github_config {
            info!("Initializing GitHub integration for queen coordinator");
            let client = GitHubClient::new(config).await?;
            Some(ProgressReporter::new(client))
        } else {
            warn!("No GitHub configuration provided - running without GitHub integration");
            None
        };
        
        Ok(Self {
            id: agent_id.clone(),
            metadata: AgentMetadata {
                name: format!("Queen Coordinator: {}", agent_id),
                description: "Master coordinator agent with GitHub progress reporting".to_string(),
                version: "1.0.0".to_string(),
                cognitive_pattern: CognitivePattern::Coordinator,
                capabilities: vec![
                    "swarm_monitoring".to_string(),
                    "progress_reporting".to_string(),
                    "github_integration".to_string(),
                    "performance_analysis".to_string(),
                ],
                resource_requirements: Default::default(),
            },
            status: AgentStatus::Idle,
            github_reporter,
            update_interval: Duration::from_secs(update_interval_minutes * 60),
            performance_history: Vec::new(),
            error_history: Vec::new(),
            last_status_hash: None,
        })
    }
    
    /// Start monitoring and reporting on swarm status
    pub async fn start_monitoring(&mut self, swarm: &mut Swarm) -> Result<(), Box<dyn std::error::Error>> {
        info!("Queen coordinator starting swarm monitoring");
        self.status = AgentStatus::Running;
        
        // Initial status report
        if let Some(ref mut reporter) = self.github_reporter {
            reporter.post_swarm_status(swarm, "🚀 Queen Coordinator activated - monitoring initiated").await?;
        }
        
        let mut update_timer = interval(self.update_interval);
        let mut health_check_timer = interval(Duration::from_secs(30));
        
        loop {
            tokio::select! {
                _ = update_timer.tick() => {
                    if let Err(e) = self.perform_scheduled_update(swarm).await {
                        error!("Scheduled update failed: {}", e);
                    }
                }
                
                _ = health_check_timer.tick() => {
                    if let Err(e) = self.perform_health_check(swarm).await {
                        error!("Health check failed: {}", e);
                    }
                }
                
                // In a real implementation, you'd also listen for:
                // - Swarm events
                // - Task completion notifications  
                // - Agent status changes
                // - External commands
            }
        }
    }
    
    /// Perform scheduled status update
    async fn perform_scheduled_update(&mut self, swarm: &Swarm) -> Result<(), Box<dyn std::error::Error>> {
        debug!("Performing scheduled swarm status update");
        
        let metrics = swarm.metrics();
        let status_hash = self.calculate_status_hash(&metrics);
        
        // Only update if status changed significantly or enough time passed
        let should_update = if let Some(last_hash) = self.last_status_hash {
            status_hash != last_hash
        } else {
            true
        };
        
        if should_update {
            if let Some(ref mut reporter) = self.github_reporter {
                let context = format!(
                    "📊 Scheduled update - {} agents active, {} tasks queued",
                    metrics.active_agents, metrics.queued_tasks
                );
                
                reporter.post_swarm_status(swarm, &context).await?;
                self.last_status_hash = Some(status_hash);
                
                info!("Posted scheduled status update to GitHub");
            }
        } else {
            debug!("Status unchanged - skipping update");
        }
        
        // Generate and store performance metrics
        let perf_metrics = self.generate_performance_metrics(swarm);
        self.performance_history.push(perf_metrics.clone());
        
        // Keep only last 24 hours of data (assuming hourly updates)
        if self.performance_history.len() > 24 {
            self.performance_history.remove(0);
        }
        
        // Report performance if significant changes
        if self.should_report_performance(&perf_metrics) {
            if let Some(ref mut reporter) = self.github_reporter {
                reporter.post_performance_metrics(&perf_metrics, "📈 Performance metrics update").await?;
                info!("Posted performance metrics to GitHub");
            }
        }
        
        Ok(())
    }
    
    /// Perform health check and report issues
    async fn perform_health_check(&mut self, swarm: &Swarm) -> Result<(), Box<dyn std::error::Error>> {
        debug!("Performing swarm health check");
        
        let metrics = swarm.metrics();
        let agent_statuses = swarm.agent_statuses();
        
        // Check for critical issues
        let mut issues = Vec::new();
        
        // No active agents
        if metrics.total_agents > 0 && metrics.active_agents == 0 {
            issues.push("🚨 **CRITICAL**: No active agents detected");
        }
        
        // High error rate
        let error_agents = agent_statuses.values()
            .filter(|&&status| status == AgentStatus::Error)
            .count();
        
        if error_agents > metrics.total_agents / 2 {
            issues.push("🚨 **CRITICAL**: Over 50% of agents in error state");
        }
        
        // Task queue overflow
        if metrics.queued_tasks > metrics.active_agents * 10 {
            issues.push("⚠️ **WARNING**: Task queue is growing rapidly");
        }
        
        // Report critical issues immediately
        if !issues.is_empty() {
            warn!("Critical swarm issues detected: {:?}", issues);
            
            if let Some(ref mut reporter) = self.github_reporter {
                let alert_message = format!(
                    "🚨 **SWARM ALERT** - Critical issues detected:\n\n{}\n\n**Immediate attention required!**",
                    issues.join("\n")
                );
                
                reporter.post_swarm_status(swarm, &alert_message).await?;
                info!("Posted critical alert to GitHub");
            }
        }
        
        Ok(())
    }
    
    /// Generate performance metrics from swarm state
    fn generate_performance_metrics(&self, swarm: &Swarm) -> SwarmPerformanceMetrics {
        let metrics = swarm.metrics();
        
        // In a real implementation, you'd track these over time
        SwarmPerformanceMetrics {
            tasks_completed: metrics.assigned_tasks * 10, // Simulated
            tasks_failed: metrics.assigned_tasks / 20,    // Simulated 5% failure rate
            avg_response_time_ms: 150.0 + (metrics.queued_tasks as f64 * 10.0), // Simulated
            success_rate: 0.95,
            tasks_per_second: metrics.active_agents as f64 * 0.5,
            memory_usage_percent: 45.0 + (metrics.total_agents as f64 * 2.0),
            cpu_usage_percent: 30.0 + (metrics.active_agents as f64 * 5.0),
        }
    }
    
    /// Determine if performance metrics should be reported
    fn should_report_performance(&self, current: &SwarmPerformanceMetrics) -> bool {
        if self.performance_history.is_empty() {
            return true; // First report
        }
        
        if let Some(last) = self.performance_history.last() {
            // Report if significant changes
            let response_time_change = (current.avg_response_time_ms - last.avg_response_time_ms).abs() / last.avg_response_time_ms;
            let success_rate_change = (current.success_rate - last.success_rate).abs();
            
            response_time_change > 0.2 || success_rate_change > 0.05
        } else {
            true
        }
    }
    
    /// Calculate hash of current swarm status for change detection
    fn calculate_status_hash(&self, metrics: &ruv_swarm_core::SwarmMetrics) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        metrics.total_agents.hash(&mut hasher);
        metrics.active_agents.hash(&mut hasher);
        metrics.queued_tasks.hash(&mut hasher);
        metrics.assigned_tasks.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Handle task completion events
    pub async fn on_task_completed(&mut self, task_id: &TaskId, success: bool) -> Result<(), Box<dyn std::error::Error>> {
        debug!("Task {} completed: {}", task_id, if success { "success" } else { "failed" });
        
        // In a real implementation, you'd:
        // 1. Update performance counters
        // 2. Check for patterns in failures
        // 3. Trigger reports if needed
        // 4. Update task assignment strategies
        
        Ok(())
    }
    
    /// Handle agent status change events
    pub async fn on_agent_status_change(&mut self, agent_id: &AgentId, old_status: AgentStatus, new_status: AgentStatus) -> Result<(), Box<dyn std::error::Error>> {
        info!("Agent {} status changed: {:?} -> {:?}", agent_id, old_status, new_status);
        
        // Report critical status changes immediately
        if new_status == AgentStatus::Error && old_status != AgentStatus::Error {
            if let Some(ref mut reporter) = self.github_reporter {
                let alert = format!("⚠️ **Agent Alert**: Agent `{}` entered error state", agent_id);
                
                // In a real implementation, you'd create a temporary swarm reference
                // For now, we'll skip the immediate report and let the health check handle it
                warn!("Agent {} entered error state - will be reported in next health check", agent_id);
            }
        }
        
        Ok(())
    }
}

#[async_trait]
impl Agent for QueenCoordinator {
    type Input = Task;
    type Output = String;
    type Error = Box<dyn std::error::Error + Send + Sync>;
    
    async fn process(&mut self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        info!("Queen coordinator processing coordination task: {}", input.id);
        
        // Coordination logic would go here
        // For now, just acknowledge the task
        Ok(format!("Coordination task {} processed", input.id))
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
        info!("Starting Queen Coordinator agent");
        self.status = AgentStatus::Running;
        Ok(())
    }
    
    async fn stop(&mut self) -> SwarmResult<()> {
        info!("Stopping Queen Coordinator agent");
        self.status = AgentStatus::Stopped;
        
        // Final status report
        if let Some(ref mut reporter) = self.github_reporter {
            // Note: In a real implementation, you'd have access to the swarm here
            info!("Queen coordinator shutdown - final status report skipped (no swarm reference)");
        }
        
        Ok(())
    }
    
    fn can_handle(&self, task: &Task) -> bool {
        // Queen coordinator handles coordination and monitoring tasks
        task.metadata.get("type").map_or(false, |t| {
            t == "coordination" || t == "monitoring" || t == "reporting"
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("debug")
        .init();
    
    info!("Starting Queen Coordinator GitHub Integration Example");
    
    // Load GitHub configuration from environment
    let github_config = match GitHubConfig::from_env() {
        Ok(config) => {
            info!("GitHub integration enabled");
            Some(config)
        }
        Err(e) => {
            warn!("GitHub integration disabled: {}", e);
            warn!("Set GITHUB_TOKEN, GITHUB_REPO_OWNER, and GITHUB_REPO_NAME to enable");
            None
        }
    };
    
    // Create queen coordinator
    let mut queen = QueenCoordinator::new(
        "queen-main",
        github_config,
        5, // Update every 5 minutes
    ).await?;
    
    // Create a sample swarm
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    // In a real implementation, you'd:
    // 1. Register agents with the swarm
    // 2. Submit tasks
    // 3. Start the swarm processing
    
    info!("Starting monitoring loop...");
    info!("Press Ctrl+C to stop");
    
    // Start monitoring (this would run indefinitely)
    // For the example, we'll run for a short time
    tokio::select! {
        result = queen.start_monitoring(&mut swarm) => {
            if let Err(e) = result {
                error!("Monitoring failed: {}", e);
            }
        }
        _ = tokio::signal::ctrl_c() => {
            info!("Received shutdown signal");
        }
        _ = sleep(Duration::from_secs(30)) => {
            info!("Example completed after 30 seconds");
        }
    }
    
    // Shutdown
    queen.stop().await?;
    info!("Queen Coordinator example completed");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_queen_coordinator_creation() {
        let queen = QueenCoordinator::new("test-queen", None, 1).await.unwrap();
        assert_eq!(queen.id(), "test-queen");
        assert_eq!(queen.status(), AgentStatus::Idle);
    }
    
    #[tokio::test]
    async fn test_performance_metrics_generation() {
        let queen = QueenCoordinator::new("test-queen", None, 1).await.unwrap();
        let swarm = Swarm::new(SwarmConfig::default());
        
        let metrics = queen.generate_performance_metrics(&swarm);
        assert!(metrics.tasks_completed >= 0);
        assert!(metrics.success_rate >= 0.0 && metrics.success_rate <= 1.0);
    }
}