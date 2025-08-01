//! Progress reporting functionality for swarm coordination

use crate::{client::GitHubClient, error::{GitHubError, Result}};
use chrono::{DateTime, Utc};
use ruv_swarm_core::{
    swarm_trait::{SwarmMetricsCore, SwarmPerformanceMetrics, SwarmErrorStatistics},
    Swarm, SwarmMetrics, Agent, AgentStatus, AgentId
};
use serde_json::Value;
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Progress reporter for GitHub integration
pub struct ProgressReporter {
    client: GitHubClient,
    last_update: Option<DateTime<Utc>>,
}

impl ProgressReporter {
    /// Create a new progress reporter
    pub fn new(client: GitHubClient) -> Self {
        Self {
            client,
            last_update: None,
        }
    }
    
    /// Post swarm status update to GitHub
    /// 
    /// # Arguments
    /// 
    /// * `swarm` - The swarm to generate status for
    /// * `message` - Additional context message
    /// 
    /// # Errors
    /// 
    /// Returns an error if the status cannot be posted to GitHub
    pub async fn post_swarm_status(&mut self, swarm: &Swarm, message: &str) -> Result<()> {
        let report = self.generate_swarm_status_report(swarm, message)?;
        self.post_update(&report).await?;
        self.last_update = Some(Utc::now());
        Ok(())
    }
    
    /// Post agent performance metrics to GitHub
    /// 
    /// # Arguments
    /// 
    /// * `metrics` - Performance metrics to report
    /// * `context` - Additional context about the metrics
    /// 
    /// # Errors
    /// 
    /// Returns an error if the metrics cannot be posted to GitHub
    pub async fn post_performance_metrics(
        &mut self,
        metrics: &SwarmPerformanceMetrics,
        context: &str,
    ) -> Result<()> {
        let report = self.format_performance_report(metrics, context)?;
        self.post_update(&report).await?;
        self.last_update = Some(Utc::now());
        Ok(())
    }
    
    /// Post error statistics to GitHub
    /// 
    /// # Arguments
    /// 
    /// * `errors` - Error statistics to report
    /// * `context` - Additional context about the errors
    /// 
    /// # Errors
    /// 
    /// Returns an error if the error report cannot be posted to GitHub
    pub async fn post_error_statistics(
        &mut self,
        errors: &SwarmErrorStatistics,
        context: &str,
    ) -> Result<()> {
        let report = self.format_error_report(errors, context)?;
        self.post_update(&report).await?;
        self.last_update = Some(Utc::now());
        Ok(())
    }
    
    /// Generate comprehensive swarm status report
    fn generate_swarm_status_report(&self, swarm: &Swarm, message: &str) -> Result<String> {
        let metrics = swarm.metrics();
        let agent_statuses = swarm.agent_statuses();
        let timestamp = Utc::now();
        
        let mut report = String::new();
        
        // Header
        report.push_str("# 🐝 Swarm Orchestration Status Report\n\n");
        report.push_str(&format!("**Generated:** {}\n", timestamp.format("%Y-%m-%d %H:%M:%S UTC")));
        report.push_str(&format!("**Repository:** {}\n", self.client.config().repo_identifier()));
        
        if !message.is_empty() {
            report.push_str(&format!("**Context:** {}\n", message));
        }
        
        report.push_str("\n---\n\n");
        
        // Overview section
        report.push_str("## 📊 Overview\n\n");
        report.push_str("| Metric | Value |\n");
        report.push_str("|--------|-------|\n");
        report.push_str(&format!("| Total Agents | {} |\n", metrics.total_agents));
        report.push_str(&format!("| Active Agents | {} |\n", metrics.active_agents));
        report.push_str(&format!("| Queued Tasks | {} |\n", metrics.queued_tasks));
        report.push_str(&format!("| Assigned Tasks | {} |\n", metrics.assigned_tasks));
        report.push_str(&format!("| Total Connections | {} |\n", metrics.total_connections));
        
        // Agent status breakdown
        report.push_str("\n## 🤖 Agent Status Breakdown\n\n");
        let status_counts = self.count_agent_statuses(&agent_statuses);
        
        for (status, count) in &status_counts {
            let emoji = match status {
                AgentStatus::Running => "🟢",
                AgentStatus::Idle => "🟡",
                AgentStatus::Stopped => "🔴",
                AgentStatus::Error => "❌",
                AgentStatus::Starting => "🔵",
                AgentStatus::Stopping => "🟠",
            };
            report.push_str(&format!("- {} **{:?}**: {} agents\n", emoji, status, count));
        }
        
        // Detailed agent list (if not too many)
        if agent_statuses.len() <= 20 {
            report.push_str("\n### Agent Details\n\n");
            report.push_str("| Agent ID | Status |\n");
            report.push_str("|----------|--------|\n");
            
            let mut sorted_agents: Vec<_> = agent_statuses.iter().collect();
            sorted_agents.sort_by_key(|(id, _)| id.as_str());
            
            for (agent_id, status) in sorted_agents {
                let emoji = match status {
                    AgentStatus::Running => "🟢",
                    AgentStatus::Idle => "🟡",
                    AgentStatus::Stopped => "🔴",
                    AgentStatus::Error => "❌",
                    AgentStatus::Starting => "🔵",
                    AgentStatus::Stopping => "🟠",
                };
                report.push_str(&format!("| `{}` | {} {:?} |\n", agent_id, emoji, status));
            }
        } else {
            report.push_str(&format!("\n*{} agents total - detailed list omitted for brevity*\n", agent_statuses.len()));
        }
        
        // Health indicators
        report.push_str("\n## 🏥 Health Indicators\n\n");
        let health_score = self.calculate_health_score(&metrics, &agent_statuses);
        let health_emoji = if health_score >= 0.9 {
            "🟢"
        } else if health_score >= 0.7 {
            "🟡"
        } else {
            "🔴"
        };
        
        report.push_str(&format!("**Overall Health:** {} {:.1}%\n\n", health_emoji, health_score * 100.0));
        
        // Recommendations
        if health_score < 0.9 {
            report.push_str("### 💡 Recommendations\n\n");
            
            if metrics.active_agents == 0 && metrics.total_agents > 0 {
                report.push_str("- ⚠️ No active agents detected - investigate agent startup issues\n");
            }
            
            if metrics.queued_tasks > metrics.active_agents * 2 {
                report.push_str("- ⚠️ High task queue - consider scaling up agents\n");
            }
            
            let error_agents = status_counts.get(&AgentStatus::Error).unwrap_or(&0);
            if *error_agents > 0 {
                report.push_str(&format!("- ❌ {} agents in error state - investigate and restart\n", error_agents));
            }
        }
        
        // Footer
        report.push_str("\n---\n");
        report.push_str("*Report generated by ruv-swarm-github*\n");
        
        debug!("Generated swarm status report ({} chars)", report.len());
        Ok(report)
    }
    
    /// Format performance metrics report
    fn format_performance_report(&self, metrics: &SwarmPerformanceMetrics, context: &str) -> Result<String> {
        let mut report = String::new();
        let timestamp = Utc::now();
        
        report.push_str("# 📈 Swarm Performance Metrics\n\n");
        report.push_str(&format!("**Generated:** {}\n", timestamp.format("%Y-%m-%d %H:%M:%S UTC")));
        report.push_str(&format!("**Context:** {}\n\n", context));
        
        // Core metrics
        report.push_str("## 🎯 Core Metrics\n\n");
        report.push_str("| Metric | Value |\n");
        report.push_str("|--------|-------|\n");
        report.push_str(&format!("| Tasks Completed | {} |\n", metrics.tasks_completed));
        report.push_str(&format!("| Tasks Failed | {} |\n", metrics.tasks_failed));
        report.push_str(&format!("| Average Response Time | {:.2}ms |\n", metrics.avg_response_time_ms));
        report.push_str(&format!("| Success Rate | {:.1}% |\n", metrics.success_rate * 100.0));
        
        // Throughput metrics
        if metrics.tasks_per_second > 0.0 {
            report.push_str(&format!("| Throughput | {:.2} tasks/sec |\n", metrics.tasks_per_second));
        }
        
        // Resource utilization
        report.push_str("\n## 💾 Resource Utilization\n\n");
        report.push_str("| Resource | Usage |\n");
        report.push_str("|----------|-------|\n");
        report.push_str(&format!("| Memory Usage | {:.1}% |\n", metrics.memory_usage_percent));
        report.push_str(&format!("| CPU Usage | {:.1}% |\n", metrics.cpu_usage_percent));
        
        // Performance trends
        if metrics.avg_response_time_ms > 1000.0 {
            report.push_str("\n### ⚠️ Performance Alerts\n\n");
            report.push_str("- High response time detected - consider optimization\n");
        }
        
        if metrics.success_rate < 0.95 {
            report.push_str(&format!("- Low success rate ({:.1}%) - investigate failure causes\n", metrics.success_rate * 100.0));
        }
        
        report.push_str("\n---\n");
        report.push_str("*Performance report generated by ruv-swarm-github*\n");
        
        Ok(report)
    }
    
    /// Format error statistics report
    fn format_error_report(&self, errors: &SwarmErrorStatistics, context: &str) -> Result<String> {
        let mut report = String::new();
        let timestamp = Utc::now();
        
        report.push_str("# 🚨 Swarm Error Analysis\n\n");
        report.push_str(&format!("**Generated:** {}\n", timestamp.format("%Y-%m-%d %H:%M:%S UTC")));
        report.push_str(&format!("**Context:** {}\n\n", context));
        
        // Error summary
        report.push_str("## 📊 Error Summary\n\n");
        report.push_str("| Metric | Value |\n");
        report.push_str("|--------|-------|\n");
        report.push_str(&format!("| Total Errors | {} |\n", errors.total_errors));
        report.push_str(&format!("| Critical Errors | {} |\n", errors.critical_errors));
        report.push_str(&format!("| Warnings | {} |\n", errors.warnings));
        report.push_str(&format!("| Error Rate | {:.2}% |\n", errors.error_rate * 100.0));
        
        // Error categories
        if !errors.error_categories.is_empty() {
            report.push_str("\n## 📋 Error Categories\n\n");
            report.push_str("| Category | Count | Percentage |\n");
            report.push_str("|----------|-------|------------|\n");
            
            for (category, count) in &errors.error_categories {
                let percentage = (*count as f64 / errors.total_errors as f64) * 100.0;
                report.push_str(&format!("| {} | {} | {:.1}% |\n", category, count, percentage));
            }
        }
        
        // Recent errors
        if !errors.recent_errors.is_empty() {
            report.push_str("\n## 🕐 Recent Errors\n\n");
            
            for (i, error) in errors.recent_errors.iter().take(5).enumerate() {
                report.push_str(&format!("{}. **{}** - {}\n", i + 1, error.error_type, error.message));
                if let Some(timestamp) = error.timestamp {
                    report.push_str(&format!("   *Occurred at: {}*\n", timestamp.format("%Y-%m-%d %H:%M:%S UTC")));
                }
                report.push_str("\n");
            }
        }
        
        // Recommendations
        if errors.total_errors > 0 {
            report.push_str("## 💡 Recommendations\n\n");
            
            if errors.critical_errors > 0 {
                report.push_str("- 🚨 **Critical**: Address critical errors immediately\n");
            }
            
            if errors.error_rate > 0.1 {
                report.push_str("- ⚠️ **High Error Rate**: Investigate root causes and implement error handling improvements\n");
            }
            
            // Category-specific recommendations
            for (category, count) in &errors.error_categories {
                if *count as f64 / errors.total_errors as f64 > 0.3 {
                    report.push_str(&format!("- 🎯 **{}**: Focus on addressing {} category errors ({} occurrences)\n", 
                        category, category.to_lowercase(), count));
                }
            }
        }
        
        report.push_str("\n---\n");
        report.push_str("*Error analysis generated by ruv-swarm-github*\n");
        
        Ok(report)
    }
    
    /// Post an update to GitHub
    async fn post_update(&self, content: &str) -> Result<()> {
        let config = self.client.config();
        
        if let Some(issue_number) = config.issue_number {
            info!("Posting swarm update to issue #{}", issue_number);
            self.client.post_issue_comment(issue_number, content).await?;
        } else if let Some(pr_number) = config.pr_number {
            info!("Posting swarm update to PR #{}", pr_number);
            self.client.post_pr_comment(pr_number, content).await?;
        } else {
            warn!("No issue or PR number configured - creating new issue");
            let issue_number = self.client.create_issue(
                "Swarm Orchestration Status",
                content,
                Some(vec!["swarm".to_string(), "status".to_string()]),
            ).await?;
            info!("Created new issue #{} for swarm updates", issue_number);
        }
        
        Ok(())
    }
    
    /// Count agents by status
    fn count_agent_statuses(&self, statuses: &HashMap<AgentId, AgentStatus>) -> HashMap<AgentStatus, usize> {
        let mut counts = HashMap::new();
        
        for status in statuses.values() {
            *counts.entry(*status).or_insert(0) += 1;
        }
        
        counts
    }
    
    /// Calculate overall health score
    fn calculate_health_score(&self, metrics: &SwarmMetrics, statuses: &HashMap<AgentId, AgentStatus>) -> f64 {
        if metrics.total_agents == 0 {
            return 1.0; // Empty swarm is technically "healthy"
        }
        
        let active_ratio = metrics.active_agents as f64 / metrics.total_agents as f64;
        let error_agents = statuses.values()
            .filter(|&&status| status == AgentStatus::Error)
            .count();
        let error_penalty = (error_agents as f64 * 0.2).min(0.5);
        
        let task_efficiency = if metrics.queued_tasks > 0 && metrics.active_agents > 0 {
            let load_per_agent = metrics.queued_tasks as f64 / metrics.active_agents as f64;
            if load_per_agent > 5.0 {
                0.8 // High queue indicates potential bottleneck
            } else {
                1.0
            }
        } else {
            1.0
        };
        
        (active_ratio * task_efficiency - error_penalty).max(0.0).min(1.0)
    }
    
    /// Get the last update timestamp
    pub fn last_update(&self) -> Option<DateTime<Utc>> {
        self.last_update
    }
    
    /// Check if enough time has passed since last update
    pub fn should_update(&self) -> bool {
        if let Some(last_update) = self.last_update {
            let elapsed = Utc::now().signed_duration_since(last_update);
            elapsed >= chrono::Duration::from_std(self.client.config().update_frequency()).unwrap_or_default()
        } else {
            true // First update
        }
    }
}