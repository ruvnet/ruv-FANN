use anyhow::{Context, Result};
use chrono::{DateTime, Local, Utc};
use colored::Colorize;
use std::collections::HashMap;
use std::path::Path;

use crate::commands::orchestrate::{Task, TaskStatus};
use crate::commands::spawn::{Agent, AgentMetrics, AgentStatus};
use crate::output::{OutputHandler, StatusLevel};

#[derive(Debug)]
struct SwarmStatus {
    swarm_id: String,
    topology: String,
    created_at: DateTime<Utc>,
    total_agents: usize,
    active_agents: usize,
    total_tasks: usize,
    running_tasks: usize,
    completed_tasks: usize,
    failed_tasks: usize,
    uptime: chrono::Duration,
}

#[derive(Debug)]
struct AgentStatusInfo {
    agent: Agent,
    current_task: Option<String>,
    performance_score: f32,
}

/// Execute the status command
pub async fn execute(
    config: &crate::config::Config,
    output: &OutputHandler,
    detailed: bool,
    agent_type: Option<String>,
    active_only: bool,
    metrics: bool,
) -> Result<()> {
    output.section("RUV Swarm Status");

    // Load current swarm
    let swarm_config = load_current_swarm(output).await?;

    // Load agents
    let all_agents = load_agents(&swarm_config).await?;

    // Filter agents based on criteria
    let filtered_agents: Vec<Agent> = all_agents
        .into_iter()
        .filter(|a| {
            // Filter by type if specified
            if let Some(ref agent_type) = agent_type {
                if a.agent_type != *agent_type {
                    return false;
                }
            }

            // Filter by active status if requested
            if active_only {
                match &a.status {
                    AgentStatus::Ready | AgentStatus::Busy => true,
                    _ => false,
                }
            } else {
                true
            }
        })
        .collect();

    // Load tasks
    let all_tasks = load_tasks().await?;

    // Calculate swarm statistics
    let swarm_status = calculate_swarm_status(&swarm_config, &filtered_agents, &all_tasks)?;

    // Display swarm overview
    display_swarm_overview(&swarm_status, output);

    // Display agent summary
    display_agent_summary(&filtered_agents, output);

    // Display task summary
    display_task_summary(&all_tasks, output);

    if detailed {
        // Display detailed agent information
        display_detailed_agents(&filtered_agents, &all_tasks, output);

        // Display active tasks
        display_active_tasks(&all_tasks, output);
    }

    if metrics {
        // Display performance metrics
        display_performance_metrics(&filtered_agents, &all_tasks, output);

        // Display resource usage
        display_resource_usage(&filtered_agents, output);
    }

    // Display alerts if any
    display_alerts(&filtered_agents, &all_tasks, config, output);

    Ok(())
}

fn calculate_swarm_status(
    swarm_config: &crate::commands::init::SwarmInit,
    agents: &[Agent],
    tasks: &[Task],
) -> Result<SwarmStatus> {
    let now = Utc::now();
    let uptime = now - swarm_config.created_at;

    let active_agents = agents
        .iter()
        .filter(|a| matches!(a.status, AgentStatus::Ready | AgentStatus::Busy))
        .count();

    let running_tasks = tasks
        .iter()
        .filter(|t| matches!(t.status, TaskStatus::Running))
        .count();

    let completed_tasks = tasks
        .iter()
        .filter(|t| matches!(t.status, TaskStatus::Completed))
        .count();

    let failed_tasks = tasks
        .iter()
        .filter(|t| matches!(t.status, TaskStatus::Failed(_)))
        .count();

    Ok(SwarmStatus {
        swarm_id: swarm_config.swarm_id.clone(),
        topology: swarm_config.topology.clone(),
        created_at: swarm_config.created_at,
        total_agents: agents.len(),
        active_agents,
        total_tasks: tasks.len(),
        running_tasks,
        completed_tasks,
        failed_tasks,
        uptime,
    })
}

fn display_swarm_overview(status: &SwarmStatus, output: &OutputHandler) {
    output.section("Swarm Overview");

    let uptime_str = format_duration(status.uptime);
    let health = calculate_swarm_health(status);

    output.print_status("Swarm ID", &status.swarm_id[..8], StatusLevel::Info);
    output.print_status("Topology", &status.topology, StatusLevel::Info);
    output.print_status("Uptime", &uptime_str, StatusLevel::Info);
    output.print_status(
        "Health",
        &format!("{:.1}%", health),
        if health >= 80.0 {
            StatusLevel::Good
        } else if health >= 60.0 {
            StatusLevel::Warning
        } else {
            StatusLevel::Error
        },
    );

    output.key_value(&[(
        "Created".to_string(),
        status
            .created_at
            .with_timezone(&Local)
            .format("%Y-%m-%d %H:%M:%S")
            .to_string(),
    )]);
}

fn display_agent_summary(agents: &[Agent], output: &OutputHandler) {
    output.section("Agent Summary");

    let total = agents.len();
    let ready = agents
        .iter()
        .filter(|a| matches!(a.status, AgentStatus::Ready))
        .count();
    let busy = agents
        .iter()
        .filter(|a| matches!(a.status, AgentStatus::Busy))
        .count();
    let idle = agents
        .iter()
        .filter(|a| matches!(a.status, AgentStatus::Idle))
        .count();
    let error = agents
        .iter()
        .filter(|a| matches!(a.status, AgentStatus::Error(_)))
        .count();
    let offline = agents
        .iter()
        .filter(|a| matches!(a.status, AgentStatus::Offline))
        .count();

    output.key_value(&[
        ("Total Agents".to_string(), total.to_string()),
        (
            "Ready".to_string(),
            format!("{} ({}%)", ready, ready * 100 / total.max(1)),
        ),
        (
            "Busy".to_string(),
            format!("{} ({}%)", busy, busy * 100 / total.max(1)),
        ),
        (
            "Idle".to_string(),
            format!("{} ({}%)", idle, idle * 100 / total.max(1)),
        ),
        (
            "Error".to_string(),
            format!("{} ({}%)", error, error * 100 / total.max(1)),
        ),
        (
            "Offline".to_string(),
            format!("{} ({}%)", offline, offline * 100 / total.max(1)),
        ),
    ]);

    // Agent type breakdown
    let mut type_counts: HashMap<String, usize> = HashMap::new();
    for agent in agents {
        *type_counts.entry(agent.agent_type.clone()).or_insert(0) += 1;
    }

    if !type_counts.is_empty() {
        output.info("\nAgent Types:");
        let type_list: Vec<String> = type_counts
            .iter()
            .map(|(t, c)| format!("{}: {}", t, c))
            .collect();
        output.list(&type_list, false);
    }
}

fn display_task_summary(tasks: &[Task], output: &OutputHandler) {
    output.section("Task Summary");

    let total = tasks.len();
    let pending = tasks
        .iter()
        .filter(|t| matches!(t.status, TaskStatus::Pending))
        .count();
    let running = tasks
        .iter()
        .filter(|t| matches!(t.status, TaskStatus::Running))
        .count();
    let completed = tasks
        .iter()
        .filter(|t| matches!(t.status, TaskStatus::Completed))
        .count();
    let failed = tasks
        .iter()
        .filter(|t| matches!(t.status, TaskStatus::Failed(_)))
        .count();
    let timeout = tasks
        .iter()
        .filter(|t| matches!(t.status, TaskStatus::Timeout))
        .count();

    output.key_value(&[
        ("Total Tasks".to_string(), total.to_string()),
        ("Pending".to_string(), pending.to_string()),
        ("Running".to_string(), running.to_string()),
        (
            "Completed".to_string(),
            format!("{} ({}%)", completed, completed * 100 / total.max(1)),
        ),
        (
            "Failed".to_string(),
            format!("{} ({}%)", failed, failed * 100 / total.max(1)),
        ),
        ("Timeout".to_string(), timeout.to_string()),
    ]);

    if total > 0 {
        let success_rate = completed as f32 / (completed + failed + timeout).max(1) as f32 * 100.0;
        output.print_status(
            "Success Rate",
            &format!("{:.1}%", success_rate),
            if success_rate >= 90.0 {
                StatusLevel::Good
            } else if success_rate >= 70.0 {
                StatusLevel::Warning
            } else {
                StatusLevel::Error
            },
        );
    }
}

fn display_detailed_agents(agents: &[Agent], tasks: &[Task], output: &OutputHandler) {
    output.section("Agent Details");

    let headers = vec![
        "Name", "Type", "Status", "Tasks", "Success", "Uptime", "Memory", "CPU",
    ];
    let mut rows = Vec::new();

    for agent in agents {
        let uptime = Utc::now() - agent.created_at;
        let uptime_str = format_duration(uptime);

        let success_rate = if agent.metrics.tasks_completed > 0 {
            let total = agent.metrics.tasks_completed + agent.metrics.tasks_failed;
            format!(
                "{:.0}%",
                agent.metrics.tasks_completed as f32 / total as f32 * 100.0
            )
        } else {
            "N/A".to_string()
        };

        let status_str = match &agent.status {
            AgentStatus::Ready => "Ready".green().to_string(),
            AgentStatus::Busy => "Busy".yellow().to_string(),
            AgentStatus::Idle => "Idle".blue().to_string(),
            AgentStatus::Error(e) => format!("Error: {}", e).red().to_string(),
            AgentStatus::Offline => "Offline".red().to_string(),
            AgentStatus::Initializing => "Initializing".yellow().to_string(),
        };

        rows.push(vec![
            agent.name.clone(),
            agent.agent_type.clone(),
            status_str,
            format!(
                "{}/{}",
                agent.metrics.tasks_completed,
                agent.metrics.tasks_completed + agent.metrics.tasks_failed
            ),
            success_rate,
            uptime_str,
            format!("{:.1}MB", agent.metrics.memory_usage_mb),
            format!("{:.1}%", agent.metrics.cpu_usage_percent),
        ]);
    }

    output.print_table(headers, rows);
}

fn display_active_tasks(tasks: &[Task], output: &OutputHandler) {
    let active_tasks: Vec<&Task> = tasks
        .iter()
        .filter(|t| matches!(t.status, TaskStatus::Running | TaskStatus::Pending))
        .collect();

    if active_tasks.is_empty() {
        return;
    }

    output.section("Active Tasks");

    let headers = vec![
        "ID", "Strategy", "Priority", "Status", "Agents", "Progress", "Duration",
    ];
    let mut rows = Vec::new();

    for task in active_tasks {
        let duration = if let Some(started) = task.started_at {
            format_duration(Utc::now() - started)
        } else {
            "Not started".to_string()
        };

        let progress = if !task.subtasks.is_empty() {
            let completed = task
                .subtasks
                .iter()
                .filter(|s| matches!(s.status, TaskStatus::Completed))
                .count();
            format!("{}/{}", completed, task.subtasks.len())
        } else {
            "0/0".to_string()
        };

        let status_str = match &task.status {
            TaskStatus::Pending => "Pending".yellow().to_string(),
            TaskStatus::Running => "Running".green().to_string(),
            _ => format!("{:?}", task.status),
        };

        rows.push(vec![
            task.id[..8].to_string(),
            format!("{:?}", task.strategy),
            task.priority.to_string(),
            status_str,
            task.assigned_agents.len().to_string(),
            progress,
            duration,
        ]);
    }

    output.print_table(headers, rows);
}

fn display_performance_metrics(agents: &[Agent], tasks: &[Task], output: &OutputHandler) {
    output.section("Performance Metrics");

    // Calculate aggregate metrics
    let total_tasks_completed: u64 = agents.iter().map(|a| a.metrics.tasks_completed).sum();
    let total_tasks_failed: u64 = agents.iter().map(|a| a.metrics.tasks_failed).sum();
    let avg_task_duration: u64 = if !agents.is_empty() {
        agents
            .iter()
            .map(|a| a.metrics.avg_task_duration_ms)
            .sum::<u64>()
            / agents.len() as u64
    } else {
        0
    };

    output.key_value(&[
        (
            "Total Tasks Completed".to_string(),
            total_tasks_completed.to_string(),
        ),
        (
            "Total Tasks Failed".to_string(),
            total_tasks_failed.to_string(),
        ),
        (
            "Average Task Duration".to_string(),
            format!("{}ms", avg_task_duration),
        ),
    ]);

    // Top performing agents
    let mut agent_scores: Vec<(&Agent, f32)> = agents
        .iter()
        .map(|a| (a, calculate_agent_score(&a.metrics)))
        .collect();
    agent_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    if !agent_scores.is_empty() {
        output.info("\nTop Performing Agents:");
        let top_agents: Vec<String> = agent_scores
            .iter()
            .take(5)
            .map(|(a, score)| format!("{} ({}) - Score: {:.1}", a.name, a.agent_type, score))
            .collect();
        output.list(&top_agents, true);
    }
}

fn display_resource_usage(agents: &[Agent], output: &OutputHandler) {
    output.section("Resource Usage");

    let total_memory: f64 = agents.iter().map(|a| a.metrics.memory_usage_mb).sum();
    let avg_cpu: f64 = if !agents.is_empty() {
        agents
            .iter()
            .map(|a| a.metrics.cpu_usage_percent)
            .sum::<f64>()
            / agents.len() as f64
    } else {
        0.0
    };

    output.key_value(&[
        (
            "Total Memory Usage".to_string(),
            format!("{:.1} MB", total_memory),
        ),
        ("Average CPU Usage".to_string(), format!("{:.1}%", avg_cpu)),
    ]);

    // High resource consumers
    let high_memory: Vec<&Agent> = agents
        .iter()
        .filter(|a| a.metrics.memory_usage_mb > 100.0)
        .collect();

    let high_cpu: Vec<&Agent> = agents
        .iter()
        .filter(|a| a.metrics.cpu_usage_percent > 50.0)
        .collect();

    if !high_memory.is_empty() {
        output.warning("\nHigh Memory Usage:");
        let mem_list: Vec<String> = high_memory
            .iter()
            .map(|a| format!("{}: {:.1} MB", a.name, a.metrics.memory_usage_mb))
            .collect();
        output.list(&mem_list, false);
    }

    if !high_cpu.is_empty() {
        output.warning("\nHigh CPU Usage:");
        let cpu_list: Vec<String> = high_cpu
            .iter()
            .map(|a| format!("{}: {:.1}%", a.name, a.metrics.cpu_usage_percent))
            .collect();
        output.list(&cpu_list, false);
    }
}

fn display_alerts(
    agents: &[Agent],
    tasks: &[Task],
    config: &crate::config::Config,
    output: &OutputHandler,
) {
    let mut alerts = Vec::new();

    // Check for offline agents
    let offline_count = agents
        .iter()
        .filter(|a| matches!(a.status, AgentStatus::Offline))
        .count();

    if offline_count > 0 {
        alerts.push(format!("{} agent(s) are offline", offline_count));
    }

    // Check for agents in error state
    let error_agents: Vec<&Agent> = agents
        .iter()
        .filter(|a| matches!(a.status, AgentStatus::Error(_)))
        .collect();

    if !error_agents.is_empty() {
        alerts.push(format!("{} agent(s) in error state", error_agents.len()));
    }

    // Check for high failure rate
    let total_tasks = tasks.len();
    let failed_tasks = tasks
        .iter()
        .filter(|t| matches!(t.status, TaskStatus::Failed(_)))
        .count();

    if total_tasks > 10
        && failed_tasks as f32 / total_tasks as f32 * 100.0
            > config.monitoring.alerts.failure_rate_threshold
    {
        alerts.push(format!(
            "High task failure rate: {:.1}%",
            failed_tasks as f32 / total_tasks as f32 * 100.0
        ));
    }

    // Check for resource alerts
    for agent in agents {
        if agent.metrics.cpu_usage_percent > config.monitoring.alerts.cpu_threshold as f64 {
            alerts.push(format!(
                "Agent '{}' CPU usage exceeds threshold: {:.1}%",
                agent.name, agent.metrics.cpu_usage_percent
            ));
        }

        if agent.metrics.memory_usage_mb
            > config.agent.memory_limit as f64 * (config.monitoring.alerts.memory_threshold as f64)
                / 100.0
        {
            alerts.push(format!(
                "Agent '{}' memory usage near limit: {:.1} MB",
                agent.name, agent.metrics.memory_usage_mb
            ));
        }
    }

    if !alerts.is_empty() {
        output.section("Alerts");
        for alert in alerts {
            output.warning(&alert);
        }
    }
}

async fn load_current_swarm(output: &OutputHandler) -> Result<crate::commands::init::SwarmInit> {
    let config_dir = directories::ProjectDirs::from("com", "ruv-fann", "ruv-swarm")
        .map(|dirs| dirs.data_local_dir().to_path_buf())
        .unwrap_or_else(|| Path::new(".").to_path_buf());

    let current_file = config_dir.join("current-swarm.json");

    if !current_file.exists() {
        output.error("No active swarm found. Run 'ruv-swarm init' first.");
        return Err(anyhow::anyhow!("No active swarm"));
    }

    let content = std::fs::read_to_string(current_file)?;
    serde_json::from_str(&content).context("Failed to parse swarm configuration")
}

async fn load_agents(swarm_config: &crate::commands::init::SwarmInit) -> Result<Vec<Agent>> {
    let config_dir = directories::ProjectDirs::from("com", "ruv-fann", "ruv-swarm")
        .map(|dirs| dirs.data_local_dir().to_path_buf())
        .unwrap_or_else(|| Path::new(".").to_path_buf());

    let agents_file = config_dir.join(format!("agents-{}.json", swarm_config.swarm_id));

    if agents_file.exists() {
        let content = std::fs::read_to_string(&agents_file)?;
        Ok(serde_json::from_str(&content).unwrap_or_default())
    } else {
        Ok(Vec::new())
    }
}

async fn load_tasks() -> Result<Vec<Task>> {
    let tasks_dir = directories::ProjectDirs::from("com", "ruv-fann", "ruv-swarm")
        .map(|dirs| dirs.data_local_dir().join("tasks"))
        .unwrap_or_else(|| Path::new("./tasks").to_path_buf());

    let mut tasks = Vec::new();

    if tasks_dir.exists() {
        for entry in std::fs::read_dir(tasks_dir)? {
            let entry = entry?;
            if entry
                .path()
                .extension()
                .map(|e| e == "json")
                .unwrap_or(false)
            {
                if let Ok(content) = std::fs::read_to_string(entry.path()) {
                    if let Ok(task) = serde_json::from_str::<Task>(&content) {
                        tasks.push(task);
                    }
                }
            }
        }
    }

    Ok(tasks)
}

fn format_duration(duration: chrono::Duration) -> String {
    let days = duration.num_days();
    let hours = duration.num_hours() % 24;
    let minutes = duration.num_minutes() % 60;

    if days > 0 {
        format!("{}d {}h {}m", days, hours, minutes)
    } else if hours > 0 {
        format!("{}h {}m", hours, minutes)
    } else {
        format!("{}m", minutes)
    }
}

fn calculate_swarm_health(status: &SwarmStatus) -> f32 {
    let mut score = 100.0;

    // Deduct points for inactive agents
    let inactive_ratio = 1.0 - (status.active_agents as f32 / status.total_agents.max(1) as f32);
    score -= inactive_ratio * 30.0;

    // Deduct points for failed tasks
    let failure_ratio = status.failed_tasks as f32 / status.total_tasks.max(1) as f32;
    score -= failure_ratio * 40.0;

    // Bonus for running tasks
    if status.running_tasks > 0 {
        score += 5.0;
    }

    score.max(0.0).min(100.0)
}

fn calculate_agent_score(metrics: &AgentMetrics) -> f32 {
    let total_tasks = metrics.tasks_completed + metrics.tasks_failed;
    if total_tasks == 0 {
        return 0.0;
    }

    let success_rate = metrics.tasks_completed as f32 / total_tasks as f32;
    let efficiency = 1.0 / (1.0 + metrics.avg_task_duration_ms as f32 / 1000.0);
    let resource_efficiency = 1.0
        - (metrics.cpu_usage_percent as f32 / 100.0 + metrics.memory_usage_mb as f32 / 512.0) / 2.0;

    (success_rate * 0.5 + efficiency * 0.3 + resource_efficiency * 0.2) * 100.0
}
