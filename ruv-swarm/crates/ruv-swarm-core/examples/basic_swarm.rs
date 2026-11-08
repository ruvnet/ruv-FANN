//! Run with `cargo run -p ruv-swarm-core --example basic_swarm`.
use ruv_swarm_core::{DistributionStrategy, DynamicAgent, Swarm, SwarmConfig, Task};

#[tokio::main]
async fn main() -> ruv_swarm_core::Result<()> {
    let mut swarm = Swarm::new(SwarmConfig {
        distribution_strategy: DistributionStrategy::RoundRobin,
        ..Default::default()
    });
    for id in ["worker-a", "worker-b"] {
        swarm.register_agent(DynamicAgent::new(id, vec!["summarize".into()]))?;
    }
    swarm.start_all_agents().await?;
    for id in ["report-1", "report-2", "report-3"] {
        swarm.submit_task(Task::new(id, "summarize").require_capability("summarize"))?;
    }
    // The scheduler returns assignments. Your application runs each worker.
    for (task, agent) in swarm.distribute_tasks().await? {
        println!("{task} -> {agent}");
    }
    Ok(())
}
