//! Verify public scheduling contracts over real DynamicAgents.
use crate::async_swarm::{AsyncSwarm, AsyncSwarmConfig};
use crate::{
    agent::DynamicAgent,
    swarm::{Swarm, SwarmConfig},
    task::{DistributionStrategy, Task, TaskPriority},
};

#[tokio::test]
async fn synchronous_round_robin_persists_between_batches() {
    let mut swarm = Swarm::new(SwarmConfig {
        distribution_strategy: DistributionStrategy::RoundRobin,
        ..Default::default()
    });
    for id in ["a", "b", "c"] {
        swarm.register_agent(DynamicAgent::new(id, vec![])).unwrap();
    }
    swarm.start_all_agents().await.unwrap();
    let mut assigned = vec![];
    for id in 0..7 {
        swarm
            .submit_task(Task::new(id.to_string(), "work"))
            .unwrap();
        assigned.push(swarm.distribute_tasks().await.unwrap().remove(0).1);
    }
    assert_eq!(assigned, ["a", "b", "c", "a", "b", "c", "a"]);
}

#[tokio::test]
async fn asynchronous_round_robin_persists_between_batches() {
    let swarm = AsyncSwarm::new(AsyncSwarmConfig {
        distribution_strategy: DistributionStrategy::RoundRobin,
        ..Default::default()
    });
    for id in ["a", "b", "c"] {
        swarm
            .register_agent(DynamicAgent::new(id, vec![]))
            .await
            .unwrap();
    }
    swarm.start_all_agents().await.unwrap();
    let mut assigned = vec![];
    for id in 0..7 {
        swarm
            .submit_task(Task::new(id.to_string(), "work"))
            .await
            .unwrap();
        assigned.push(swarm.distribute_tasks().await.unwrap().remove(0).1);
    }
    assert_eq!(assigned, ["a", "b", "c", "a", "b", "c", "a"]);
}

#[tokio::test]
async fn priority_is_stable_and_fifo_is_preserved_for_other_strategies() {
    let mut priority = Swarm::new(SwarmConfig {
        distribution_strategy: DistributionStrategy::Priority,
        ..Default::default()
    });
    let mut fifo = Swarm::new(SwarmConfig::default());
    for swarm in [&mut priority, &mut fifo] {
        swarm
            .register_agent(DynamicAgent::new("a", vec![]))
            .unwrap();
        swarm.start_all_agents().await.unwrap();
        for (id, level) in [
            ("low", TaskPriority::Low),
            ("high1", TaskPriority::High),
            ("high2", TaskPriority::High),
            ("normal", TaskPriority::Normal),
        ] {
            swarm
                .submit_task(Task::new(id, "work").with_priority(level))
                .unwrap();
        }
    }
    let ids = |assignments: Vec<(crate::task::TaskId, String)>| {
        assignments
            .into_iter()
            .map(|(id, _)| id.to_string())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        ids(priority.distribute_tasks().await.unwrap()),
        ["high1", "high2", "normal", "low"]
    );
    assert_eq!(
        ids(fifo.distribute_tasks().await.unwrap()),
        ["low", "high1", "high2", "normal"]
    );
}

#[tokio::test]
async fn duplicate_registration_cannot_replace_live_agent() {
    let mut swarm = Swarm::new(SwarmConfig::default());
    swarm
        .register_agent(DynamicAgent::new("a", vec!["original".into()]))
        .unwrap();
    assert!(swarm
        .register_agent(DynamicAgent::new("a", vec!["replacement".into()]))
        .is_err());
    let swarm = AsyncSwarm::new(AsyncSwarmConfig::default());
    swarm
        .register_agent(DynamicAgent::new("a", vec![]))
        .await
        .unwrap();
    assert!(swarm
        .register_agent(DynamicAgent::new("a", vec![]))
        .await
        .is_err());
    assert_eq!(swarm.metrics().await.total_agents, 1);
}
