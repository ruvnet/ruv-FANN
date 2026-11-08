use crate::ExecutionResult;
use anyhow::{bail, Context, Result};
use ruv_swarm_core::{task::Task, Agent};
use std::time::{Duration, Instant};

pub(crate) async fn execute<A>(agent: &mut A, task: Task) -> Result<ExecutionResult>
where
    A: Agent<Input = Task, Output = ExecutionResult>,
{
    let duration = Duration::from_millis(task.timeout_ms.unwrap_or(300_000).min(300_000));
    let start = Instant::now();
    let mut result = tokio::time::timeout(duration, agent.process(task))
        .await
        .context("Agent execution timed out")?
        .map_err(|error| anyhow::anyhow!("Agent execution failed: {error:?}"))?;
    if result.exit_code != 0 {
        bail!("Agent execution exited with code {}", result.exit_code);
    }
    const MAX_OUTPUT: usize = 16 * 1024 * 1024;
    if result.output.len() > MAX_OUTPUT || result.patch.len() > MAX_OUTPUT {
        bail!("Agent output or patch exceeds the 16 MiB limit");
    }
    result.duration = start.elapsed();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruv_swarm_agents::FunctionAgent;

    #[tokio::test]
    async fn calls_the_application_handler_and_uses_its_patch() {
        let mut agent = FunctionAgent::new("patcher", vec![], |task: Task| async move {
            assert_eq!(task.id.0, "issue-1");
            Ok::<_, &'static str>(ExecutionResult {
                output: "application output".into(),
                patch: "application patch".into(),
                exit_code: 0,
                duration: Duration::from_secs(9999),
            })
        })
        .unwrap();
        let result = execute(&mut agent, Task::new("issue-1", "swe-bench-evaluation"))
            .await
            .unwrap();
        assert_eq!(result.patch, "application patch");
        assert!(result.duration < Duration::from_secs(1));
    }

    #[tokio::test]
    async fn propagates_handler_failures_without_fabricating_results() {
        let mut agent = FunctionAgent::new("failure", vec![], |_: Task| async {
            Err::<ExecutionResult, _>("provider unavailable")
        })
        .unwrap();
        let error = execute(&mut agent, Task::new("issue-1", "swe-bench-evaluation"))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("provider unavailable"));
    }
}
