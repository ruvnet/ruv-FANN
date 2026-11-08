//! Quota updates must enforce limits even when callers race after a preflight.
use crate::limits::{ResourceLimiter, ResourceLimits};
use std::sync::Arc;
use uuid::Uuid;

#[tokio::test]
async fn concurrent_agent_reservations_are_atomic() {
    let limiter = Arc::new(ResourceLimiter::new(ResourceLimits {
        max_agents_per_session: 3,
        ..Default::default()
    }));
    let session = Uuid::new_v4();
    limiter.init_session(session).await;
    let mut workers = vec![];
    for _ in 0..30 {
        let limiter = limiter.clone();
        workers.push(tokio::spawn(async move {
            limiter.increment_agents(&session).await.is_ok()
        }));
    }
    let mut accepted = 0;
    for worker in workers {
        accepted += usize::from(worker.await.unwrap());
    }
    assert_eq!(accepted, 3);
    assert_eq!(
        limiter
            .get_session_stats(&session)
            .await
            .unwrap()
            .agent_count,
        3
    );
    limiter.decrement_agents(&session).await.unwrap();
    assert!(limiter.increment_agents(&session).await.is_ok());
}

#[tokio::test]
async fn memory_updates_enforce_bytes_entries_and_overflow() {
    let limiter = ResourceLimiter::new(ResourceLimits {
        max_memory_per_session: 10,
        max_memory_value_size: usize::MAX,
        max_memory_entries: 2,
        ..Default::default()
    });
    let session = Uuid::new_v4();
    limiter.init_session(session).await;
    limiter
        .update_memory_usage(&session, 6, true)
        .await
        .unwrap();
    assert!(limiter
        .update_memory_usage(&session, usize::MAX, true)
        .await
        .is_err());
    assert!(limiter
        .check_memory_limit(&session, usize::MAX)
        .await
        .is_err());
    assert!(limiter
        .update_memory_usage(&session, 5, true)
        .await
        .is_err());
    limiter
        .update_memory_usage(&session, 4, true)
        .await
        .unwrap();
    assert!(limiter
        .update_memory_usage(&session, 0, true)
        .await
        .is_err());
    let stats = limiter.get_session_stats(&session).await.unwrap();
    assert_eq!((stats.memory_usage, stats.memory_entries), (10, 2));
}

#[test]
fn default_network_limits_fit_small_server_budgets() {
    let limits = ResourceLimits::default();
    assert_eq!(limits.max_message_size, 1024 * 1024);
    assert_eq!(limits.max_memory_per_session, 64 * 1024 * 1024);
    assert_eq!(limits.max_agents_per_session, 100);
}
