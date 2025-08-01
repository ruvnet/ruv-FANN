//! Tests for query builder and advanced queries

use crate::models::*;
use crate::*;

#[test]
fn test_basic_query_builder() {
    let (query, params) = QueryBuilder::<AgentModel>::new("agents").build();

    assert_eq!(query, "SELECT * FROM agents");
    assert_eq!(params.len(), 0);
}

#[test]
fn test_query_with_conditions() {
    let (query, params) = QueryBuilder::<AgentModel>::new("agents")
        .where_eq("status", "running")
        .where_eq("agent_type", "compute")
        .build();

    assert_eq!(
        query,
        "SELECT * FROM agents WHERE status = ? AND agent_type = ?"
    );
    assert_eq!(params.len(), 2);
    assert_eq!(params[0], "running");
    assert_eq!(params[1], "compute");
}

#[test]
fn test_query_with_like_pattern() {
    let (query, params) = QueryBuilder::<TaskModel>::new("tasks")
        .where_like("task_type", "train%")
        .where_eq("status", "pending")
        .build();

    assert_eq!(
        query,
        "SELECT * FROM tasks WHERE task_type LIKE ? AND status = ?"
    );
    assert_eq!(params.len(), 2);
    assert_eq!(params[0], "train%");
    assert_eq!(params[1], "pending");
}

#[test]
fn test_query_with_comparison() {
    let (query, params) = QueryBuilder::<MetricModel>::new("metrics")
        .where_gt("value", 90)
        .where_eq("metric_type", "accuracy")
        .build();

    assert_eq!(
        query,
        "SELECT * FROM metrics WHERE value > 90 AND metric_type = ?"
    );
    assert_eq!(params.len(), 1);
    assert_eq!(params[0], "accuracy");
}

#[test]
fn test_query_with_ordering() {
    let (query, params) = QueryBuilder::<EventModel>::new("events")
        .where_eq("event_type", "task_completed")
        .order_by("timestamp", true)
        .build();

    assert_eq!(
        query,
        "SELECT * FROM events WHERE event_type = ? ORDER BY timestamp DESC"
    );
    assert_eq!(params.len(), 1);
    assert_eq!(params[0], "task_completed");
}

#[test]
fn test_query_with_limit_offset() {
    let (query, params) = QueryBuilder::<MessageModel>::new("messages")
        .where_eq("read", "false")
        .order_by("timestamp", false)
        .limit(10)
        .offset(20)
        .build();

    assert_eq!(
        query,
        "SELECT * FROM messages WHERE read = ? ORDER BY timestamp ASC LIMIT 10 OFFSET 20"
    );
    assert_eq!(params.len(), 1);
    assert_eq!(params[0], "false");
}

#[test]
fn test_complex_query() {
    let (query, params) = QueryBuilder::<AgentModel>::new("agents")
        .where_eq("status", "running")
        .where_like("capabilities", "%neural%")
        .where_gt("created_at", 1000000)
        .order_by("updated_at", true)
        .limit(50)
        .build();

    assert_eq!(
        query,
        "SELECT * FROM agents WHERE status = ? AND capabilities LIKE ? AND created_at > 1000000 ORDER BY updated_at DESC LIMIT 50"
    );
    assert_eq!(params.len(), 2);
    assert_eq!(params[0], "running");
    assert_eq!(params[1], "%neural%");
}

#[test]
fn test_query_builder_chaining() {
    let base_query = QueryBuilder::<TaskModel>::new("tasks");

    let filtered_query = base_query
        .where_eq("priority", "10")
        .where_eq("assigned_to", "agent-123");

    let (final_query, params) = filtered_query
        .order_by("created_at", false)
        .limit(5)
        .build();

    assert_eq!(
        final_query,
        "SELECT * FROM tasks WHERE priority = ? AND assigned_to = ? ORDER BY created_at ASC LIMIT 5"
    );
    assert_eq!(params.len(), 2);
    assert_eq!(params[0], "10");
    assert_eq!(params[1], "agent-123");
}

#[test]
fn test_query_injection_prevention() {
    // Test that special characters are handled properly
    let (query, params) = QueryBuilder::<AgentModel>::new("agents")
        .where_eq("name", "Agent'; DROP TABLE agents; --")
        .build();

    // The query builder should use parameterized queries
    assert_eq!(query, "SELECT * FROM agents WHERE name = ?");
    assert_eq!(params.len(), 1);
    assert_eq!(params[0], "Agent'; DROP TABLE agents; --");
}

#[test]
fn test_empty_conditions() {
    let (query, params) = QueryBuilder::<AgentModel>::new("agents")
        .order_by("id", false)
        .limit(100)
        .build();

    assert_eq!(query, "SELECT * FROM agents ORDER BY id ASC LIMIT 100");
    assert_eq!(params.len(), 0);
}

#[test]
fn test_pagination_queries() {
    let page_size = 20;
    let queries: Vec<(String, Vec<String>)> = (0..5)
        .map(|page| {
            QueryBuilder::<TaskModel>::new("tasks")
                .where_eq("status", "completed")
                .order_by("completed_at", true)
                .limit(page_size)
                .offset(page * page_size)
                .build()
        })
        .collect();

    // Verify pagination offsets
    assert!(queries[0].0.contains("OFFSET 0"));
    assert!(queries[1].0.contains("OFFSET 20"));
    assert!(queries[2].0.contains("OFFSET 40"));
    assert!(queries[3].0.contains("OFFSET 60"));
    assert!(queries[4].0.contains("OFFSET 80"));
}

#[test]
fn test_aggregate_query_patterns() {
    // While the basic QueryBuilder doesn't support aggregates,
    // test patterns that would be used for aggregate queries

    let (metrics_query, params) = QueryBuilder::<MetricModel>::new("metrics")
        .where_eq("metric_type", "performance")
        .where_gt("timestamp", 1000000)
        .build();

    // In a real implementation, this might be extended to:
    // SELECT AVG(value), COUNT(*) FROM metrics WHERE ...
    assert!(metrics_query.starts_with("SELECT * FROM metrics"));
    assert_eq!(params.len(), 1);
    assert_eq!(params[0], "performance");
}

#[test]
fn test_join_query_patterns() {
    // Test query patterns that might be used for joins
    let agents_with_tasks = QueryBuilder::<AgentModel>::new("agents")
        .where_eq("status", "running")
        .build();

    let tasks_for_agents = QueryBuilder::<TaskModel>::new("tasks")
        .where_eq("status", "assigned")
        .build();

    // In a real implementation, these might be combined into:
    // SELECT * FROM agents JOIN tasks ON agents.id = tasks.assigned_to
    assert!(agents_with_tasks.0.contains("agents"));
    assert!(tasks_for_agents.0.contains("tasks"));
}

#[test]
fn test_subquery_patterns() {
    // Test patterns for subqueries
    let active_agents = QueryBuilder::<AgentModel>::new("agents")
        .where_eq("status", "running")
        .build();

    // This could be used as a subquery in:
    // SELECT * FROM tasks WHERE assigned_to IN (SELECT id FROM agents WHERE status = 'running')
    assert!(active_agents.0.contains("WHERE status = ?"));
}

#[test]
fn test_date_range_queries() {
    let start_timestamp = 1000000;
    let _end_timestamp = 2000000;

    // Pattern for date range queries (would need additional methods in real implementation)
    let (events_in_range, _) = QueryBuilder::<EventModel>::new("events")
        .where_gt("timestamp", start_timestamp)
        .build();

    // Would ideally support: .where_between("timestamp", start, end)
    assert!(events_in_range.contains(&format!("timestamp > {}", start_timestamp)));
}

#[cfg(feature = "proptest")]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_query_builder_consistency(
            table in "[a-z_]{1,20}",
            field1 in "[a-z_]{1,20}",
            value1 in "[a-zA-Z0-9]{1,20}",
            field2 in "[a-z_]{1,20}",
            value2 in "[a-zA-Z0-9]{1,20}",
            order_field in "[a-z_]{1,20}",
            desc in any::<bool>(),
            limit in 1usize..1000,
            offset in 0usize..1000,
        ) {
            let query = QueryBuilder::<AgentModel>::new(&table)
                .where_eq(&field1, &value1)
                .where_eq(&field2, &value2)
                .order_by(&order_field, desc)
                .limit(limit)
                .offset(offset)
                .build();

            // Destructure tuple returned by parameterized query builder
            let (query_sql, params) = query;

            // Verify query structure
            assert!(query_sql.starts_with("SELECT * FROM"));
            assert!(query_sql.contains(&table));
            assert!(query_sql.contains("WHERE"));
            assert!(query_sql.contains(&field1));
            assert!(query_sql.contains(&field2));
            assert!(query_sql.contains("ORDER BY"));
            assert!(query_sql.contains(&order_field));
            assert!(query_sql.contains(if desc { "DESC" } else { "ASC" }));
            assert!(query_sql.contains(&format!("LIMIT {}", limit)));
            assert!(query_sql.contains(&format!("OFFSET {}", offset)));
            
            // Verify parameters are properly set (should have ? placeholders instead of direct values)
            assert!(!query_sql.contains(&value1), "SQL should use ? placeholders, not direct values");
            assert!(!query_sql.contains(&value2), "SQL should use ? placeholders, not direct values");
            assert_eq!(params.len(), 2, "Should have 2 parameter values");
            assert!(params.contains(&value1), "Parameters should contain value1");
            assert!(params.contains(&value2), "Parameters should contain value2");
        }

        #[test]
        fn test_query_condition_count(
            conditions in prop::collection::vec(
                ("[a-z_]{1,10}", "[a-zA-Z0-9]{1,20}"),
                0..10
            )
        ) {
            let mut builder = QueryBuilder::<AgentModel>::new("test_table");

            for (field, value) in &conditions {
                builder = builder.where_eq(field, value);
            }

            let (query_sql, _params) = builder.build();

            if conditions.is_empty() {
                assert!(!query_sql.contains("WHERE"));
            } else {
                assert!(query_sql.contains("WHERE"));
                // Count ANDs - should be one less than number of conditions
                let and_count = query_sql.matches(" AND ").count();
                assert_eq!(and_count, conditions.len().saturating_sub(1));
            }
        }
    }
}
