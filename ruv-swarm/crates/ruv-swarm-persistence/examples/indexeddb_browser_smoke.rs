//! Actual browser integration fixture, built as WASM and driven by tests/browser-smoke.cjs.
#[cfg(target_arch = "wasm32")]
mod browser {
    use ruv_swarm_persistence::{models::*, Storage, StorageError, WasmMutation, WasmStorage};
    use serde_json::json;
    use wasm_bindgen::prelude::*;

    fn require(value: bool, message: &str) -> Result<(), StorageError> {
        if value {
            Ok(())
        } else {
            Err(StorageError::Other(message.into()))
        }
    }
    async fn verify() -> Result<serde_json::Value, StorageError> {
        let name = format!("ruv-persistence-browser-{}", uuid::Uuid::new_v4());
        let storage = WasmStorage::open(&name).await?;
        let mut checks = Vec::new();
        let mut agent = AgentModel::new(
            "browser-agent".into(),
            "worker".into(),
            vec!["persist".into()],
        );
        agent.status = AgentStatus::Active;
        agent
            .metadata
            .insert("large-integer".into(), json!(u64::MAX));
        agent.metadata.insert("unicode".into(), json!("persist 🦀"));
        storage.store_agent(&agent).await?;
        require(
            storage.get_agent(&agent.id).await? == Some(agent.clone()),
            "Agent roundtrip lost model data or integer precision",
        )?;
        require(
            storage.get_agent("missing").await?.is_none(),
            "Missing record must return None",
        )?;
        require(
            storage.list_agents_by_status("active").await?.len() == 1,
            "Agent status index failed",
        )?;
        checks.push("model/index/u64/unicode roundtrip");

        let high = TaskModel::new("high".into(), json!({}), TaskPriority::High);
        let critical = TaskModel::new("critical".into(), json!({}), TaskPriority::Critical);
        storage
            .write_batch(&[
                WasmMutation::Task(high.clone()),
                WasmMutation::Task(critical.clone()),
            ])
            .await?;
        let pending = storage.get_pending_tasks().await?;
        require(
            pending.len() == 2 && pending[0].id == critical.id,
            "Pending priority order failed",
        )?;
        let second_connection = WasmStorage::open(&name).await?;
        let (one, two) = futures::future::join(
            storage.claim_task(&high.id, "first"),
            second_connection.claim_task(&high.id, "second"),
        )
        .await;
        require(
            u8::from(one?) + u8::from(two?) == 1,
            "Racing task claims must have exactly one winner",
        )?;
        let claimed = storage
            .get_task(&high.id)
            .await?
            .ok_or_else(|| StorageError::Other("Claimed task missing".into()))?;
        require(
            claimed.status == TaskStatus::Assigned && claimed.assigned_to.is_some(),
            "Claim state not persisted",
        )?;
        checks.push("pending order and atomic racing claims");

        let mut first = EventModel::new("change".into(), json!({"integer":u64::MAX}))
            .with_agent(agent.id.clone());
        first.timestamp = chrono::Utc::now() - chrono::Duration::seconds(2);
        storage.store_event(&first).await?;
        require(
            storage.store_event(&first).await.is_err(),
            "Duplicate event must fail",
        )?;
        let mut second = EventModel::new("change".into(), json!({})).with_agent(agent.id.clone());
        second.timestamp = first.timestamp + chrono::Duration::seconds(1);
        storage.store_event(&second).await?;
        let events = storage.get_events_since(0).await?;
        require(
            events.len() == 2 && events[0].sequence == 1 && events[1].sequence == 2,
            "Event counter failed or aborted increment leaked",
        )?;
        require(
            events[0].payload["integer"].as_u64() == Some(u64::MAX),
            "Event u64 was rounded by JavaScript",
        )?;
        require(
            storage.get_events_by_agent(&agent.id, 1).await?.len() == 1,
            "Event query limit failed",
        )?;
        require(
            storage.get_events_by_type("change", 10).await?.len() == 2,
            "Event type index failed",
        )?;
        checks.push("event sequence and aborted counter rollback");

        let message = MessageModel::new(
            "first".into(),
            "second".into(),
            "data".into(),
            json!({"value":7}),
        );
        storage.store_message(&message).await?;
        require(
            storage.get_unread_messages("second").await?.len() == 1,
            "Unread message query failed",
        )?;
        storage.mark_message_read(&message.id).await?;
        require(
            storage.get_unread_messages("second").await?.is_empty(),
            "Read flag was not persisted",
        )?;
        require(
            storage
                .get_messages_between("first", "second", 1)
                .await?
                .len()
                == 1,
            "Conversation query failed",
        )?;
        checks.push("message query and atomic mark-read");

        for (value, unit) in [(2.0, "percent"), (4.0, "percent"), (10.0, "bytes")] {
            storage
                .store_metric(
                    &MetricModel::new("cpu".into(), value, unit.into())
                        .with_agent(agent.id.clone()),
                )
                .await?;
        }
        let metrics = storage.get_aggregated_metrics("cpu", 0, i64::MAX).await?;
        let percent = metrics
            .iter()
            .find(|m| m.unit == "percent")
            .ok_or_else(|| StorageError::Other("Metric group missing".into()))?;
        require(
            metrics.len() == 2
                && percent.value == 3.0
                && percent.tags.get("count").map(String::as_str) == Some("2"),
            "Metric aggregate lost groups or samples",
        )?;
        checks.push("metric grouping and aggregation");

        let rollback_agent = AgentModel::new("must-rollback".into(), "worker".into(), vec![]);
        let invalid = MetricModel::new("invalid".into(), f64::NAN, "units".into());
        require(
            storage
                .write_batch(&[
                    WasmMutation::Agent(rollback_agent.clone()),
                    WasmMutation::Metric(invalid),
                ])
                .await
                .is_err(),
            "Invalid batch must fail",
        )?;
        require(
            storage.get_agent(&rollback_agent.id).await?.is_none(),
            "Aborted batch partially committed",
        )?;
        require(
            storage.begin_transaction().await.is_err(),
            "Unsupported generic transaction must not pretend success",
        )?;
        storage.checkpoint().await?;
        require(
            storage.get_storage_size().await? > 0,
            "Storage size must measure stored records",
        )?;
        checks.push("atomic multi-store batch rollback and commit barrier");

        drop(second_connection);
        drop(storage);
        let reopened = WasmStorage::open(&name).await?;
        require(
            reopened.get_agent(&agent.id).await? == Some(agent.clone()),
            "Reopen lost committed data",
        )?;
        reopened.delete_agent(&agent.id).await?;
        require(
            reopened.get_agent(&agent.id).await?.is_none(),
            "Delete was not persisted",
        )?;
        drop(reopened);
        rexie::Rexie::delete(&name)
            .await
            .map_err(|e| StorageError::Database(format!("{e:?}")))?;
        checks.push("close/reopen persistence and deletion");
        Ok(json!({"checks":checks,"database_deleted":true,"claim_winners":1}))
    }
    #[wasm_bindgen]
    pub async fn run_persistence_smoke() -> Result<JsValue, JsValue> {
        verify()
            .await
            .map(|result| JsValue::from_str(&result.to_string()))
            .map_err(|error| JsValue::from_str(&error.to_string()))
    }
}
