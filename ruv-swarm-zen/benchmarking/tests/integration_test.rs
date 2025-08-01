//! Integration tests for the benchmarking framework

use ruv_swarm_benchmarking::{
    stream_parser::ClaudeStreamEvent, BenchmarkConfig, BenchmarkScenario, BenchmarkingFramework,
    Difficulty, ExecutionMode,
};
use std::path::PathBuf;
use tempfile::tempdir;

#[tokio::test]
async fn test_framework_initialization() {
    let temp_dir = tempdir().unwrap();
    let config = BenchmarkConfig {
        database_path: temp_dir.path().join("test.db"),
        enable_real_time_monitoring: false,
        monitor_port: 0,
        claude_executable: PathBuf::from("echo"), // Use echo as dummy command
        execution_timeout: std::time::Duration::from_secs(5),
        trial_count: 1,
    };

    let framework = BenchmarkingFramework::new(config).await;
    assert!(framework.is_ok());
}

#[test]
fn test_stream_event_parsing() {
    // Test parsing various Claude stream events
    let tool_use_json = r#"{
        "type": "tool_use",
        "id": "test-id",
        "name": "Read",
        "input": {"file": "test.rs"}
    }"#;

    let event: Result<ClaudeStreamEvent, _> = serde_json::from_str(tool_use_json);
    assert!(event.is_ok());

    match event.unwrap() {
        ClaudeStreamEvent::ToolUse { name, .. } => {
            assert_eq!(name, "Read");
        }
        _ => panic!("Expected ToolUse event"),
    }
}

#[test]
fn test_scenario_creation() {
    let scenario = BenchmarkScenario {
        name: "test_scenario".to_string(),
        instance_id: "test-001".to_string(),
        repository: "test/repo".to_string(),
        issue_description: "Test issue".to_string(),
        difficulty: Difficulty::Easy,
        claude_command: "test command".to_string(),
        expected_files_modified: vec!["test.rs".to_string()],
        validation_tests: vec!["test_function".to_string()],
    };

    assert_eq!(scenario.name, "test_scenario");
    assert_eq!(scenario.difficulty, Difficulty::Easy);
}

#[test]
fn test_execution_modes() {
    assert_eq!(ExecutionMode::Baseline.to_string(), "baseline");
    assert_eq!(ExecutionMode::MLOptimized.to_string(), "ml_optimized");
}
