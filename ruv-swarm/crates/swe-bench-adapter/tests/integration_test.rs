//! Integration tests for SWE-Bench adapter

use std::path::PathBuf;
use swe_bench_adapter::{
    ClaudePromptGenerator, DifficultyLevel, InstanceLoader, PatchEvaluator, PromptConfig,
    SWEBenchAdapter, SWEBenchConfig, SWEBenchInstance, StreamParser,
};

#[tokio::test]
async fn test_instance_loader() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut loader = InstanceLoader::new(temp_dir.path()).unwrap();

    let fixture = serde_json::json!({
        "instance_id": "test-001", "repo": "test/repo", "version": "1.0",
        "issue_title": "Test issue", "issue_description": "Test description",
        "hints": [], "test_patch": "", "test_directives": [], "difficulty": "Medium",
        "metrics": { "files_changed": 2, "lines_changed": 50, "test_count": 5,
            "dependencies": 3, "complexity_score": 0.5, "domain_specific": false,
            "requires_external_api": false }, "metadata": {}
    });
    std::fs::write(
        temp_dir.path().join("test-001.json"),
        serde_json::to_vec(&fixture).unwrap(),
    )
    .unwrap();
    let result = loader.load_instance("test-001").await;
    assert!(result.is_ok());

    let instance = result.unwrap();
    assert_eq!(instance.instance_id, "test-001");
    assert_eq!(instance.difficulty, DifficultyLevel::Medium);
}

#[test]
fn test_prompt_generation() {
    use std::collections::HashMap;
    use swe_bench_adapter::loader::InstanceMetrics;

    let config = PromptConfig::default();
    let generator = ClaudePromptGenerator::new(config);

    let instance = SWEBenchInstance {
        instance_id: "test-001".to_string(),
        repo: "test/repo".to_string(),
        version: "1.0".to_string(),
        issue_title: "Test issue".to_string(),
        issue_description: "Test description".to_string(),
        hints: vec!["Test hint".to_string()],
        test_patch: "diff --git a/test.py b/test.py".to_string(),
        test_directives: vec!["pytest test.py".to_string()],
        difficulty: DifficultyLevel::Easy,
        metrics: InstanceMetrics {
            files_changed: 1,
            lines_changed: 10,
            test_count: 1,
            dependencies: 0,
            complexity_score: 0.1,
            domain_specific: false,
            requires_external_api: false,
        },
        metadata: HashMap::new(),
    };

    let result = generator.generate_prompt(&instance);
    assert!(result.is_ok());

    let prompt = result.unwrap();
    assert!(prompt.token_count > 0);
    assert!(prompt.content.contains("test/repo"));
}

#[test]
fn test_stream_parser() {
    let mut parser = StreamParser::new();

    let output = r#"
Starting execution...
<function_calls>
Read file: test.py
</function_calls>
Error: Test failed
Tokens: 500
"#;

    let result = parser.parse_stream(output);
    assert!(result.is_ok());

    let metrics = result.unwrap();
    assert_eq!(metrics.total_tokens, 500);
    // The parser counts both "<function_calls>" and lines with "Read"
    assert_eq!(metrics.tool_calls, 2);
    // File operations are not counted because "Read file:" doesn't match tool_call pattern on same line
    assert_eq!(metrics.file_operations.reads, 0);
    assert_eq!(metrics.errors.len(), 1);
}

#[tokio::test]
async fn test_adapter_creation() {
    let directory = tempfile::tempdir().unwrap();
    let config = SWEBenchConfig {
        instances_path: directory.path().join("instances"),
        memory_path: directory.path().join("memory.db"),
        ..Default::default()
    };

    let result = SWEBenchAdapter::new(config).await;
    assert!(result.is_ok());
}

#[test]
fn test_difficulty_categorization() {
    use swe_bench_adapter::loader::InstanceMetrics;

    let easy_metrics = InstanceMetrics {
        files_changed: 1,
        lines_changed: 10,
        test_count: 1,
        dependencies: 0,
        complexity_score: 0.1,
        domain_specific: false,
        requires_external_api: false,
    };

    let hard_metrics = InstanceMetrics {
        files_changed: 5,
        lines_changed: 150,
        test_count: 10,
        dependencies: 5,
        complexity_score: 0.8,
        domain_specific: true,
        requires_external_api: true,
    };

    assert_eq!(
        DifficultyLevel::from_metrics(&easy_metrics),
        DifficultyLevel::Easy
    );
    assert_eq!(
        DifficultyLevel::from_metrics(&hard_metrics),
        DifficultyLevel::Hard
    );
}

#[tokio::test]
async fn local_patch_evaluation_uses_explicit_repository_and_preserves_existing_files() {
    let directory = tempfile::tempdir().unwrap();
    let run_git = |arguments: &[&str]| {
        let output = std::process::Command::new("git")
            .current_dir(directory.path())
            .args(arguments)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    };
    run_git(&["init", "--quiet"]);
    std::fs::write(directory.path().join("example.txt"), "before\n").unwrap();
    run_git(&["add", "example.txt"]);
    run_git(&[
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "-m",
        "fixture",
    ]);
    let sentinel = directory.path().join(".swe_bench_patch.diff");
    std::fs::write(&sentinel, "existing application file").unwrap();
    let instance: SWEBenchInstance = serde_json::from_value(serde_json::json!({
        "instance_id": "local-001", "repo": "unused-local-repository", "version": "fixture",
        "issue_title": "fixture", "issue_description": "fixture", "hints": [],
        "test_patch": "", "test_directives": ["git diff --check"], "difficulty": "Easy",
        "metrics": {"files_changed": 1, "lines_changed": 2, "test_count": 1,
            "dependencies": 0, "complexity_score": 0.1, "domain_specific": false,
            "requires_external_api": false}, "metadata": {}
    }))
    .unwrap();
    let evaluator = PatchEvaluator::new(swe_bench_adapter::EvalConfig {
        sandbox_enabled: false,
        ..Default::default()
    })
    .with_working_directory(directory.path())
    .unwrap();
    let patch = "diff --git a/example.txt b/example.txt\n--- a/example.txt\n+++ b/example.txt\n@@ -1 +1 @@\n-before\n+after\n";
    let result = evaluator.evaluate_patch(&instance, patch).await.unwrap();
    assert!(result.patch_applied && result.passed, "{:?}", result.errors);
    assert_eq!(result.metrics.files_modified, 1);
    assert_eq!(
        std::fs::read_to_string(directory.path().join("example.txt")).unwrap(),
        "after\n"
    );
    assert_eq!(
        std::fs::read_to_string(&sentinel).unwrap(),
        "existing application file"
    );
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 3);
}
