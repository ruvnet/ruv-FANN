//! Basic example of using the SWE-Bench adapter

use anyhow::Result;
use std::path::PathBuf;
use swe_bench_adapter::{
    ClaudePromptGenerator, DifficultyLevel, InstanceLoader, PatchEvaluator, SWEBenchAdapter,
    SWEBenchConfig, SWEBenchInstance, StreamParser,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("swe_bench_adapter=debug")
        .init();

    println!("SWE-Bench Adapter Example\n");

    // Example 1: Load and categorize instances
    example_instance_loading().await?;

    // Example 2: Generate prompts
    example_prompt_generation()?;

    // Example 3: Parse execution streams
    example_stream_parsing()?;

    // Example 4: Full evaluation pipeline
    example_full_evaluation().await?;

    Ok(())
}

async fn example_instance_loading() -> Result<()> {
    println!("=== Example 1: Loading SWE-Bench Instances ===\n");

    let mut loader = InstanceLoader::new("./swe-bench-instances")?;

    // Load a specific instance
    match loader.load_instance("test-001").await {
        Ok(instance) => {
            println!("Loaded instance: {}", instance.instance_id);
            println!("Repository: {}", instance.repo);
            println!("Difficulty: {}", instance.difficulty);
            println!("Files changed: {}", instance.metrics.files_changed);
            println!("Complexity score: {:.2}", instance.metrics.complexity_score);
        }
        Err(e) => {
            println!("Note: Instance not found ({}), creating mock instance", e);
            // The loader will create a mock instance for demonstration
        }
    }

    // Load instances with filtering
    let filter = swe_bench_adapter::loader::InstanceFilter {
        difficulty: Some(DifficultyLevel::Medium),
        repo: None,
        min_files: Some(2),
        max_files: Some(5),
        keywords: vec!["fix".to_string()],
    };

    let instances = loader.load_instances(Some(filter)).await?;
    println!("\nFiltered instances: {} found", instances.len());

    println!();
    Ok(())
}

fn example_prompt_generation() -> Result<()> {
    println!("=== Example 2: Prompt Generation ===\n");

    // Create a test instance
    let instance = create_test_instance();

    // Initialize prompt generator
    let config = swe_bench_adapter::PromptConfig::default();
    let generator = ClaudePromptGenerator::new(config);

    // Generate prompts for different difficulty levels
    for difficulty in [
        DifficultyLevel::Easy,
        DifficultyLevel::Medium,
        DifficultyLevel::Hard,
        DifficultyLevel::Expert,
    ] {
        let mut test_instance = instance.clone();
        test_instance.difficulty = difficulty;

        let prompt = generator.generate_prompt(&test_instance)?;
        println!("Prompt for {} difficulty:", difficulty);
        println!("- Template: {:?}", prompt.template_used);
        println!("- Token count: {}", prompt.token_count);
        println!(
            "- First 100 chars: {}...\n",
            &prompt.content[..100.min(prompt.content.len())]
        );
    }

    Ok(())
}

fn example_stream_parsing() -> Result<()> {
    println!("=== Example 3: Stream Parsing ===\n");

    let claude_output = r#"
Starting task execution for fixing memory leak...

<function_calls>
Read file: src/memory_manager.rs
Write file: src/memory_manager.rs  
Create file: tests/memory_test.rs
</function_calls>

File operations completed successfully.

Error: Failed to compile module - missing import
Warning: Deprecated API usage detected

Running tests...
<function_calls>
Read file: tests/memory_test.rs
</function_calls>

Test results: 3 passed, 1 failed

Tokens: 2567
Task completed with 4 tool calls.
"#;

    // Parse the output
    let mut parser = StreamParser::new();
    let metrics = parser.parse_stream(claude_output)?;

    println!("Stream parsing results:");
    println!("- Total tokens: {}", metrics.total_tokens);
    println!("- Tool calls: {}", metrics.tool_calls);
    println!("- File operations:");
    println!("  - Reads: {}", metrics.file_operations.reads);
    println!("  - Writes: {}", metrics.file_operations.writes);
    println!("  - Creates: {}", metrics.file_operations.creates);
    println!("  - Deletes: {}", metrics.file_operations.deletes);
    println!("- Errors: {:?}", metrics.errors);
    println!("- Warnings: {:?}", metrics.warnings);

    // Use metrics collector for aggregation
    use swe_bench_adapter::stream_parser::MetricsCollector;

    let mut collector = MetricsCollector::new();
    collector.parse_stream(claude_output)?;

    let summary = collector.get_summary();
    println!("\nMetrics summary:");
    println!(
        "- Average tokens per call: {:.2}",
        summary.avg_tokens_per_call
    );
    println!(
        "- File operation ratio: {:.2}",
        summary.file_operation_ratio
    );
    println!("- Error rate: {:.2}", summary.error_rate);
    println!("- Most common operation: {}", summary.most_common_operation);

    println!();
    Ok(())
}

async fn example_full_evaluation() -> Result<()> {
    println!("=== Example 4: Full Evaluation Pipeline ===\n");

    // Configure the adapter
    let config = SWEBenchConfig {
        instances_path: PathBuf::from("./swe-bench-instances"),
        memory_path: PathBuf::from("./swe-bench-memory"),
        prompt_config: swe_bench_adapter::PromptConfig {
            max_tokens: 4000,
            include_test_hints: true,
            include_context_files: true,
            template_style: swe_bench_adapter::PromptStyle::Standard,
        },
        eval_config: swe_bench_adapter::EvalConfig {
            timeout: std::time::Duration::from_secs(300),
            test_command: "pytest".to_string(),
            sandbox_enabled: false, // Disabled for example
            max_retries: 3,
        },
        benchmark_config: swe_bench_adapter::BenchmarkConfig {
            iterations: 5,
            warm_up: 2,
            measure_memory: true,
            profile_enabled: false,
        },
    };

    // Create adapter
    let adapter = SWEBenchAdapter::new(config).await?;

    // Create a mock agent
    use ruv_swarm_core::{agent::Agent, AgentId};

    // Note: In a real scenario, you would use an actual agent implementation
    // that implements the Agent trait

    // Note: In a real scenario, you would evaluate an actual instance
    println!("Full evaluation pipeline configured successfully!");
    println!("In production, you would call:");
    println!("  adapter.evaluate_instance(\"instance-id\", &agent).await?");
    println!("\nOr for batch evaluation:");
    println!("  adapter.evaluate_batch(instance_ids, agents, true).await?");

    // Get statistics (empty in this example)
    let stats = adapter.get_statistics().await?;
    println!("\nEvaluation statistics:");
    println!("- Total evaluations: {}", stats.total_evaluations);
    println!("- Success rate: {:.2}%", stats.success_rate * 100.0);

    Ok(())
}

fn create_test_instance() -> SWEBenchInstance {
    use std::collections::HashMap;
    use swe_bench_adapter::loader::InstanceMetrics;

    SWEBenchInstance {
        instance_id: "example-001".to_string(),
        repo: "python/cpython".to_string(),
        version: "3.10".to_string(),
        issue_title: "Fix memory leak in dict implementation".to_string(),
        issue_description: r#"
There is a memory leak in the dict implementation when using weak references.
The leak occurs when a weakref callback modifies the dict during garbage collection.

Steps to reproduce:
1. Create a dict with weak references
2. Trigger garbage collection
3. Observe memory usage increase

Expected: Memory should be properly freed
Actual: Memory usage increases with each cycle
"#
        .to_string(),
        hints: vec![
            "Check the weakref callback handling".to_string(),
            "Look at the reference counting logic".to_string(),
        ],
        test_patch: r#"diff --git a/Objects/dictobject.c b/Objects/dictobject.c
--- a/Objects/dictobject.c
+++ b/Objects/dictobject.c
@@ -1234,6 +1234,8 @@ insertdict(PyDictObject *mp, PyObject *key, Py_hash_t hash, PyObject *value)
     if (old_value != NULL) {
         Py_DECREF(old_value); /* which **CAN** re-enter */
+        /* Fix: Ensure proper cleanup after callback */
+        _PyDict_CheckConsistency(mp);
     }
     return 0;
 }"#
        .to_string(),
        test_directives: vec![
            "python -m pytest test_dict.py::test_weakref_leak".to_string(),
            "python -m pytest test_dict.py::test_gc_behavior".to_string(),
        ],
        difficulty: DifficultyLevel::Hard,
        metrics: InstanceMetrics {
            files_changed: 1,
            lines_changed: 15,
            test_count: 2,
            dependencies: 3,
            complexity_score: 0.75,
            domain_specific: false,
            requires_external_api: false,
        },
        metadata: HashMap::new(),
    }
}
