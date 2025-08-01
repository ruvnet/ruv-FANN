//! Benchmarks for SWE-Bench adapter

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;
use swe_bench_adapter::{
    ClaudePromptGenerator, DifficultyLevel, InstanceLoader, PatchEvaluator, SWEBenchAdapter,
    SWEBenchConfig, SWEBenchInstance, StreamParser,
};

fn create_test_instance() -> SWEBenchInstance {
    use std::collections::HashMap;
    use swe_bench_adapter::loader::InstanceMetrics;

    SWEBenchInstance {
        instance_id: "bench-001".to_string(),
        repo: "test/repo".to_string(),
        version: "1.0.0".to_string(),
        issue_title: "Benchmark test issue".to_string(),
        issue_description: "This is a test instance for benchmarking purposes. It contains a complex issue that requires multiple file changes and sophisticated problem-solving skills.".to_string(),
        hints: vec![
            "Check the main module".to_string(),
            "Consider edge cases".to_string(),
        ],
        test_patch: r#"diff --git a/src/lib.rs b/src/lib.rs
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -10,6 +10,8 @@
 pub fn process_data(input: &str) -> Result<String> {
     let parsed = parse_input(input)?;
+    // Fix: Add validation
+    validate_data(&parsed)?;
     transform_data(parsed)
 }"#.to_string(),
        test_directives: vec![
            "cargo test test_process_data".to_string(),
            "cargo test test_edge_cases".to_string(),
        ],
        difficulty: DifficultyLevel::Medium,
        metrics: InstanceMetrics {
            files_changed: 3,
            lines_changed: 75,
            test_count: 5,
            dependencies: 4,
            complexity_score: 0.65,
            domain_specific: false,
            requires_external_api: false,
        },
        metadata: HashMap::new(),
    }
}

fn benchmark_instance_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("instance_loading");
    group.measurement_time(Duration::from_secs(10));

    let runtime = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("load_single_instance", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let mut loader = InstanceLoader::new("./bench-instances").unwrap();
                let instance = loader.load_instance(black_box("bench-001")).await;
                instance
            })
        })
    });

    group.bench_function("load_multiple_instances", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let mut loader = InstanceLoader::new("./bench-instances").unwrap();
                let instances = loader.load_instances(None).await;
                instances
            })
        })
    });

    group.finish();
}

fn benchmark_prompt_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("prompt_generation");

    let config = swe_bench_adapter::PromptConfig::default();
    let generator = ClaudePromptGenerator::new(config);
    let instance = create_test_instance();

    group.bench_function("generate_simple_prompt", |b| {
        let easy_instance = SWEBenchInstance {
            difficulty: DifficultyLevel::Easy,
            ..instance.clone()
        };
        b.iter(|| generator.generate_prompt(black_box(&easy_instance)))
    });

    group.bench_function("generate_expert_prompt", |b| {
        let expert_instance = SWEBenchInstance {
            difficulty: DifficultyLevel::Expert,
            ..instance.clone()
        };
        b.iter(|| generator.generate_prompt(black_box(&expert_instance)))
    });

    group.bench_function("generate_batch_prompts", |b| {
        let instances = vec![instance.clone(); 10];
        b.iter(|| generator.generate_batch(black_box(&instances)))
    });

    group.finish();
}

fn benchmark_stream_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("stream_parsing");

    let test_output = r#"
Starting task execution...
<function_calls>
Read file: src/main.rs
</function_calls>
File read successfully.
<function_calls>
Write file: src/lib.rs
</function_calls>
File written successfully.
Error: Failed to compile module
Warning: Deprecated function used
Tokens: 1234
Task completed with 2 tool calls.
"#;

    group.bench_function("parse_small_stream", |b| {
        let mut parser = StreamParser::new();
        b.iter(|| parser.parse_stream(black_box(test_output)))
    });

    group.bench_function("parse_large_stream", |b| {
        let large_output = test_output.repeat(100);
        let mut parser = StreamParser::new();
        b.iter(|| parser.parse_stream(black_box(&large_output)))
    });

    group.bench_function("parse_with_many_tool_calls", |b| {
        let mut output = String::new();
        for i in 0..50 {
            output.push_str(&format!(
                "<function_calls>\nRead file: file{}.rs\n</function_calls>\n",
                i
            ));
        }
        let mut parser = StreamParser::new();
        b.iter(|| parser.parse_stream(black_box(&output)))
    });

    group.finish();
}

fn benchmark_patch_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("patch_evaluation");
    group.measurement_time(Duration::from_secs(20));

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let config = swe_bench_adapter::EvalConfig::default();
    let evaluator = PatchEvaluator::new(config);
    let instance = create_test_instance();

    let test_patch = r#"
diff --git a/src/main.rs b/src/main.rs
--- a/src/main.rs
+++ b/src/main.rs
@@ -1,5 +1,6 @@
 fn main() {
+    // Added initialization
     println!("Hello, world!");
 }
"#;

    group.bench_function("evaluate_simple_patch", |b| {
        b.iter(|| {
            runtime.block_on(async {
                evaluator
                    .evaluate_patch(black_box(&instance), black_box(test_patch))
                    .await
            })
        })
    });

    group.bench_function("patch_comparison", |b| {
        let patch1 = test_patch;
        let patch2 = test_patch.replace("initialization", "setup");
        b.iter(|| evaluator.compare_patches(black_box(patch1), black_box(patch2)))
    });

    group.finish();
}

fn benchmark_full_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_evaluation");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    let runtime = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("end_to_end_evaluation", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let config = SWEBenchConfig::default();
                let adapter = SWEBenchAdapter::new(config).await.unwrap();

                // Note: In a real benchmark, you would use an actual agent implementation
                // For now, this is commented out as it requires a concrete Agent implementation

                // adapter.evaluate_instance(black_box("test-001"), black_box(&agent)).await
                Ok(())
            })
        })
    });

    group.finish();
}

fn benchmark_metrics_collection(c: &mut Criterion) {
    use swe_bench_adapter::stream_parser::MetricsCollector;

    let mut group = c.benchmark_group("metrics_collection");

    let test_streams = vec![
        "Simple output with no tool calls",
        "<function_calls>\nRead file\n</function_calls>\nDone.",
        "Error: Something went wrong\nWarning: Check this",
    ];

    group.bench_function("collect_metrics_single_stream", |b| {
        let mut collector = MetricsCollector::new();
        b.iter(|| {
            for stream in &test_streams {
                let _ = collector.parse_stream(black_box(stream));
            }
        })
    });

    group.bench_function("get_metrics_summary", |b| {
        let mut collector = MetricsCollector::new();
        for stream in &test_streams {
            let _ = collector.parse_stream(stream);
        }
        b.iter(|| collector.get_summary())
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_instance_loading,
    benchmark_prompt_generation,
    benchmark_stream_parsing,
    benchmark_patch_evaluation,
    benchmark_full_evaluation,
    benchmark_metrics_collection
);
criterion_main!(benches);
