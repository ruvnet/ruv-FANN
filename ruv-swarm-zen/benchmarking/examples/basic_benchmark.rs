//! Basic benchmarking example
//!
//! This example demonstrates how to use the RUV-SWARM benchmarking framework
//! to compare baseline vs ML-optimized swarm performance.

use anyhow::Result;
use ruv_swarm_benchmarking::{
    BenchmarkConfig, BenchmarkScenario, BenchmarkingFramework, Difficulty, ExecutionMode,
};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("ruv_swarm_benchmarking=debug")
        .init();

    // Configure the benchmarking framework
    let config = BenchmarkConfig {
        database_path: PathBuf::from("benchmark_results.db"),
        enable_real_time_monitoring: true,
        monitor_port: 8080,
        claude_executable: PathBuf::from("claude"),
        execution_timeout: std::time::Duration::from_secs(900), // 15 minutes
        trial_count: 3,
    };

    // Create the benchmarking framework
    let mut framework = BenchmarkingFramework::new(config).await?;

    // Define test scenarios
    let scenarios = vec![
        BenchmarkScenario {
            name: "simple_bug_fix".to_string(),
            instance_id: "django__django-11099".to_string(),
            repository: "django/django".to_string(),
            issue_description: "Fix incorrect URL pattern in documentation".to_string(),
            difficulty: Difficulty::Easy,
            claude_command: "Fix the URL pattern error in Django documentation".to_string(),
            expected_files_modified: vec!["docs/ref/urls.txt".to_string()],
            validation_tests: vec!["test_docs_build".to_string()],
        },
        BenchmarkScenario {
            name: "feature_addition".to_string(),
            instance_id: "sympy__sympy-20639".to_string(),
            repository: "sympy/sympy".to_string(),
            issue_description: "Add support for Matrix.is_hermitian property".to_string(),
            difficulty: Difficulty::Medium,
            claude_command: "Add is_hermitian property to Matrix class with tests".to_string(),
            expected_files_modified: vec![
                "sympy/matrices/matrices.py".to_string(),
                "sympy/matrices/tests/test_matrices.py".to_string(),
            ],
            validation_tests: vec![
                "test_is_hermitian".to_string(),
                "test_hermitian_properties".to_string(),
            ],
        },
        BenchmarkScenario {
            name: "complex_refactor".to_string(),
            instance_id: "astropy__astropy-7746".to_string(),
            repository: "astropy/astropy".to_string(),
            issue_description: "Refactor coordinate transformations for performance".to_string(),
            difficulty: Difficulty::Hard,
            claude_command: "Refactor coordinate transformation system for better performance"
                .to_string(),
            expected_files_modified: vec![
                "astropy/coordinates/transformations.py".to_string(),
                "astropy/coordinates/builtin_frames.py".to_string(),
                "astropy/coordinates/tests/test_transformations.py".to_string(),
            ],
            validation_tests: vec![
                "test_transform_accuracy".to_string(),
                "test_transform_performance".to_string(),
            ],
        },
    ];

    println!(
        "Starting benchmark suite with {} scenarios",
        scenarios.len()
    );
    println!("Monitor available at: http://localhost:8080");

    // Run the benchmark suite
    let report = framework.run_benchmark_suite(scenarios).await?;

    // Print summary results
    println!("\n=== Benchmark Results ===");
    println!("Total scenarios: {}", report.summary.total_scenarios);
    println!("Successful: {}", report.summary.successful_scenarios);
    println!("Failed: {}", report.summary.failed_scenarios);
    println!(
        "Average improvement: {:.1}%",
        report.summary.average_improvement * 100.0
    );

    println!("\nKey findings:");
    for finding in &report.summary.key_findings {
        println!("  - {}", finding);
    }

    // Print detailed results for each scenario
    println!("\n=== Detailed Results ===");
    for result in &report.results {
        println!("\nScenario: {}", result.scenario.name);
        println!("  Baseline duration: {:?}", result.baseline.duration);
        println!(
            "  ML-optimized duration: {:?}",
            result.ml_optimized.duration
        );

        if let Some(speed_imp) = result.comparison.speed_improvement {
            println!("  Speed improvement: {:.1}%", speed_imp * 100.0);
        }

        if let Some(quality_imp) = result.comparison.quality_improvement {
            println!("  Quality improvement: {:.1}%", quality_imp * 100.0);
        }

        if let Some(resource_eff) = result.comparison.resource_efficiency {
            println!("  Resource efficiency: {:.1}%", resource_eff * 100.0);
        }

        println!(
            "  Statistical significance: p={:.4}",
            result.comparison.statistical_significance.p_value
        );
    }

    Ok(())
}
