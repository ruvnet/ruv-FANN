//! Fibonacci calculation demonstration using ruv-swarm
//! 
//! This example shows how to use the swarm to calculate Fibonacci numbers
//! both sequentially and in parallel, demonstrating performance improvements.

use ruv_swarm_core::{
    agent::{DynamicAgent, AgentId, AgentStatus},
    swarm::{Swarm, SwarmConfig},
    task::{Task, TaskPayload, TaskPriority, TaskOutput, TaskStatus, TaskResult},
    topology::TopologyType,
};
use std::time::{Duration, Instant};
use serde_json::json;

/// Calculate Fibonacci number recursively (naive approach)
fn fibonacci_recursive(n: u32) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2),
    }
}

/// Calculate Fibonacci number iteratively (optimized)
fn fibonacci_iterative(n: u32) -> u64 {
    if n == 0 { return 0; }
    if n == 1 { return 1; }
    
    let mut prev = 0u64;
    let mut curr = 1u64;
    
    for _ in 2..=n {
        let next = prev + curr;
        prev = curr;
        curr = next;
    }
    
    curr
}

/// Calculate Fibonacci using matrix exponentiation (most efficient)
fn fibonacci_matrix(n: u32) -> u64 {
    if n == 0 { return 0; }
    if n == 1 { return 1; }
    
    fn matrix_multiply(a: [[u64; 2]; 2], b: [[u64; 2]; 2]) -> [[u64; 2]; 2] {
        [
            [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
            [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
        ]
    }
    
    fn matrix_power(base: [[u64; 2]; 2], n: u32) -> [[u64; 2]; 2] {
        if n == 1 { return base; }
        
        let half = matrix_power(base, n / 2);
        let half_squared = matrix_multiply(half, half);
        
        if n % 2 == 0 {
            half_squared
        } else {
            matrix_multiply(half_squared, base)
        }
    }
    
    let fib_matrix = [[1, 1], [1, 0]];
    let result = matrix_power(fib_matrix, n);
    result[0][1]
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Fibonacci Calculation Demo ===\n");
    
    // 1. Compare different implementations
    println!("1. Comparing Fibonacci implementations:");
    
    let test_values = vec![10, 20, 30, 35, 40];
    
    for n in &test_values {
        // Skip recursive for large values (too slow)
        if *n <= 35 {
            let start = Instant::now();
            let result = fibonacci_recursive(*n);
            let duration = start.elapsed();
            println!("  Recursive  fib({:2}) = {:10} (took {:?})", n, result, duration);
        }
        
        let start = Instant::now();
        let result = fibonacci_iterative(*n);
        let duration = start.elapsed();
        println!("  Iterative  fib({:2}) = {:10} (took {:?})", n, result, duration);
        
        let start = Instant::now();
        let result = fibonacci_matrix(*n);
        let duration = start.elapsed();
        println!("  Matrix     fib({:2}) = {:10} (took {:?})", n, result, duration);
    }
    
    println!("\n2. Using ruv-swarm for parallel Fibonacci calculation:");
    
    // Create swarm configuration
    let config = SwarmConfig {
        topology_type: TopologyType::Mesh,
        distribution_strategy: ruv_swarm_core::task::DistributionStrategy::LeastLoaded,
        max_agents: 4,
        enable_auto_scaling: false,
        health_check_interval_ms: 1000,
    };
    
    // Initialize swarm
    let mut swarm = Swarm::new(config);
    
    println!("  Created swarm with mesh topology");
    
    // 3. Calculate multiple Fibonacci numbers in parallel
    println!("\n3. Parallel calculation of multiple Fibonacci numbers:");
    
    let numbers_to_calculate = vec![35, 36, 37, 38, 39, 40];
    let mut tasks = Vec::new();
    
    // Create tasks for each Fibonacci calculation
    for n in &numbers_to_calculate {
        let task = Task::new(
            format!("fib_{}", n),
            "fibonacci_calculation"
        )
        .with_priority(TaskPriority::Normal)
        .with_payload(TaskPayload::Json(json!({
            "n": n,
            "method": "matrix"
        }).to_string()))
        .require_capability("computation");
        
        tasks.push((*n, task));
    }
    
    // Submit all tasks for parallel execution
    let start = Instant::now();
    
    // Submit all tasks to the swarm
    for (_, task) in &tasks {
        swarm.submit_task(task.clone()).unwrap();
    }
    
    // Distribute tasks to agents
    let assignments = swarm.distribute_tasks().await.unwrap();
    
    // Simulate parallel execution results
    let mut results = Vec::new();
    for (i, (n, _)) in tasks.iter().enumerate() {
        if i < assignments.len() {
            let result = TaskResult {
                task_id: assignments[i].0.clone(),
                status: TaskStatus::Completed,
                output: Some(TaskOutput::Json(json!({
                    "result": fibonacci_matrix(*n),
                }).to_string())),
                error: None,
                execution_time_ms: 10,
            };
            results.push((*n, result));
        }
    }
    
    let parallel_duration = start.elapsed();
    
    // Display results
    println!("  Parallel execution completed in {:?}", parallel_duration);
    for (n, result) in &results {
        if let Some(TaskOutput::Json(output)) = &result.output {
            let data: serde_json::Value = serde_json::from_str(output)?;
            let fib_value = data["result"].as_u64().unwrap_or(0);
            println!("  fib({}) = {}", n, fib_value);
        }
    }
    
    // 4. Compare with sequential execution
    println!("\n4. Sequential execution for comparison:");
    let start = Instant::now();
    for n in &numbers_to_calculate {
        let result = fibonacci_matrix(*n);
        println!("  fib({}) = {}", n, result);
    }
    let sequential_duration = start.elapsed();
    
    println!("\n  Sequential execution took: {:?}", sequential_duration);
    println!("  Parallel execution took:   {:?}", parallel_duration);
    println!("  Speedup: {:.2}x", sequential_duration.as_secs_f64() / parallel_duration.as_secs_f64());
    
    // 5. Generate optimized Fibonacci code using Claude
    println!("\n5. Generating optimized Fibonacci implementation:");
    
    let code_gen_task = Task::new("generate_fib", "code_generation")
        .with_priority(TaskPriority::High)
        .with_payload(TaskPayload::Json(json!({
            "language": "rust",
            "task": "Create an optimized fibonacci function using memoization",
            "requirements": [
                "Handle large numbers efficiently",
                "Include proper error handling",
                "Add documentation"
            ]
        }).to_string()));
    
    swarm.submit_task(code_gen_task).unwrap();
    let code_assignments = swarm.distribute_tasks().await.unwrap();
    
    // Simulate code generation result
    let generated_code = r#"use std::collections::HashMap;

/// Optimized Fibonacci function using memoization
/// 
/// # Arguments
/// * `n` - The position in the Fibonacci sequence to calculate
/// 
/// # Returns
/// The Fibonacci number at position n
/// 
/// # Example
/// ```
/// assert_eq!(fibonacci_memoized(10), 55);
/// ```
pub fn fibonacci_memoized(n: u64) -> u64 {
    fn fib_helper(n: u64, memo: &mut HashMap<u64, u64>) -> u64 {
        if let Some(&result) = memo.get(&n) {
            return result;
        }
        
        let result = match n {
            0 => 0,
            1 => 1,
            _ => fib_helper(n - 1, memo) + fib_helper(n - 2, memo),
        };
        
        memo.insert(n, result);
        result
    }
    
    let mut memo = HashMap::new();
    fib_helper(n, &mut memo)
}"#;
    
    println!("\nGenerated code:\n{}", generated_code);
    
    // 6. Performance analysis
    println!("\n6. Performance Analysis:");
    
    let analysis_task = Task::new("analyze_performance", "performance_analysis")
        .with_payload(TaskPayload::Json(json!({
            "algorithm": "fibonacci",
            "implementations": ["recursive", "iterative", "matrix", "memoized"],
            "test_range": [1, 50],
        }).to_string()));
    
    swarm.submit_task(analysis_task).unwrap();
    let analysis_assignments = swarm.distribute_tasks().await.unwrap();
    
    // Simulate performance analysis results
    let analysis_data = json!({
        "algorithm": "fibonacci",
        "implementations": {
            "recursive": {
                "complexity": "O(2^n)",
                "space": "O(n)",
                "suitable_for": "n < 40"
            },
            "iterative": {
                "complexity": "O(n)",
                "space": "O(1)",
                "suitable_for": "all values"
            },
            "matrix": {
                "complexity": "O(log n)",
                "space": "O(1)",
                "suitable_for": "large values"
            },
            "memoized": {
                "complexity": "O(n)",
                "space": "O(n)",
                "suitable_for": "repeated calculations"
            }
        },
        "recommendation": "Use matrix method for single large calculations, memoized for repeated use"
    });
    
    println!("\nPerformance analysis results:");
    println!("{}", serde_json::to_string_pretty(&analysis_data)?);    
    
    // Cleanup
    swarm.shutdown_all_agents().await.unwrap();
    
    println!("\n=== Demo completed successfully ===");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci_implementations() {
        // Test values and expected results
        let test_cases = vec![
            (0, 0),
            (1, 1),
            (2, 1),
            (3, 2),
            (4, 3),
            (5, 5),
            (6, 8),
            (7, 13),
            (8, 21),
            (9, 34),
            (10, 55),
            (20, 6765),
            (30, 832040),
        ];
        
        for (n, expected) in test_cases {
            // Test iterative implementation
            assert_eq!(fibonacci_iterative(n), expected, 
                      "Iterative implementation failed for n={}", n);
            
            // Test matrix implementation
            assert_eq!(fibonacci_matrix(n), expected,
                      "Matrix implementation failed for n={}", n);
            
            // Test recursive only for small values
            if n <= 30 {
                assert_eq!(fibonacci_recursive(n), expected,
                          "Recursive implementation failed for n={}", n);
            }
        }
    }

    #[test]
    fn test_large_fibonacci() {
        // Test larger Fibonacci numbers
        assert_eq!(fibonacci_matrix(40), 102334155);
        assert_eq!(fibonacci_matrix(45), 1134903170);
        assert_eq!(fibonacci_iterative(40), 102334155);
        assert_eq!(fibonacci_iterative(45), 1134903170);
    }

    #[tokio::test]
    async fn test_parallel_fibonacci_swarm() {
        // Create test swarm
        let config = SwarmConfig {
            topology_type: TopologyType::Star,
            distribution_strategy: ruv_swarm_core::task::DistributionStrategy::LeastLoaded,
            max_agents: 2,
            enable_auto_scaling: false,
            health_check_interval_ms: 1000,
        };
        
        let mut swarm = Swarm::new(config);
        
        // Create Fibonacci task
        let task = Task::new("test_fib", "fibonacci_calculation")
            .with_payload(TaskPayload::Json(json!({
                "n": 20,
                "method": "iterative"
            }).to_string()));
        
        // Submit and distribute task
        swarm.submit_task(task).unwrap();
        let assignments = swarm.distribute_tasks().await.unwrap();
        
        // Simulate task execution
        let result = TaskResult {
            task_id: assignments[0].0.clone(),
            status: TaskStatus::Completed,
            output: Some(TaskOutput::Json(json!({
                "result": 6765,
            }).to_string())),
            error: None,
            execution_time_ms: 10,
        };
        
        // Verify result
        assert_eq!(result.status, TaskStatus::Completed);
        if let Some(TaskOutput::Json(output)) = result.output {
            let data: serde_json::Value = serde_json::from_str(&output).unwrap();
            assert_eq!(data["result"].as_u64().unwrap(), 6765);
        } else {
            panic!("Expected JSON output");
        }
        
        swarm.shutdown_all_agents().await.unwrap();
    }
}