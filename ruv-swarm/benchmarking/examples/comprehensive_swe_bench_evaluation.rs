//! Inspect the local evaluation dataset without claiming unsupported trained-model evaluation.
use anyhow::Result;
use ruv_swarm_benchmarking::swe_bench_evaluator::SWEBenchInstances;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    let Some(path) = std::env::args_os().nth(1) else {
        println!("Usage: cargo run -p ruv-swarm-benchmarking --example comprehensive_swe_bench_evaluation -- DATASET.json");
        println!("Inspects local dataset records. Trained-model evaluation requires an implemented application backend and the official SWE-bench harness.");
        return Ok(());
    };
    let dataset = SWEBenchInstances::load_from_file(&PathBuf::from(path)).await?;
    println!("Loaded {} dataset records", dataset.instances.len());
    Ok(())
}
