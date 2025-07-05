use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .file_descriptor_set_path(out_dir.join("ran_descriptor.bin"))
        .compile(
            &[
                "proto/common.proto",
                "proto/ml_core.proto",
                "proto/data_ingestion.proto",
                "proto/feature_engineering.proto",
                "proto/model_registry.proto",
                "proto/predictive_optimization.proto",
                "proto/service_assurance.proto",
                "proto/network_intelligence.proto",
            ],
            &["proto"],
        )?;
    
    Ok(())
}