use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?)
        .join("../../shared/proto/proto");
    
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir("src/proto")
        .compile(
            &[
                proto_path.join("service_assurance.proto"),
                proto_path.join("common.proto"),
            ],
            &[proto_path],
        )?;
    
    Ok(())
}