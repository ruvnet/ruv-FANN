use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir(&out_dir)
        .compile_protos(&["proto/neural_service.proto"], &["proto/"])?;
    
    // Re-run if proto files change
    println!("cargo:rerun-if-changed=proto/neural_service.proto");
    
    Ok(())
}