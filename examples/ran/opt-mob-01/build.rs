use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_file = "proto/handover.proto";
    let proto_dir = "proto";

    // Tell cargo to recompile if the proto file changes
    println!("cargo:rerun-if-changed={}", proto_file);

    // Check if the proto file exists
    if !Path::new(proto_file).exists() {
        eprintln!("Proto file not found: {}", proto_file);
        return Err("Proto file not found".into());
    }

    // Generate the gRPC code
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir("src/generated")
        .compile(&[proto_file], &[proto_dir])?;

    println!("Generated gRPC code successfully");
    Ok(())
}