fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir("src/proto")
        .compile(
            &["proto/cell_clustering.proto"],
            &["proto/"],
        )?;
    
    println!("cargo:rerun-if-changed=proto/cell_clustering.proto");
    
    Ok(())
}