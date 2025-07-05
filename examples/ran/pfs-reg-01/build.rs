fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir("src/generated")
        .compile(
            &["proto/model_registry.proto"],
            &["proto"],
        )?;

    // Create the generated directory if it doesn't exist
    std::fs::create_dir_all("src/generated")?;

    Ok(())
}