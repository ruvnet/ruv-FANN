use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    
    tonic_build::configure()
        .file_descriptor_set_path(out_dir.join("scell_manager_descriptor.bin"))
        .compile(&["proto/scell_manager.proto"], &["proto"])?;
    
    Ok(())
}