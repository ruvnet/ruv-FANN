use ruv_fann_wsp_tensor::{apply, fingerprint_file, inspect, read_json, rollback, Result};
use std::{ffi::OsString, fs::File, path::Path};

fn run() -> Result<()> {
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    let result = match args.first().and_then(|arg| arg.to_str()) {
        Some("inspect") if args.len() == 2 => {
            // Inspect and fingerprint the same open file. Reopening the path after
            // inspection could report a hash for a replacement file.
            let mut file = File::open(Path::new(&args[1]))?;
            let checkpoint = inspect(&mut file)?;
            let sha256 = fingerprint_file(&mut file)?;
            serde_json::json!({"checkpoint": checkpoint, "sha256": sha256})
        }
        Some("apply") if args.len() == 4 => serde_json::to_value(apply(
            Path::new(&args[1]),
            Path::new(&args[2]),
            &read_json(Path::new(&args[3]))?,
        )?)?,
        Some("rollback") if args.len() == 4 => {
            let restored = rollback(
                Path::new(&args[1]),
                Path::new(&args[2]),
                &read_json(Path::new(&args[3]))?,
            )?;
            serde_json::json!({"restored": restored})
        }
        _ => {
            return Err("usage: ruv-fann-wsp-tensor inspect FILE | apply BASE OUTPUT PATCH.json | rollback CANDIDATE OUTPUT RECEIPT.json".into());
        }
    };
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
