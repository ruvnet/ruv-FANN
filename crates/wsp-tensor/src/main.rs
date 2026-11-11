use ruv_fann_wsp_tensor::{apply, fingerprint, inspect, read_json, rollback, Result};
use std::{fs::File, path::Path};
fn run() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let result=match a.get(1).map(String::as_str) {
        Some("inspect") if a.len()==3=>serde_json::json!({"checkpoint":inspect(&mut File::open(&a[2])?)?,"sha256":fingerprint(Path::new(&a[2]))?}),
        Some("apply") if a.len()==5=>serde_json::to_value(apply(Path::new(&a[2]),Path::new(&a[3]),&read_json(Path::new(&a[4]))?)?)?,
        Some("rollback") if a.len()==5=>{rollback(Path::new(&a[2]),Path::new(&a[3]),&read_json(Path::new(&a[4]))?)?;serde_json::json!({"restored":fingerprint(Path::new(&a[3]))?})},
        _=>return Err("usage: ruv-fann-wsp-tensor inspect FILE | apply BASE OUTPUT PATCH.json | rollback CANDIDATE OUTPUT RECEIPT.json".into())
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
