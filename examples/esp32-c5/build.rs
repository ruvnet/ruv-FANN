use serde::Deserialize;
use std::{env, fs, path::PathBuf};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    schema: String,
    layers: [Layer; 2],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Layer {
    inputs: usize,
    outputs: usize,
    weights: Vec<f64>,
    bias: Option<Vec<f64>>,
    activation: String,
    steepness: f64,
}

fn q10(value: f64) -> Option<u16> {
    let scaled = value * 1024.0;
    ((0.0..=1024.0).contains(&scaled) && scaled.fract() == 0.0).then_some(scaled as u16)
}

fn compile_gate(model: Checkpoint) -> Result<(u16, u16), &'static str> {
    if model.schema != "ruv-fann.dense-layers/1" {
        return Err("expected ruv-fann.dense-layers/1");
    }
    let [hidden, output] = model.layers;
    if (hidden.inputs, hidden.outputs, output.inputs, output.outputs) != (2, 2, 2, 1)
        || hidden.activation != "ReLU"
        || output.activation != "Linear"
        || hidden.steepness != 1.0
        || output.steepness != 1.0
        || hidden.weights != [1.0, -1.0, 1.0, -1.0]
        || output.weights != [1.0, -1.0]
        || output.bias.as_deref().is_some_and(|bias| bias != [0.0])
    {
        return Err("expected the two hinge ReLU gate documented in the repository README");
    }
    let Some([first, second]) = hidden.bias.as_deref() else {
        return Err("expected exactly two hidden biases");
    };
    let margin = q10(-first).ok_or("first hinge must be an exact Q10 value in [0, 1]")?;
    let upper = q10(-second).ok_or("second hinge must be an exact Q10 value in [0, 1]")?;
    if upper <= margin {
        return Err("second hinge must be greater than the first hinge");
    }
    Ok((margin, upper - margin))
}

fn main() {
    println!("cargo:rerun-if-changed=model.json");
    if env::var("TARGET").as_deref() == Ok("riscv32imac-unknown-none-elf")
        && env::var_os("CARGO_FEATURE_FIRMWARE").is_some()
    {
        println!("cargo:rustc-link-arg-bin=ruvfann-c5=-Tlinkall.x");
    }

    let json = fs::read_to_string("model.json").expect("read model.json");
    let checkpoint = serde_json::from_str(&json).expect("parse RuFANN model.json");
    let (margin, cap) = compile_gate(checkpoint).expect("compile RuFANN C5 gate");
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo OUT_DIR"));
    fs::write(
        out.join("model.rs"),
        format!("pub const MARGIN_Q10: u16 = {margin};\npub const CAP_Q10: u16 = {cap};\n"),
    )
    .expect("write compiled model");
}
