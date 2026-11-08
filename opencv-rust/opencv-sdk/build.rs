use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=src/lib.rs");
    // Parse the two exported C functions directly. Resolving this whole crate
    // would recursively launch cargo metadata during packaged-source builds.
    let output = PathBuf::from(env::var("OUT_DIR").expect("Cargo OUT_DIR")).join("opencv-sdk.h");
    cbindgen::Builder::new()
        .with_src("src/lib.rs")
        .with_language(cbindgen::Language::C)
        .with_include_guard("RUV_OPENCV_SDK_H")
        .exclude_item("SDK_VERSION")
        .generate()
        .expect("Generate SDK C exports from source")
        .write_to_file(output);
}
