use std::{env, path::PathBuf};

fn main() {
    for file in [
        "wrapper.h",
        "bridge.hpp",
        "mat.cpp",
        "algorithms.cpp",
        "vectors.cpp",
        "cuda.cpp",
    ] {
        println!("cargo:rerun-if-changed=src/{file}");
    }
    let target = env::var("TARGET").expect("Cargo TARGET");
    if target.starts_with("wasm32") {
        panic!("opencv-sys links native OpenCV 4; use opencv-wasm for browser targets");
    }
    let opencv = pkg_config::Config::new().atleast_version("4.4").cargo_metadata(false)
        .probe("opencv4").expect("OpenCV >= 4.4 development files required: install libopencv-dev, or set PKG_CONFIG_PATH to opencv4.pc");
    let cuda = env::var_os("CARGO_FEATURE_CUDA").is_some();
    if cuda
        && !opencv
            .include_paths
            .iter()
            .any(|p| p.join("opencv2/cudafilters.hpp").is_file())
    {
        panic!("opencv-sys feature cuda requires a CUDA-enabled OpenCV build with cudafilters; this system OpenCV does not provide it. Disable cuda or install that build.");
    }
    let mut cpp = cc::Build::new();
    cpp.cpp(true)
        .std("c++17")
        .include("src")
        .includes(&opencv.include_paths)
        .files([
            "src/mat.cpp",
            "src/algorithms.cpp",
            "src/vectors.cpp",
            "src/cuda.cpp",
        ]);
    if cuda {
        cpp.define("RUV_OPENCV_CUDA", None);
    }
    cpp.compile("ruv_opencv_bridge");
    // Put OpenCV after the static shim so linkers retain its dependencies.
    for path in &opencv.link_paths {
        println!("cargo:rustc-link-search=native={}", path.display());
    }
    for module in [
        "highgui",
        "videoio",
        "imgcodecs",
        "objdetect",
        "calib3d",
        "features2d",
        "flann",
        "ml",
        "imgproc",
        "core",
    ] {
        let library = format!("opencv_{module}");
        assert!(
            opencv.libs.contains(&library),
            "OpenCV module {library} is required"
        );
        println!("cargo:rustc-link-lib={library}");
    }
    if cuda {
        println!("cargo:rustc-link-lib=opencv_cudafilters");
    }
    let mut bindings = bindgen::Builder::default()
        .header("src/wrapper.h")
        .allowlist_function("cv_.*")
        .allowlist_type("cv_.*")
        .derive_default(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));
    if cuda {
        bindings = bindings.clang_arg("-DRUV_OPENCV_CUDA");
    }
    bindings
        .generate()
        .expect("libclang required to generate C ABI bindings")
        .write_to_file(PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs"))
        .expect("write generated bindings");
}
