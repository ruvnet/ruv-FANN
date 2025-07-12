use std::env;
use std::path::PathBuf;

fn main() {
    // Tell cargo to rerun this build script if these files change
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/wrapper.h");
    
    // Try to find OpenCV using pkg-config first
    if let Ok(opencv) = pkg_config::Config::new()
        .atleast_version("4.0")
        .probe("opencv4") {
        for path in &opencv.link_paths {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
        for lib in &opencv.libs {
            println!("cargo:rustc-link-lib={}", lib);
        }
    } else {
        // Fallback to manual configuration
        println!("cargo:rustc-link-lib=opencv_core");
        println!("cargo:rustc-link-lib=opencv_imgproc");
        println!("cargo:rustc-link-lib=opencv_imgcodecs");
        println!("cargo:rustc-link-lib=opencv_videoio");
        println!("cargo:rustc-link-lib=opencv_highgui");
        println!("cargo:rustc-link-lib=opencv_objdetect");
        println!("cargo:rustc-link-lib=opencv_features2d");
        println!("cargo:rustc-link-lib=opencv_calib3d");
        println!("cargo:rustc-link-lib=opencv_ml");
        
        #[cfg(feature = "cuda")]
        {
            println!("cargo:rustc-link-lib=opencv_cudaimgproc");
            println!("cargo:rustc-link-lib=opencv_cudaarithm");
            println!("cargo:rustc-link-lib=opencv_cudafeatures2d");
        }
    }

    // Generate bindings
    let bindings = bindgen::Builder::default()
        .header("src/wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}