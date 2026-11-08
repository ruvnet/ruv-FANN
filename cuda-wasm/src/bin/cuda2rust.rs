use clap::{Parser, Subcommand};
use cuda_rust_wasm::transpiler::CudaTranspiler;
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "cuda2rust")]
#[command(about = "CUDA to Rust/WASM transpiler")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Transpile CUDA code to Rust
    Transpile {
        /// Input CUDA file
        #[arg(short, long)]
        input: PathBuf,
        
        /// Output Rust file
        #[arg(short, long)]
        output: Option<PathBuf>,
        
        /// Enable optimizations
        #[arg(long)]
        optimize: bool,
        
        /// Enable pattern detection
        #[arg(long)]
        detect_patterns: bool,
        
        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },
    
    /// Generate WebGPU shader
    GenerateWgsl {
        /// Input CUDA file
        #[arg(short, long)]
        input: PathBuf,
        
        /// Output WGSL file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Transpile { input, output, optimize, detect_patterns, verbose } => {
            if verbose {
                env_logger::init();
            }
            
            let cuda_source = fs::read_to_string(&input)?;
            let transpiler = CudaTranspiler::new();
            
            let rust_code = transpiler.transpile(&cuda_source, optimize, detect_patterns)?;
            
            if let Some(output_path) = output {
                fs::write(&output_path, rust_code)?;
                println!("Transpiled CUDA to Rust: {} -> {}", input.display(), output_path.display());
            } else {
                println!("{rust_code}");
            }
        }
        
        Commands::GenerateWgsl { input, output } => {
            let cuda_source = fs::read_to_string(&input)?;
            let transpiler = CudaTranspiler::new();
            
            let wgsl_code = transpiler.generate_wgsl(&cuda_source)?;
            
            if let Some(output_path) = output {
                fs::write(&output_path, wgsl_code)?;
                println!("Generated WGSL shader: {} -> {}", input.display(), output_path.display());
            } else {
                println!("{wgsl_code}");
            }
        }
    }
    
    Ok(())
}