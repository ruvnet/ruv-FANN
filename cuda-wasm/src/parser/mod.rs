//! CUDA code parsing module

mod limits;
pub mod cuda_parser;
pub mod ptx_parser;
pub mod ast;
pub mod kernel_extractor;
pub mod lexer;

pub use cuda_parser::CudaParser;
pub use ast::{Ast, KernelDef, Statement, Expression};
pub use kernel_extractor::{extract_kernels, extract_kernel_by_name, KernelInfo};

/// Parse CUDA source code and return AST
pub fn parse(source: &str) -> crate::Result<Ast> {
    let parser = CudaParser::new();
    parser.parse(source)
}
